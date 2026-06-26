"""
feature_engineering.py — Genera features para el modelo XGBoost.

Calcula estadísticas históricas por equipo:
- Media de goles anotados/recibidos
- Win rate
- Racha actual
- Head-to-head
"""

import pandas as pd
import numpy as np


def compute_team_stats(df: pd.DataFrame, team: str, n_last: int = 20) -> dict:
    """
    Calcula estadísticas agregadas de un equipo basadas en sus últimos n_last partidos.
    """
    # Partidos como local
    home_matches = df[df["home_team"] == team].copy()
    home_matches["goals_for"] = home_matches["home_score"]
    home_matches["goals_against"] = home_matches["away_score"]
    home_matches["is_home"] = 1
    home_matches["win"] = (home_matches["home_score"] > home_matches["away_score"]).astype(int)
    home_matches["draw"] = (home_matches["home_score"] == home_matches["away_score"]).astype(int)
    home_matches["loss"] = (home_matches["home_score"] < home_matches["away_score"]).astype(int)

    # Partidos como visitante
    away_matches = df[df["away_team"] == team].copy()
    away_matches["goals_for"] = away_matches["away_score"]
    away_matches["goals_against"] = away_matches["home_score"]
    away_matches["is_home"] = 0
    away_matches["win"] = (away_matches["away_score"] > away_matches["home_score"]).astype(int)
    away_matches["draw"] = (away_matches["away_score"] == away_matches["home_score"]).astype(int)
    away_matches["loss"] = (away_matches["away_score"] < away_matches["home_score"]).astype(int)

    # Combinar y ordenar por fecha
    all_matches = pd.concat([
        home_matches[["date", "goals_for", "goals_against", "is_home", "win", "draw", "loss", "weight"]],
        away_matches[["date", "goals_for", "goals_against", "is_home", "win", "draw", "loss", "weight"]],
    ]).sort_values("date", ascending=False)

    # Últimos N partidos
    recent = all_matches.head(n_last)

    if len(recent) == 0:
        return _empty_stats()

    # Estadísticas ponderadas
    w = recent["weight"].values
    w_sum = w.sum() if w.sum() > 0 else 1.0

    stats = {
        "avg_goals_scored": np.average(recent["goals_for"].values, weights=w),
        "avg_goals_conceded": np.average(recent["goals_against"].values, weights=w),
        "win_rate": np.average(recent["win"].values, weights=w),
        "draw_rate": np.average(recent["draw"].values, weights=w),
        "loss_rate": np.average(recent["loss"].values, weights=w),
        "total_matches": len(all_matches),
        "recent_matches": len(recent),
        "avg_goal_diff": np.average(
            (recent["goals_for"] - recent["goals_against"]).values, weights=w
        ),
    }

    # Racha actual (victorias o derrotas consecutivas)
    streak = 0
    streak_type = None
    for _, row in recent.iterrows():
        if streak_type is None:
            if row["win"] == 1:
                streak_type = "W"
                streak = 1
            elif row["loss"] == 1:
                streak_type = "L"
                streak = 1
            else:
                streak_type = "D"
                streak = 1
        elif (streak_type == "W" and row["win"] == 1) or \
             (streak_type == "L" and row["loss"] == 1) or \
             (streak_type == "D" and row["draw"] == 1):
            streak += 1
        else:
            break

    stats["streak"] = streak if streak_type == "W" else (-streak if streak_type == "L" else 0)

    return stats


def _empty_stats() -> dict:
    """Estadísticas vacías para equipos sin datos."""
    return {
        "avg_goals_scored": 1.0,
        "avg_goals_conceded": 1.0,
        "win_rate": 0.33,
        "draw_rate": 0.33,
        "loss_rate": 0.33,
        "total_matches": 0,
        "recent_matches": 0,
        "avg_goal_diff": 0.0,
        "streak": 0,
    }


def compute_head_to_head(df: pd.DataFrame, team_a: str, team_b: str,
                         n_last: int = 10) -> dict:
    """
    Estadísticas head-to-head entre dos equipos.
    """
    mask_ab = (df["home_team"] == team_a) & (df["away_team"] == team_b)
    mask_ba = (df["home_team"] == team_b) & (df["away_team"] == team_a)
    h2h = df[mask_ab | mask_ba].sort_values("date", ascending=False).head(n_last)

    if len(h2h) == 0:
        return {
            "h2h_matches": 0,
            "h2h_a_wins": 0,
            "h2h_b_wins": 0,
            "h2h_draws": 0,
            "h2h_a_goals_avg": 0.0,
            "h2h_b_goals_avg": 0.0,
        }

    a_wins = 0
    b_wins = 0
    draws = 0
    a_goals = []
    b_goals = []

    for _, row in h2h.iterrows():
        if row["home_team"] == team_a:
            a_g, b_g = row["home_score"], row["away_score"]
        else:
            a_g, b_g = row["away_score"], row["home_score"]
        a_goals.append(a_g)
        b_goals.append(b_g)
        if a_g > b_g:
            a_wins += 1
        elif b_g > a_g:
            b_wins += 1
        else:
            draws += 1

    return {
        "h2h_matches": len(h2h),
        "h2h_a_wins": a_wins,
        "h2h_b_wins": b_wins,
        "h2h_draws": draws,
        "h2h_a_goals_avg": np.mean(a_goals),
        "h2h_b_goals_avg": np.mean(b_goals),
    }


def build_features_for_match(df: pd.DataFrame, home_team: str, away_team: str,
                             is_neutral: bool = True, altitude: float = 0.0,
                             home_starting_val: float = None, away_starting_val: float = None,
                             home_lineup: list = None, away_lineup: list = None) -> dict:
    """
    Construye el vector de features para una predicción de partido.
    """
    home_stats = compute_team_stats(df, home_team)
    away_stats = compute_team_stats(df, away_team)
    h2h = compute_head_to_head(df, home_team, away_team)

    from src.utils import load_advanced_data, get_advanced_global_defaults, get_team_advanced_stats
    adv_data = load_advanced_data()
    defaults = get_advanced_global_defaults(adv_data)
    h_adv = get_team_advanced_stats(home_team, adv_data, defaults)
    a_adv = get_team_advanced_stats(away_team, adv_data, defaults)

    h_val = home_starting_val if home_starting_val is not None else h_adv["valor_plantilla_mde"]
    a_val = away_starting_val if away_starting_val is not None else a_adv["valor_plantilla_mde"]

    h_dep = h_adv.get("dependencia_estrella_porcentaje", 0.0)
    h_star = h_adv.get("jugador_estrella_nombre", None)
    a_dep = a_adv.get("dependencia_estrella_porcentaje", 0.0)
    a_star = a_adv.get("jugador_estrella_nombre", None)

    import unicodedata
    def clean_p(s):
        s = unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode("utf-8")
        return s.lower().replace("_", " ").strip()

    if home_lineup and h_star:
        clean_star = clean_p(h_star)
        star_words = [w.strip(".") for w in clean_star.split() if len(w.strip(".")) > 2]
        if not star_words:
            star_words = [clean_star]
        is_star_present = any(any(w in clean_p(p) for w in star_words) for p in home_lineup)
        if not is_star_present:
            h_val = h_val * (1.0 - (h_dep / 100.0))

    if away_lineup and a_star:
        clean_star = clean_p(a_star)
        star_words = [w.strip(".") for w in clean_star.split() if len(w.strip(".")) > 2]
        if not star_words:
            star_words = [clean_star]
        is_star_present = any(any(w in clean_p(p) for w in star_words) for p in away_lineup)
        if not is_star_present:
            a_val = a_val * (1.0 - (a_dep / 100.0))

    features = {
        # Home team stats
        "home_avg_goals_scored": home_stats["avg_goals_scored"],
        "home_avg_goals_conceded": home_stats["avg_goals_conceded"],
        "home_win_rate": home_stats["win_rate"],
        "home_goal_diff": home_stats["avg_goal_diff"],
        "home_streak": home_stats["streak"],
        # Away team stats
        "away_avg_goals_scored": away_stats["avg_goals_scored"],
        "away_avg_goals_conceded": away_stats["avg_goals_conceded"],
        "away_win_rate": away_stats["win_rate"],
        "away_goal_diff": away_stats["avg_goal_diff"],
        "away_streak": away_stats["streak"],
        # Head to head
        "h2h_home_win_pct": h2h["h2h_a_wins"] / max(h2h["h2h_matches"], 1),
        "h2h_away_win_pct": h2h["h2h_b_wins"] / max(h2h["h2h_matches"], 1),
        "h2h_avg_goals_home": h2h["h2h_a_goals_avg"],
        "h2h_avg_goals_away": h2h["h2h_b_goals_avg"],
        # Match context
        "is_neutral": int(is_neutral),
        # Derived
        "attack_diff": home_stats["avg_goals_scored"] - away_stats["avg_goals_scored"],
        "defense_diff": away_stats["avg_goals_conceded"] - home_stats["avg_goals_conceded"],
        # Advanced features
        "home_valor_plantilla_mde": h_val,
        "home_promedio_xg_eliminatorias": h_adv["promedio_xg_eliminatorias"],
        "home_tarjetas_amarillas_por_partido": h_adv["tarjetas_amarillas_por_partido"],
        "home_dependencia_estrella": h_dep,
        "away_valor_plantilla_mde": a_val,
        "away_promedio_xg_eliminatorias": a_adv["promedio_xg_eliminatorias"],
        "away_tarjetas_amarillas_por_partido": a_adv["tarjetas_amarillas_por_partido"],
        "away_dependencia_estrella": a_dep,
        "altitude": altitude,
    }

    return features


def build_training_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye el dataset de entrenamiento para XGBoost.
    Para cada partido calcula las features basadas en datos ANTERIORES al partido.
    """
    rows = []
    df_sorted = df.sort_values("date").reset_index(drop=True)

    # Usar ventana de datos previos para cada partido
    for i in range(50, len(df_sorted)):  # Empezamos en 50 para tener historial
        row = df_sorted.iloc[i]
        # Datos disponibles hasta antes de este partido
        df_prior = df_sorted.iloc[:i]

        home_stats = compute_team_stats(df_prior, row["home_team"], n_last=15)
        away_stats = compute_team_stats(df_prior, row["away_team"], n_last=15)

        # Solo incluir si ambos equipos tienen suficiente historial
        if home_stats["recent_matches"] < 3 or away_stats["recent_matches"] < 3:
            continue

        h2h = compute_head_to_head(df_prior, row["home_team"], row["away_team"], n_last=8)

        feature_row = {
            "home_avg_goals_scored": home_stats["avg_goals_scored"],
            "home_avg_goals_conceded": home_stats["avg_goals_conceded"],
            "home_win_rate": home_stats["win_rate"],
            "home_goal_diff": home_stats["avg_goal_diff"],
            "home_streak": home_stats["streak"],
            "away_avg_goals_scored": away_stats["avg_goals_scored"],
            "away_avg_goals_conceded": away_stats["avg_goals_conceded"],
            "away_win_rate": away_stats["win_rate"],
            "away_goal_diff": away_stats["avg_goal_diff"],
            "away_streak": away_stats["streak"],
            "h2h_home_win_pct": h2h["h2h_a_wins"] / max(h2h["h2h_matches"], 1),
            "h2h_away_win_pct": h2h["h2h_b_wins"] / max(h2h["h2h_matches"], 1),
            "h2h_avg_goals_home": h2h["h2h_a_goals_avg"],
            "h2h_avg_goals_away": h2h["h2h_b_goals_avg"],
            "is_neutral": int(row["neutral"]),
            "attack_diff": home_stats["avg_goals_scored"] - away_stats["avg_goals_scored"],
            "defense_diff": away_stats["avg_goals_conceded"] - home_stats["avg_goals_conceded"],
            # Advanced features
            "home_valor_plantilla_mde": row["home_valor_plantilla_mde"],
            "home_promedio_xg_eliminatorias": row["home_promedio_xg_eliminatorias"],
            "home_tarjetas_amarillas_por_partido": row["home_tarjetas_amarillas_por_partido"],
            "home_dependencia_estrella": row["home_dependencia_estrella"],
            "away_valor_plantilla_mde": row["away_valor_plantilla_mde"],
            "away_promedio_xg_eliminatorias": row["away_promedio_xg_eliminatorias"],
            "away_tarjetas_amarillas_por_partido": row["away_tarjetas_amarillas_por_partido"],
            "away_dependencia_estrella": row["away_dependencia_estrella"],
            "altitude": row["altitude"],
            # Targets
            "home_goals": row["home_score"],
            "away_goals": row["away_score"],
            "weight": row["weight"],
        }
        rows.append(feature_row)

    result = pd.DataFrame(rows)
    print(f"Dataset de entrenamiento construido: {len(result)} muestras")
    return result
