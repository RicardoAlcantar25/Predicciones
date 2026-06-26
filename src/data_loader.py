"""
data_loader.py — Descarga y preprocesa el dataset de resultados internacionales.

Fuente: https://github.com/martj42/international_results
"""

import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
from src.utils import get_tournament_weight

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATA_FILE = os.path.join(DATA_DIR, "results.csv")


def download_data(force: bool = False) -> str:
    """Descarga el CSV de resultados si no existe localmente."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_FILE) or force:
        print("Descargando dataset de resultados internacionales...")
        resp = requests.get(DATA_URL, timeout=60)
        resp.raise_for_status()
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            f.write(resp.text)
        print(f"Dataset guardado en {DATA_FILE}")
    return DATA_FILE


def load_pinnacle_odds() -> pd.DataFrame:
    """Carga el CSV de cuotas de Pinnacle si existe en data/pinnacle_odds.csv."""
    odds_file = os.path.join(DATA_DIR, "pinnacle_odds.csv")
    if os.path.exists(odds_file):
        try:
            odds_df = pd.read_csv(odds_file)
            odds_df["date"] = pd.to_datetime(odds_df["date"])
            required_cols = ["date", "home_team", "away_team", "pinnacle_odds_home", "pinnacle_odds_draw", "pinnacle_odds_away"]
            if all(col in odds_df.columns for col in required_cols):
                return odds_df[required_cols].drop_duplicates(subset=["date", "home_team", "away_team"])
            else:
                print("El archivo pinnacle_odds.csv no contiene las columnas requeridas.")
        except Exception as e:
            print(f"Error al leer pinnacle_odds.csv: {e}")
    return pd.DataFrame(columns=["date", "home_team", "away_team", "pinnacle_odds_home", "pinnacle_odds_draw", "pinnacle_odds_away"])


def load_and_preprocess(min_year: int = 2018) -> pd.DataFrame:
    """
    Carga el dataset y aplica preprocesamiento:
    - Filtra partidos desde min_year
    - Calcula peso por torneo
    - Aplica decay temporal exponencial
    - Crea columnas auxiliares
    """
    filepath = download_data()
    df = pd.read_csv(filepath)

    # Parsear fechas
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    # Filtrar por año mínimo
    df = df[df["year"] >= min_year].copy()

    # Eliminar partidos que no se han jugado (marcador NaN)
    df = df.dropna(subset=["home_score", "away_score"]).copy()

    # Peso por torneo
    df["tournament_weight"] = df["tournament"].apply(get_tournament_weight)

    # Decay temporal: partidos más recientes pesan más
    # Half-life de 2 años (~730 días)
    today = pd.Timestamp(datetime.now())
    days_ago = (today - df["date"]).dt.days
    half_life = 730  # días
    df["time_decay"] = np.exp(-np.log(2) * days_ago / half_life)

    # Peso combinado
    df["weight"] = df["tournament_weight"] * df["time_decay"]

    # Convertir neutral a booleano
    df["neutral"] = df["neutral"].astype(str).str.upper() == "TRUE"

    # Total de goles
    df["total_goals"] = df["home_score"] + df["away_score"]

    # Resultado categórico
    df["result"] = np.where(
        df["home_score"] > df["away_score"], "home_win",
        np.where(df["home_score"] < df["away_score"], "away_win", "draw")
    )

    # Resetear índice
    df = df.reset_index(drop=True)

    # Cargar datos avanzados del JSON
    from src.utils import load_advanced_data, get_advanced_global_defaults, get_team_advanced_stats
    adv_data = load_advanced_data()
    defaults = get_advanced_global_defaults(adv_data)

    # Precalcular estadísticas avanzadas para todos los equipos únicos
    unique_teams = pd.concat([df["home_team"], df["away_team"]]).unique()
    team_stats_cache = {
        team: get_team_advanced_stats(team, adv_data, defaults)
        for team in unique_teams
    }

    # Combinar variables para Home y Away de forma vectorial
    df["home_valor_plantilla_mde"] = df["home_team"].map(lambda t: team_stats_cache[t]["valor_plantilla_mde"])
    df["home_promedio_xg_eliminatorias"] = df["home_team"].map(lambda t: team_stats_cache[t]["promedio_xg_eliminatorias"])
    df["home_tarjetas_amarillas_por_partido"] = df["home_team"].map(lambda t: team_stats_cache[t]["tarjetas_amarillas_por_partido"])
    df["home_dependencia_estrella"] = df["home_team"].map(lambda t: team_stats_cache[t]["dependencia_estrella_porcentaje"])

    df["away_valor_plantilla_mde"] = df["away_team"].map(lambda t: team_stats_cache[t]["valor_plantilla_mde"])
    df["away_promedio_xg_eliminatorias"] = df["away_team"].map(lambda t: team_stats_cache[t]["promedio_xg_eliminatorias"])
    df["away_tarjetas_amarillas_por_partido"] = df["away_team"].map(lambda t: team_stats_cache[t]["tarjetas_amarillas_por_partido"])
    df["away_dependencia_estrella"] = df["away_team"].map(lambda t: team_stats_cache[t]["dependencia_estrella_porcentaje"])

    # Mapear altitud de la sede
    host_altitudes = adv_data.get("sedes_mundial_altitud", {})
    def get_altitude(city):
        if not isinstance(city, str):
            return 0.0
        norm_city = city.lower().replace("_", " ").strip()
        for k, v in host_altitudes.items():
            if k.lower().replace("_", " ").strip() == norm_city:
                return float(v)
        # Traducción de "Mexico City" -> "CDMX" y "New York" -> "New_York"
        if norm_city in ["mexico city", "ciudad de mexico"]:
            return float(host_altitudes.get("CDMX", 2240))
        if norm_city == "new york":
            return float(host_altitudes.get("New_York", 10))
        if norm_city == "kansas city":
            return float(host_altitudes.get("Kansas_City", 277))
        if norm_city == "los angeles":
            return float(host_altitudes.get("Los_Angeles", 71))
        if norm_city == "san francisco":
            return float(host_altitudes.get("San_Francisco", 16))
        return 0.0

    df["altitude"] = df["city"].apply(get_altitude)

    # Ingesta de cuotas reales si existen, si no se rellenan con NaN
    odds_df = load_pinnacle_odds()
    if not odds_df.empty:
        df = pd.merge(df, odds_df, on=["date", "home_team", "away_team"], how="left")
    else:
        df["pinnacle_odds_home"] = np.nan
        df["pinnacle_odds_draw"] = np.nan
        df["pinnacle_odds_away"] = np.nan

    print(f"Dataset cargado: {len(df)} partidos desde {min_year}")
    return df


def get_team_index(df: pd.DataFrame) -> dict:
    """
    Crea un mapeo equipo -> índice numérico para el modelo bayesiano.
    """
    all_teams = sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))
    return {team: idx for idx, team in enumerate(all_teams)}


def get_teams_list(df: pd.DataFrame) -> list:
    """Devuelve lista ordenada de todos los equipos en el dataset."""
    return sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))


def prepare_bayesian_data(df: pd.DataFrame) -> dict:
    """
    Prepara los datos en formato para PyMC.
    Returns dict con arrays numéricos.
    """
    team_index = get_team_index(df)
    n_teams = len(team_index)

    home_team_idx = df["home_team"].map(team_index).values
    away_team_idx = df["away_team"].map(team_index).values
    home_goals = df["home_score"].values.astype(int)
    away_goals = df["away_score"].values.astype(int)
    weights = df["weight"].values
    is_neutral = df["neutral"].values.astype(int)

    return {
        "home_team_idx": home_team_idx,
        "away_team_idx": away_team_idx,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "weights": weights,
        "is_neutral": is_neutral,
        "n_teams": n_teams,
        "n_matches": len(df),
        "team_index": team_index,
        "index_to_team": {v: k for k, v in team_index.items()},
    }
