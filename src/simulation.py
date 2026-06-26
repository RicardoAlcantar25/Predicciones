"""
simulation.py — Motor de simulación para el Mundial 2026.

Simula el formato real del Mundial 2026:
- 12 grupos de 4 equipos (Grupos A-L).
- Pasan los 2 mejores de cada grupo (24 equipos) + los 8 mejores terceros (8 equipos).
- Fase de eliminación directa desde Dieciseisavos de Final (R32) hasta la Final.
"""

import numpy as np
import pandas as pd
from src.utils import WORLD_CUP_2026_GROUPS


def simulate_match(model, home_team: str, away_team: str, df: pd.DataFrame,
                   model_type: str = "xgboost", neutral: bool = True) -> tuple:
    """
    Simula los goles de un partido usando el modelo seleccionado.
    Usa una distribución de Poisson basada en las lambdas (xG) predichas.
    
    Returns:
        tuple: (home_goals, away_goals)
    """
    try:
        if model_type == "bayesian":
            pred = model.predict(home_team, away_team, neutral=neutral)
        else:  # xgboost
            pred = model.predict(df, home_team, away_team, neutral=neutral)
        
        lambda_home = pred["xg_home"]
        lambda_away = pred["xg_away"]
    except Exception as e:
        # Fallback si el equipo no está en el dataset de entrenamiento
        print(f"Warning simulating {home_team} vs {away_team}: {e}. Usando lambda default de 1.2")
        lambda_home = 1.2
        lambda_away = 1.1

    # Generar goles aleatorios usando distribución de Poisson
    home_goals = int(np.random.poisson(lambda_home))
    away_goals = int(np.random.poisson(lambda_away))
    
    return home_goals, away_goals


def simulate_group_stage(model, df: pd.DataFrame, model_type: str = "xgboost") -> dict:
    """
    Simula la fase de grupos completa.
    Calcula puntos, diferencia de goles y goles a favor.
    
    Returns:
        dict: {
            "standings": {grupo: DataFrame},
            "matches": list of dict,
            "qualified_teams": list of str (32 equipos en total)
        }
    """
    group_standings = {}
    all_group_matches = []

    for group_name, teams in WORLD_CUP_2026_GROUPS.items():
        # Inicializar tabla de posiciones del grupo
        standings = pd.DataFrame({
            "Equipo": teams,
            "PJ": 0, "PG": 0, "PE": 0, "PP": 0,
            "GF": 0, "GC": 0, "DG": 0, "Pts": 0
        }).set_index("Equipo")

        # Enfrentamientos de todos contra todos (6 partidos por grupo)
        n_teams = len(teams)
        for i in range(n_teams):
            for j in range(i + 1, n_teams):
                team_a = teams[i]
                team_b = teams[j]
                
                # Simular partido (neutral=True)
                goals_a, goals_b = simulate_match(model, team_a, team_b, df, model_type, neutral=True)
                
                # Actualizar PJ
                standings.loc[team_a, "PJ"] += 1
                standings.loc[team_b, "PJ"] += 1
                
                # Actualizar goles
                standings.loc[team_a, "GF"] += goals_a
                standings.loc[team_a, "GC"] += goals_b
                standings.loc[team_b, "GF"] += goals_b
                standings.loc[team_b, "GC"] += goals_a
                
                # Actualizar resultados y puntos
                if goals_a > goals_b:
                    standings.loc[team_a, "PG"] += 1
                    standings.loc[team_a, "Pts"] += 3
                    standings.loc[team_b, "PP"] += 1
                elif goals_a < goals_b:
                    standings.loc[team_b, "PG"] += 1
                    standings.loc[team_b, "Pts"] += 3
                    standings.loc[team_a, "PP"] += 1
                else:
                    standings.loc[team_a, "PE"] += 1
                    standings.loc[team_a, "Pts"] += 1
                    standings.loc[team_b, "PE"] += 1
                    standings.loc[team_b, "Pts"] += 1
                
                all_group_matches.append({
                    "grupo": group_name,
                    "local": team_a,
                    "visitante": team_b,
                    "goles_local": goals_a,
                    "goles_visitante": goals_b
                })
        
        # Calcular diferencia de goles
        standings["DG"] = standings["GF"] - standings["GC"]
        
        # Ordenar tabla del grupo: Pts -> DG -> GF -> Alfabético
        standings = standings.sort_values(by=["Pts", "DG", "GF"], ascending=[False, False, False])
        group_standings[group_name] = standings.reset_index()

    # Determinar clasificados (2 primeros de cada grupo)
    direct_qualified = []
    third_placed_teams = []

    for group_name, standings in group_standings.items():
        direct_qualified.append(standings.iloc[0]["Equipo"])
        direct_qualified.append(standings.iloc[1]["Equipo"])
        
        # Guardar el tercer lugar con información de su grupo para desempatar
        third_team = standings.iloc[2].to_dict()
        third_team["grupo"] = group_name
        third_placed_teams.append(third_team)

    # Clasificar los mejores terceros lugares
    df_thirds = pd.DataFrame(third_placed_teams)
    # Ordenar terceros: Pts -> DG -> GF
    df_thirds = df_thirds.sort_values(by=["Pts", "DG", "GF"], ascending=[False, False, False])
    best_thirds = df_thirds.head(8)["Equipo"].tolist()

    qualified_teams = direct_qualified + best_thirds

    return {
        "standings": group_standings,
        "matches": all_group_matches,
        "qualified_teams": qualified_teams
    }


def simulate_knockout_match(model, team_a: str, team_b: str, df: pd.DataFrame,
                            model_type: str = "xgboost") -> dict:
    """
    Simula un partido de eliminación directa (knockout).
    Si hay empate, se decide por penales (50% de probabilidad base, pero
    ajustado levemente por diferencia de xG/calidad).
    """
    goals_a, goals_b = simulate_match(model, team_a, team_b, df, model_type, neutral=True)
    
    winner = None
    penales_str = ""
    
    if goals_a > goals_b:
        winner = team_a
    elif goals_a < goals_b:
        winner = team_b
    else:
        # Tanda de penales simulada
        # Obtenemos fuerzas aproximadas para dar un pequeño bonus en penales
        try:
            if model_type == "bayesian":
                pred = model.predict(team_a, team_b, neutral=True)
            else:
                pred = model.predict(df, team_a, team_b, neutral=True)
            diff = pred["xg_home"] - pred["xg_away"]
        except Exception:
            diff = 0.0
            
        prob_a = 0.5 + min(max(diff * 0.1, -0.15), 0.15)  # Max +15% / -15%
        
        # Simular tiros de penales
        penales_a = np.random.binomial(5, 0.75 + (diff * 0.02))
        penales_b = np.random.binomial(5, 0.75 - (diff * 0.02))
        
        # Muerte súbita si empatan penales
        while penales_a == penales_b:
            penales_a += 1 if np.random.random() < 0.75 else 0
            penales_b += 1 if np.random.random() < 0.75 else 0
            
        if penales_a > penales_b:
            winner = team_a
        else:
            winner = team_b
            
        penales_str = f" ({penales_a} - {penales_b} Pen.)"
        
    return {
        "local": team_a,
        "visitante": team_b,
        "goles_local": goals_a,
        "goles_visitante": goals_b,
        "ganador": winner,
        "penales_str": penales_str
    }


def simulate_tournament_bracket(model, qualified_teams: list, df: pd.DataFrame,
                                model_type: str = "xgboost") -> dict:
    """
    Simula las fases eliminatorias desde Dieciseisavos (R32) hasta la final.
    """
    results = {}
    
    # ── Dieciseisavos de Final (32 equipos -> 16 partidos) ─────────────────────
    # Emparejamiento simplificado pero lógico (p.ej. 1 vs 32, 2 vs 31, etc.)
    # Ordenamos un poco para que no se enfrenten los del mismo grupo inmediatamente
    teams_to_pair = list(qualified_teams)
    # Mezclamos ligeramente para simular sorteo
    np.random.shuffle(teams_to_pair)
    
    r32_matches = []
    r16_teams = []
    
    for idx in range(0, 32, 2):
        match_res = simulate_knockout_match(model, teams_to_pair[idx], teams_to_pair[idx+1], df, model_type)
        r32_matches.append(match_res)
        r16_teams.append(match_res["ganador"])
        
    results["R32"] = r32_matches
    
    # ── Octavos de Final (16 equipos -> 8 partidos) ───────────────────────────
    r16_matches = []
    qf_teams = []
    
    for idx in range(0, 16, 2):
        match_res = simulate_knockout_match(model, r16_teams[idx], r16_teams[idx+1], df, model_type)
        r16_matches.append(match_res)
        qf_teams.append(match_res["ganador"])
        
    results["R16"] = r16_matches
    
    # ── Cuartos de Final (8 equipos -> 4 partidos) ────────────────────────────
    qf_matches = []
    sf_teams = []
    
    for idx in range(0, 8, 2):
        match_res = simulate_knockout_match(model, qf_teams[idx], qf_teams[idx+1], df, model_type)
        qf_matches.append(match_res)
        sf_teams.append(match_res["ganador"])
        
    results["QF"] = qf_matches
    
    # ── Semifinales (4 equipos -> 2 partidos) ─────────────────────────────────
    sf_matches = []
    final_teams = []
    third_place_teams = []
    
    for idx in range(0, 4, 2):
        match_res = simulate_knockout_match(model, sf_teams[idx], sf_teams[idx+1], df, model_type)
        sf_matches.append(match_res)
        final_teams.append(match_res["ganador"])
        
        # El perdedor va al tercer lugar
        loser = sf_teams[idx] if match_res["ganador"] == sf_teams[idx+1] else sf_teams[idx+1]
        third_place_teams.append(loser)
        
    results["SF"] = sf_matches
    
    # ── Tercer Lugar y Final ──────────────────────────────────────────────────
    third_place_match = simulate_knockout_match(model, third_place_teams[0], third_place_teams[1], df, model_type)
    results["3rd"] = third_place_match
    
    final_match = simulate_knockout_match(model, final_teams[0], final_teams[1], df, model_type)
    results["Final"] = final_match
    
    results["campeon"] = final_match["ganador"]
    results["subcampeon"] = final_match["local"] if final_match["ganador"] == final_match["visitante"] else final_match["visitante"]
    results["tercero"] = third_place_match["ganador"]
    
    return results
