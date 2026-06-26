"""
utils.py — Funciones auxiliares para el predictor del Mundial 2026.
"""

import numpy as np
from scipy.stats import poisson
from functools import lru_cache


# ── Equipos clasificados al Mundial 2026 ──────────────────────────────────
WORLD_CUP_2026_TEAMS = [
    # Grupo A
    "United States", "Morocco", "Colombia", "Mali",
    # Grupo B
    "Mexico", "Argentina", "Senegal", "Uzbekistan",
    # Grupo C
    "Canada", "Brazil", "Bolivia", "New Zealand",
    # Grupo D
    "Netherlands", "Indonesia", "Egypt", "Paraguay",
    # Grupo E
    "Germany", "Uruguay", "South Korea", "Bahrain",
    # Grupo F
    "Portugal", "Iran", "Panama", "Ivory Coast",
    # Grupo G
    "Spain", "Ecuador", "Chile", "Honduras",
    # Grupo H
    "France", "Croatia", "Australia", "Peru",
    # Grupo I
    "England", "Denmark", "Tunisia", "Trinidad and Tobago",
    # Grupo J
    "Belgium", "Japan", "Saudi Arabia", "Venezuela",
    # Grupo K
    "Italy", "Cameroon", "Costa Rica", "Serbia",
    # Grupo L
    "Switzerland", "Nigeria", "Turkey", "Jamaica",
]

# Grupos del Mundial 2026
WORLD_CUP_2026_GROUPS = {
    "A": ["United States", "Morocco", "Colombia", "Mali"],
    "B": ["Mexico", "Argentina", "Senegal", "Uzbekistan"],
    "C": ["Canada", "Brazil", "Bolivia", "New Zealand"],
    "D": ["Netherlands", "Indonesia", "Egypt", "Paraguay"],
    "E": ["Germany", "Uruguay", "South Korea", "Bahrain"],
    "F": ["Portugal", "Iran", "Panama", "Ivory Coast"],
    "G": ["Spain", "Ecuador", "Chile", "Honduras"],
    "H": ["France", "Croatia", "Australia", "Peru"],
    "I": ["England", "Denmark", "Tunisia", "Trinidad and Tobago"],
    "J": ["Belgium", "Japan", "Saudi Arabia", "Venezuela"],
    "K": ["Italy", "Cameroon", "Costa Rica", "Serbia"],
    "L": ["Switzerland", "Nigeria", "Turkey", "Jamaica"],
}


# ── Pesos por tipo de torneo ──────────────────────────────────────────────
TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 1.0,
    "FIFA World Cup qualification": 0.80,
    "Copa América": 0.75,
    "UEFA Euro": 0.75,
    "UEFA Euro qualification": 0.70,
    "African Cup of Nations": 0.70,
    "AFC Asian Cup": 0.70,
    "CONCACAF Gold Cup": 0.65,
    "UEFA Nations League": 0.65,
    "CONMEBOL–UEFA Cup of Champions": 0.70,
    "FIFA Confederations Cup": 0.70,
    "African Cup of Nations qualification": 0.60,
    "AFC Asian Cup qualification": 0.60,
    "CONCACAF Gold Cup qualification": 0.55,
    "Friendly": 0.30,
}


def get_tournament_weight(tournament_name: str) -> float:
    """Devuelve el peso de importancia de un torneo."""
    for key, weight in TOURNAMENT_WEIGHTS.items():
        if key.lower() in tournament_name.lower():
            return weight
    # Si contiene 'qualification' genérico
    if "qualification" in tournament_name.lower():
        return 0.55
    # Otros torneos oficiales
    if "friendly" in tournament_name.lower():
        return 0.30
    return 0.45  # Torneo oficial desconocido


def compute_poisson_matrix(lambda_home: float, lambda_away: float,
                           max_goals: int = 7) -> np.ndarray:
    """
    Calcula la matriz de probabilidades de marcadores usando distribución de Poisson.

    P(home=i, away=j) = P_poisson(i, λ_home) × P_poisson(j, λ_away)

    Returns:
        np.ndarray de shape (max_goals, max_goals) con las probabilidades.
    """
    home_probs = np.array([poisson.pmf(i, lambda_home) for i in range(max_goals)])
    away_probs = np.array([poisson.pmf(j, lambda_away) for j in range(max_goals)])
    matrix = np.outer(home_probs, away_probs)
    return matrix


def get_match_probabilities(matrix: np.ndarray) -> dict:
    """
    Dado la matriz de Poisson, calcula probabilidades de victoria/empate/derrota.
    """
    home_win = np.sum(np.tril(matrix, -1))  # debajo de la diagonal
    draw = np.sum(np.diag(matrix))           # en la diagonal
    away_win = np.sum(np.triu(matrix, 1))    # arriba de la diagonal

    total = home_win + draw + away_win
    return {
        "home_win": home_win / total,
        "draw": draw / total,
        "away_win": away_win / total,
    }


def get_fair_odds(probs: dict) -> dict:
    """
    Calcula las Cuotas Justas (Fair Odds) a partir de las probabilidades: Cuota = 1 / Probabilidad.
    """
    return {
        "home_win": 1.0 / probs["home_win"] if probs["home_win"] > 0 else 999.0,
        "draw": 1.0 / probs["draw"] if probs["draw"] > 0 else 999.0,
        "away_win": 1.0 / probs["away_win"] if probs["away_win"] > 0 else 999.0,
    }


def get_most_likely_scores(matrix: np.ndarray, top_n: int = 5) -> list:
    """
    Devuelve los top_n marcadores más probables.
    """
    flat_indices = np.argsort(matrix.flatten())[::-1][:top_n]
    max_goals = matrix.shape[0]
    results = []
    for idx in flat_indices:
        home_goals = idx // max_goals
        away_goals = idx % max_goals
        prob = matrix[home_goals, away_goals]
        results.append((home_goals, away_goals, prob))
    return results


@lru_cache(maxsize=1)
def load_advanced_data() -> dict:
    """
    Lee datos_avanzados.json de la raiz del proyecto.
    """
    import json
    import os
    # La raiz es el directorio padre de 'src'
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(project_root, "datos_avanzados.json")
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_advanced_global_defaults(data: dict) -> dict:
    """
    Calcula los promedios globales de las variables de equipos del JSON.
    """
    import numpy as np
    vals = []
    xgs = []
    cards = []
    for k, v in data.items():
        if k == "sedes_mundial_altitud":
            continue
        if isinstance(v, dict):
            if "valor_plantilla_mde" in v:
                vals.append(v["valor_plantilla_mde"])
            if "promedio_xg_eliminatorias" in v:
                xgs.append(v["promedio_xg_eliminatorias"])
            if "tarjetas_amarillas_por_partido" in v:
                cards.append(v["tarjetas_amarillas_por_partido"])
    
    return {
        "valor_plantilla_mde": float(np.mean(vals)) if vals else 500.0,
        "promedio_xg_eliminatorias": float(np.mean(xgs)) if xgs else 1.8,
        "tarjetas_amarillas_por_partido": float(np.mean(cards)) if cards else 1.5,
        "dependencia_estrella_porcentaje": 0.0,
        "jugador_estrella_nombre": None
    }


def get_team_advanced_stats(team_name: str, data: dict, defaults: dict) -> dict:
    """
    Obtiene las variables avanzadas para un equipo traduciendo y normalizando el nombre.
    """
    import unicodedata
    
    def clean(s):
        s = unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode("utf-8")
        return s.lower().replace("_", " ").strip()
    
    cleaned_name = clean(team_name)
    
    # Mapeo de traduccion Ingles/Español para los equipos del JSON
    translation = {
        "united states": "Estados Unidos",
        "morocco": "Marruecos",
        "colombia": "Colombia",
        "mali": "Malí",
        "mexico": "México",
        "argentina": "Argentina",
        "senegal": "Senegal",
        "uzbekistan": "Uzbekistán",
        "canada": "Canadá",
        "brazil": "Brasil",
        "bolivia": "Bolivia",
        "new zealand": "Nueva Zelanda",
        "netherlands": "Países Bajos",
        "indonesia": "Indonesia",
        "egypt": "Egipto",
        "paraguay": "Paraguay",
        "germany": "Alemania",
        "uruguay": "Uruguay",
        "south korea": "Corea del Sur",
        "bahrain": "Baréin",
        "portugal": "Portugal",
        "iran": "Irán",
        "panama": "Panamá",
        "ivory coast": "Costa de Marfil",
        "spain": "España",
        "ecuador": "Ecuador",
        "chile": "Chile",
        "honduras": "Honduras",
        "france": "Francia",
        "croatia": "Croacia",
        "australia": "Australia",
        "peru": "Perú",
        "england": "Inglaterra",
        "denmark": "Dinamarca",
        "tunisia": "Túnez",
        "trinidad and tobago": "Trinidad y Tobago",
        "belgium": "Bélgica",
        "japan": "Japón",
        "saudi arabia": "Arabia Saudita",
        "venezuela": "Venezuela",
        "italy": "Italia",
        "cameroon": "Camerún",
        "costa rica": "Costa Rica",
        "serbia": "Serbia",
        "switzerland": "Suiza",
        "nigeria": "Nigeria",
        "turkey": "Turquía",
        "jamaica": "Jamaica"
    }
    
    mapped_name = translation.get(cleaned_name, team_name)
    cleaned_mapped = clean(mapped_name)
    
    for key, val in data.items():
        if key == "sedes_mundial_altitud":
            continue
        if clean(key) == cleaned_mapped:
            return {
                "valor_plantilla_mde": val.get("valor_plantilla_mde", defaults["valor_plantilla_mde"]),
                "promedio_xg_eliminatorias": val.get("promedio_xg_eliminatorias", defaults["promedio_xg_eliminatorias"]),
                "tarjetas_amarillas_por_partido": val.get("tarjetas_amarillas_por_partido", defaults["tarjetas_amarillas_por_partido"]),
                "dependencia_estrella_porcentaje": val.get("dependencia_estrella_porcentaje", defaults["dependencia_estrella_porcentaje"]),
                "jugador_estrella_nombre": val.get("jugador_estrella_nombre", defaults["jugador_estrella_nombre"])
            }
            
    return defaults

