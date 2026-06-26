"""
app.py — Interfaz de usuario interactiva en Streamlit para el Predictor del Mundial 2026.
"""

import os
# Limitar hilos para evitar conflictos con CUDA y hilos de Streamlit
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import time

from src.data_loader import load_and_preprocess, prepare_bayesian_data, get_teams_list
from src.bayesian_model import BayesianGoalModel, BAYESIAN_PARAMS_FILE
from src.xgboost_model import XGBoostGoalModel, XGBOOST_MODEL_FILE
from src.feature_engineering import build_training_dataset
from src.utils import (
    WORLD_CUP_2026_TEAMS,
    WORLD_CUP_2026_GROUPS,
    compute_poisson_matrix,
    get_match_probabilities,
    get_most_likely_scores,
    get_fair_odds
)
from src.simulation import simulate_group_stage, simulate_tournament_bracket

# Configuración de página
st.set_page_config(
    page_title="🏆 Predictor Mundial FIFA 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para Diseño Premium (Dark/Modern glassmorphism look)
st.markdown("""
<style>
    /* Estilos generales */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Contenedores Principales */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 50% 50%, #161b22 0%, #0d1117 100%) !important;
        color: #f0f6fc !important;
    }
    
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0b0e14 !important;
        border-right: 1px solid rgba(48, 54, 65, 0.6) !important;
    }
    
    /* Título Héroe con gradiente animado */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff4b4b 0%, #ff8533 50%, #ffcc00 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        padding-top: 1rem;
        animation: shine 4s linear infinite;
    }
    
    @keyframes shine {
        to { background-position: 200% center; }
    }
    
    .hero-subtitle {
        font-size: 1.15rem;
        text-align: center;
        color: #8b949e;
        margin-bottom: 2.5rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    /* Contenedores Glassmorphism */
    .card {
        background: rgba(22, 27, 34, 0.45);
        border: 1px solid rgba(48, 54, 65, 0.7);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37), inset 0 1px 1px rgba(255, 255, 255, 0.05);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    
    .card:hover {
        transform: translateY(-2px);
        border-color: rgba(88, 166, 255, 0.3);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5), inset 0 1px 1px rgba(255, 255, 255, 0.08);
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1c2128 0%, #161b22 100%);
        border: 1px solid rgba(88, 166, 255, 0.15);
        border-radius: 14px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.25);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
        border-color: rgba(88, 166, 255, 0.4);
        box-shadow: 0 8px 25px rgba(88, 166, 255, 0.1);
    }
    
    .metric-value {
        font-size: 2.3rem;
        font-weight: 800;
        color: #58a6ff;
        text-shadow: 0 0 10px rgba(88, 166, 255, 0.2);
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    
    /* Botones Premium con gradiente */
    div.stButton > button {
        background: linear-gradient(90deg, #ff4b4b 0%, #ff8533 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.6rem 1.8rem !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.25) !important;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(255, 75, 75, 0.45) !important;
        background: linear-gradient(90deg, #ff6262 0%, #ff964f 100%) !important;
    }
    
    div.stButton > button:active {
        transform: translateY(1px) !important;
    }
    
    /* Personalización de Inputs de Streamlit */
    div[data-baseweb="select"] > div, input {
        background-color: rgba(22, 27, 34, 0.7) !important;
        border: 1px solid rgba(48, 54, 65, 0.8) !important;
        color: #f0f6fc !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-baseweb="select"] > div:hover, input:hover {
        border-color: rgba(88, 166, 255, 0.5) !important;
    }
    
    /* Expanders Premium */
    div.stExpander {
        background: rgba(22, 27, 34, 0.45) !important;
        border: 1px solid rgba(48, 54, 65, 0.7) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
        overflow: hidden !important;
    }
    
    div.stExpander details summary {
        font-weight: 600 !important;
        color: #c9d1d9 !important;
        padding: 0.5rem 1rem !important;
        transition: color 0.3s ease !important;
    }
    
    div.stExpander details summary:hover {
        color: #58a6ff !important;
    }
    
    /* Badges de Probabilidad */
    .prob-badge {
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.35rem 0.9rem;
        border-radius: 8px;
        display: inline-block;
        margin-right: 0.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    .prob-win { background-color: rgba(46, 160, 67, 0.2); color: #3fb950; border: 1px solid rgba(46, 160, 67, 0.35); }
    .prob-draw { background-color: rgba(139, 148, 158, 0.2); color: #c9d1d9; border: 1px solid rgba(139, 148, 158, 0.35); }
    .prob-loss { background-color: rgba(248, 81, 81, 0.2); color: #f85149; border: 1px solid rgba(248, 81, 81, 0.35); }
</style>
""", unsafe_allow_html=True)


# ── CARGA Y CACHÉ DE MODELOS Y DATOS ──────────────────────────────────────────
@st.cache_resource
def get_dataset():
    """Descarga y carga el dataset preprocesado."""
    try:
        df = load_and_preprocess(min_year=2018)
        return df
    except Exception as e:
        st.error(f"Error cargando los datos de partidos: {e}")
        return None

def check_models_exist():
    """Verifica si existen archivos de modelos entrenados."""
    return os.path.exists(BAYESIAN_PARAMS_FILE) and os.path.exists(XGBOOST_MODEL_FILE)

@st.cache_resource
def load_models():
    """Carga los modelos entrenados desde disco."""
    if not check_models_exist():
        return None, None
        
    bayes_model = BayesianGoalModel()
    bayes_model.load()
    
    xgb_model = XGBoostGoalModel()
    xgb_model.load()
    
    return bayes_model, xgb_model

# ── RENDERIZADO PRINCIPAL ──

st.markdown('<div class="hero-title">Mundial FIFA 2026</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Modelos Matemáticos Predictivos: MCMC Bayesiano & XGBoost Poisson</div>', unsafe_allow_html=True)

df = get_dataset()

# Inicializar modelos en session_state si es posible
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.bayes_model = None
    st.session_state.xgb_model = None

if check_models_exist() and not st.session_state.models_loaded:
    b_mod, x_mod = load_models()
    st.session_state.bayes_model = b_mod
    st.session_state.xgb_model = x_mod
    st.session_state.models_loaded = True

def save_prediction_to_log(data: dict):
    """Guarda los detalles de una predicción en predictions_log.json de forma segura."""
    import json
    log_file = "predictions_log.json"
    history = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception as e:
            logger.error(f"Error al leer predictions_log.json: {e}")
            history = []
            
    # Evitar guardar duplicados exactos en el mismo minuto
    if history:
        last = history[-1]
        if (last.get("home_team") == data.get("home_team") and 
            last.get("away_team") == data.get("away_team") and 
            last.get("timestamp", "")[:16] == data.get("timestamp", "")[:16]):
            return
            
    history.append(data)
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error al escribir predictions_log.json: {e}")

# Sidebar navegación
st.sidebar.image("logo.png", width=120)
st.sidebar.markdown("### Navegación")
page = st.sidebar.radio(
    "Selecciona una pestaña:",
    ["🏠 Inicio y Modelos", "⚽ Predicción de Partido", "📋 Registro de Predicciones", "🏆 Simular Mundial 2026", "📊 Rankings e Intensidad", "🔬 Diagnósticos y SHAP"]
)

# ──────────────────────────────────────────────────────────────────────────────
# PESTAÑA 1: INICIO Y ENTRENAMIENTO
# ──────────────────────────────────────────────────────────────────────────────
if page == "🏠 Inicio y Modelos":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Información General del Proyecto")
    st.write(
        "Este sistema predice resultados de partidos de fútbol de nivel internacional utilizando "
        "dos aproximaciones matemáticas de vanguardia. La base de datos contiene los resultados de "
        "todos los partidos oficiales y amistosos jugados a nivel mundial."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **1. Modelo Bayesiano Dixon-Coles (MCMC)**
        * Corre una simulación de Cadenas de Markov de Monte Carlo (MCMC) a través de **PyMC** con un muestreador NUTS.
        * Modela los goles anotados por cada equipo como variables Poisson independientes.
        * El ratio de goles depende de:
          * **Fuerza de ataque** del equipo local.
          * **Fuerza de defensa** del equipo visitante.
          * **Ventaja de localía** (neutralizado para el Mundial).
          * Un factor global de intercepción.
        """)
    with col2:
        st.markdown("""
        **2. Modelo XGBoost Poisson**
        * Utiliza dos modelos de **Gradient Boosting (XGBoost)** independientes.
        * Optimiza la métrica de conteo Poisson (`count:poisson`) directamente para los goles a favor y en contra.
        * Genera variables dinámicas como:
          * Rendimiento ponderado de goles anotados/recibidos en los últimos 20 partidos.
          * Racha de victorias/derrotas.
          * Estadísticas históricas de enfrentamientos directos (Head-to-Head).
          * Desviaciones defensivas y de ataque respecto al promedio global.
        """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Caja de Estado de Entrenamiento
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Estado de los Modelos")
    
    if check_models_exist():
        st.success("✅ ¡Los modelos predictivos están entrenados y listos para usar!")
        st.info(f"Ubicación de archivos:\n* Bayesiano: `{BAYESIAN_PARAMS_FILE}`\n* XGBoost: `{XGBOOST_MODEL_FILE}`")
        
        # Botón para forzar re-entrenamiento
        if st.button("🔄 Volver a Entrenar Modelos (Toma ~3-5 min)"):
            st.session_state.models_loaded = False
            st.rerun()
    else:
        st.warning("⚠️ Los modelos predictivos no están entrenados en este entorno. Debes entrenarlos para comenzar.")
        
        if df is not None:
            if st.button("🚀 Iniciar Entrenamiento de Modelos"):
                with st.spinner("Preparando datos y entrenando modelos. Por favor espera..."):
                    # 1. Entrenar XGBoost
                    st.text("🌲 Entrenando XGBoost...")
                    train_df = build_training_dataset(df)
                    xgb_model = XGBoostGoalModel()
                    xgb_model.fit(train_df)
                    xgb_model.save()
                    
                    # 2. Entrenar Bayesiano
                    st.text("🧮 Entrenando Modelo Bayesiano (PyMC MCMC)...")
                    bayes_data = prepare_bayesian_data(df)
                    bayes_model = BayesianGoalModel()
                    # muestras reducidas para balancear velocidad y calidad en Windows sin C++
                    bayes_model.fit(bayes_data, samples=150, tune=100, chains=1)
                    bayes_model.save()
                    
                    st.success("🎉 ¡Modelos entrenados y guardados exitosamente!")
                    st.session_state.models_loaded = True
                    st.session_state.bayes_model = bayes_model
                    st.session_state.xgb_model = xgb_model
                    time.sleep(2)
                    st.rerun()
        else:
            st.error("No se pudo cargar el dataset para entrenamiento.")
    st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# PESTAÑA 2: PREDICCIÓN DE PARTIDO
# ──────────────────────────────────────────────────────────────────────────────
elif page == "⚽ Predicción de Partido":
    if not st.session_state.models_loaded:
        st.warning("⚠️ Debes entrenar los modelos primero. Ve a la pestaña '🏠 Inicio y Modelos'.")
    else:
        st.subheader("Simulador de Enfrentamiento Directo")
        
        # Obtener lista de equipos
        available_teams = get_teams_list(df)
        
        # Opciones por defecto alineadas al Mundial
        wc_teams_in_dataset = [t for t in WORLD_CUP_2026_TEAMS if t in available_teams]
        other_teams = [t for t in available_teams if t not in WORLD_CUP_2026_TEAMS]
        sorted_teams = sorted(wc_teams_in_dataset) + ["--- OTROS EQUIPOS ---"] + sorted(other_teams)
        
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            # Filtrar el separador
            cleaned_teams = [t for t in sorted_teams if t != "--- OTROS EQUIPOS ---"]
            home_team = st.selectbox("Selecciona Equipo Local (o Local Administrativo):", cleaned_teams, index=cleaned_teams.index("Argentina") if "Argentina" in cleaned_teams else 0)
        with col_sel2:
            away_team = st.selectbox("Selecciona Equipo Visitante:", cleaned_teams, index=cleaned_teams.index("France") if "France" in cleaned_teams else 1)
            
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            model_choice = st.radio("Modelo a utilizar:", ["XGBoost (Recomendado)", "Bayesiano MCMC", "Promedio de Ambos"], horizontal=True)

        # Cargar sedes mundialistas y altitudes del JSON
        from src.utils import load_advanced_data, get_advanced_global_defaults, get_team_advanced_stats
        adv_data = load_advanced_data()
        defaults = get_advanced_global_defaults(adv_data)
        sedes = adv_data.get("sedes_mundial_altitud", {})
        cities_list = list(sedes.keys())

        with col_opt2:
            selected_city = st.selectbox(
                "Sede del Encuentro:",
                cities_list,
                index=cities_list.index("CDMX") if "CDMX" in cities_list else 0
            )
            altitude = float(sedes.get(selected_city, 0.0))
            st.caption(f"Altitud: **{altitude:.0f} m** ({selected_city})")

        with col_opt3:
            neutral = st.checkbox("¿Partido en cancha Neutral? (Recomendado para el Mundial)", value=True)
            
        if home_team == away_team:
            st.error("Por favor selecciona dos equipos diferentes.")
        else:
            # Opcional: Análisis de Alineaciones y Árbitro en Tiempo Real (Web Scraping)
            referee_multiplier = 1.0
            scraped_referee = None
            scraped_cards_avg = 4.5
            home_starting_val = None
            away_starting_val = None
            lineup_data = None

            with st.expander("🌐 Análisis en Tiempo Real (Alineaciones y Árbitro)", expanded=False):
                st.write("Ingresa la URL de un partido en portales públicos (ej. Transfermarkt o Flashscore) para raspar titulares confirmados y árbitro central.")
                match_url = st.text_input("URL de Match / Ficha de Partido:", key="match_url", placeholder="https://www.transfermarkt.com/...")
                
                if match_url:
                    from src.scraper import scrape_match_details
                    with st.spinner("Raspando alineaciones confirmadas y árbitro central..."):
                        try:
                            lineup_data = scrape_match_details(match_url, home_team, away_team)
                            st.success(f"¡Datos cargados con éxito! Origen: **{lineup_data['source'].upper()}**")
                            
                            # Mostrar alineaciones
                            col_l1, col_l2 = st.columns(2)
                            with col_l1:
                                st.markdown(f"**Titulares {home_team}:**")
                                st.write(", ".join(lineup_data["home_lineup"]))
                                st.write(f"Valor del Once: **{lineup_data['starting_val_home']:.1f} Mde** (vs total plantilla: {get_team_advanced_stats(home_team, adv_data, defaults)['valor_plantilla_mde']:.1f} Mde)")
                            with col_l2:
                                st.markdown(f"**Titulares {away_team}:**")
                                st.write(", ".join(lineup_data["away_lineup"]))
                                st.write(f"Valor del Once: **{lineup_data['starting_val_away']:.1f} Mde** (vs total plantilla: {get_team_advanced_stats(away_team, adv_data, defaults)['valor_plantilla_mde']:.1f} Mde)")
                                
                            home_starting_val = lineup_data["starting_val_home"]
                            away_starting_val = lineup_data["starting_val_away"]
                            
                            scraped_referee = lineup_data["referee"]
                            scraped_cards_avg = lineup_data["referee_cards_avg"]
                            referee_multiplier = scraped_cards_avg / 4.5
                            
                            st.info(f"**Árbitro Asignado:** {scraped_referee} (Promedio: **{scraped_cards_avg:.2f} tarjetas**, Factor: **x{referee_multiplier:.2f}**)")
                        except Exception as e:
                            st.error(f"Error procesando el scraping del partido: {e}")

            # Ejecutar Predicción antes de los expanders de apuestas
            try:
                pred_xgb = st.session_state.xgb_model.predict(
                    df, home_team, away_team, neutral=neutral, altitude=altitude,
                    home_starting_val=home_starting_val, away_starting_val=away_starting_val,
                    home_lineup=lineup_data["home_lineup"] if lineup_data else None,
                    away_lineup=lineup_data["away_lineup"] if lineup_data else None
                )
                l_home_xgb, l_away_xgb = pred_xgb["xg_home"], pred_xgb["xg_away"]
            except Exception as e:
                logger.warning(f"Predicción XGBoost falló para {home_team} vs {away_team}: {e}")
                l_home_xgb, l_away_xgb = 1.2, 1.1
                st.warning(f"⚠️ El modelo XGBoost usó valores de respaldo para este partido. Razón: {e}")
            matrix_xgb = compute_poisson_matrix(l_home_xgb, l_away_xgb)
            probs_xgb = get_match_probabilities(matrix_xgb)
            
            try:
                pred_bay = st.session_state.bayes_model.predict(home_team, away_team, neutral=neutral)
                l_home_bay, l_away_bay = pred_bay["xg_home"], pred_bay["xg_away"]
            except Exception as e:
                logger.warning(f"Predicción Bayesiana falló para {home_team} vs {away_team}: {e}")
                l_home_bay, l_away_bay = 1.2, 1.1
                st.warning(f"⚠️ El modelo Bayesiano usó valores de respaldo para este partido. Razón: {e}")
            matrix_bay = compute_poisson_matrix(l_home_bay, l_away_bay)
            probs_bay = get_match_probabilities(matrix_bay)
            
            if model_choice == "XGBoost (Recomendado)":
                l_home, l_away = l_home_xgb, l_away_xgb
                probs = probs_xgb
            elif model_choice == "Bayesiano MCMC":
                l_home, l_away = l_home_bay, l_away_bay
                probs = probs_bay
            else:  # Ensemble Pesado (75% Bayes + 25% XGBoost)
                l_home = (0.25 * l_home_xgb) + (0.75 * l_home_bay)
                l_away = (0.25 * l_away_xgb) + (0.75 * l_away_bay)
                matrix = compute_poisson_matrix(l_home, l_away)
                probs = get_match_probabilities(matrix)
            
            matrix = compute_poisson_matrix(l_home, l_away)
            most_likely = get_most_likely_scores(matrix, top_n=5)
            
            # Calcular probabilidades de mercados secundarios en backend vectorialmente
            goals_grid = np.arange(matrix.shape[0])[:, None] + np.arange(matrix.shape[1])[None, :]
            p_under = float(matrix[goals_grid <= 2].sum())
            p_over = float(matrix[goals_grid > 2].sum())
            p_btts_yes = float(matrix[1:, 1:].sum())
            p_btts_no = 1.0 - p_btts_yes

            # Calcular cuotas estimadas de mercado y ventajas (CLV, EV) para 1X2
            alpha = 0.35
            p_consensus = {
                "home_win": (1.0 - alpha) * probs["home_win"] + alpha * probs_xgb["home_win"],
                "draw": (1.0 - alpha) * probs["draw"] + alpha * probs_xgb["draw"],
                "away_win": (1.0 - alpha) * probs["away_win"] + alpha * probs_xgb["away_win"]
            }
            sum_p = sum(p_consensus.values())
            for k in p_consensus:
                p_consensus[k] /= sum_p
                
            odds_pinnacle = {k: max(1.01, min(20.0, 1.0 / (p_consensus[k] * 1.025))) for k in p_consensus}
            odds_bet365 = {k: max(1.01, min(20.0, 1.0 / (p_consensus[k] * 1.05))) for k in p_consensus}
            
            p_closing = {}
            sum_p_closing = 0.0
            for k in p_consensus:
                p_closing[k] = p_consensus[k] + 0.4 * (probs[k] - p_consensus[k])
                sum_p_closing += p_closing[k]
            for k in p_closing:
                p_closing[k] /= sum_p_closing
            odds_closing_pinnacle = {k: max(1.01, min(20.0, 1.0 / (p_closing[k] * 1.025))) for k in p_closing}
            
            ev_pinnacle = {k: (probs[k] * odds_pinnacle[k]) - 1.0 for k in p_consensus}
            ev_bet365 = {k: (probs[k] * odds_bet365[k]) - 1.0 for k in p_consensus}
            
            clv_pinnacle = {k: (odds_pinnacle[k] / odds_closing_pinnacle[k]) - 1.0 for k in p_consensus}
            clv_bet365 = {k: (odds_bet365[k] / odds_closing_pinnacle[k]) - 1.0 for k in p_consensus}
            
            # Calcular cuotas estimadas de mercado para mercados secundarios vectorialmente
            # Over/Under
            goals_grid_xgb = np.arange(matrix_xgb.shape[0])[:, None] + np.arange(matrix_xgb.shape[1])[None, :]
            probs_xgb_under = float(matrix_xgb[goals_grid_xgb <= 2].sum())
            probs_xgb_over = 1.0 - probs_xgb_under
            p_consensus_over = (1.0 - alpha) * p_over + alpha * probs_xgb_over
            p_consensus_under = (1.0 - alpha) * p_under + alpha * probs_xgb_under
            
            odds_pinnacle_over = max(1.01, min(20.0, 1.0 / (p_consensus_over * 1.025)))
            odds_pinnacle_under = max(1.01, min(20.0, 1.0 / (p_consensus_under * 1.025)))
            odds_bet365_over = max(1.01, min(20.0, 1.0 / (p_consensus_over * 1.05)))
            odds_bet365_under = max(1.01, min(20.0, 1.0 / (p_consensus_under * 1.05)))
            
            p_closing_over = p_consensus_over + 0.4 * (p_over - p_consensus_over)
            p_closing_under = p_consensus_under + 0.4 * (p_under - p_consensus_under)
            odds_closing_pinnacle_over = max(1.01, min(20.0, 1.0 / (p_closing_over * 1.025)))
            odds_closing_pinnacle_under = max(1.01, min(20.0, 1.0 / (p_closing_under * 1.025)))
            
            clv_pinnacle_over = (odds_pinnacle_over / odds_closing_pinnacle_over) - 1.0
            clv_pinnacle_under = (odds_pinnacle_under / odds_closing_pinnacle_under) - 1.0
            
            # BTTS
            probs_xgb_btts_yes = float(matrix_xgb[1:, 1:].sum())
            probs_xgb_btts_no = 1.0 - probs_xgb_btts_yes
            p_consensus_btts_yes = (1.0 - alpha) * p_btts_yes + alpha * probs_xgb_btts_yes
            p_consensus_btts_no = (1.0 - alpha) * p_btts_no + alpha * probs_xgb_btts_no
            
            odds_pinnacle_btts_yes = max(1.01, min(20.0, 1.0 / (p_consensus_btts_yes * 1.025)))
            odds_pinnacle_btts_no = max(1.01, min(20.0, 1.0 / (p_consensus_btts_no * 1.025)))
            odds_bet365_btts_yes = max(1.01, min(20.0, 1.0 / (p_consensus_btts_yes * 1.05)))
            odds_bet365_btts_no = max(1.01, min(20.0, 1.0 / (p_consensus_btts_no * 1.05)))
            
            p_closing_btts_yes = p_consensus_btts_yes + 0.4 * (p_btts_yes - p_consensus_btts_yes)
            p_closing_btts_no = p_consensus_btts_no + 0.4 * (p_btts_no - p_consensus_btts_no)
            odds_closing_pinnacle_btts_yes = max(1.01, min(20.0, 1.0 / (p_closing_btts_yes * 1.025)))
            odds_closing_pinnacle_btts_no = max(1.01, min(20.0, 1.0 / (p_closing_btts_no * 1.025)))
            
            clv_pinnacle_btts_yes = (odds_pinnacle_btts_yes / odds_closing_pinnacle_btts_yes) - 1.0
            clv_pinnacle_btts_no = (odds_pinnacle_btts_no / odds_closing_pinnacle_btts_no) - 1.0

            # Opcional: Comparar con Cuotas de Casa de Apuestas (Análisis de Valor +EV)
            with st.expander("💰 Análisis de Valor de Apuestas (Comparar con Cuotas de la Casa)", expanded=False):
                st.write("Ingresa las cuotas de tu casa de apuestas para verificar si existe Valor Esperado Positivo (+EV).")
                col_c1, col_c2, col_c3 = st.columns(3)
                with col_c1:
                    odds_house_home = st.number_input(f"Cuota Victoria {home_team}", min_value=1.0, value=float(odds_pinnacle["home_win"]), step=0.05, format="%.2f")
                with col_c2:
                    odds_house_draw = st.number_input("Cuota Empate", min_value=1.0, value=float(odds_pinnacle["draw"]), step=0.05, format="%.2f")
                with col_c3:
                    odds_house_away = st.number_input(f"Cuota Victoria {away_team}", min_value=1.0, value=float(odds_pinnacle["away_win"]), step=0.05, format="%.2f")
                
                st.markdown("---")
                st.write("Cuotas para Mercados Secundarios (Over/Under y Ambos Anotan):")
                col_c4, col_c5 = st.columns(2)
                with col_c4:
                    odds_house_over = st.number_input("Cuota Over 2.5 goles", min_value=1.0, value=float(odds_pinnacle_over), step=0.05, format="%.2f")
                    odds_house_under = st.number_input("Cuota Under 2.5 goles", min_value=1.0, value=float(odds_pinnacle_under), step=0.05, format="%.2f")
                with col_c5:
                    odds_house_btts_yes = st.number_input("Cuota Ambos Anotan (Sí)", min_value=1.0, value=float(odds_pinnacle_btts_yes), step=0.05, format="%.2f")
                    odds_house_btts_no = st.number_input("Cuota Ambos Anotan (No)", min_value=1.0, value=float(odds_pinnacle_btts_no), step=0.05, format="%.2f")
                
            # Mostrar Resultados
            st.markdown("---")
            
            # Encabezado visual
            col_match1, col_match_vs, col_match2 = st.columns([5, 2, 5])
            with col_match1:
                st.markdown(f"<div style='text-align: center; padding: 1rem; border-radius: 10px; background-color: rgba(88,166,255,0.1); border: 1px solid rgba(88,166,255,0.2)'><h2>{home_team}</h2><h4>Goles Esperados (xG): <span style='color:#58a6ff'>{l_home:.2f}</span></h4></div>", unsafe_allow_html=True)
            with col_match_vs:
                st.markdown("<h1 style='text-align: center; margin-top: 1.5rem; color:#8b949e;'>VS</h1>", unsafe_allow_html=True)
            with col_match2:
                st.markdown(f"<div style='text-align: center; padding: 1rem; border-radius: 10px; background-color: rgba(255,133,51,0.1); border: 1px solid rgba(255,133,51,0.2)'><h2>{away_team}</h2><h4>Goles Esperados (xG): <span style='color:#ff8533'>{l_away:.2f}</span></h4></div>", unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Calcular Cuotas Justas (Fair Odds)
            fair_odds = get_fair_odds(probs)
            
            # Métricas rápidas de probabilidad
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">Victoria {home_team}</div>'
                    f'<div class="metric-value" style="color: #3fb950">{probs["home_win"] * 100:.1f}%</div>'
                    f'<div style="color: #8b949e; font-size: 0.95rem; margin-top: 0.2rem;">Cuota Justa: <b>{fair_odds["home_win"]:.2f}</b></div>'
                    f'</div>', unsafe_allow_html=True
                )
            with col_p2:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">Empate</div>'
                    f'<div class="metric-value" style="color: #c9d1d9">{probs["draw"] * 100:.1f}%</div>'
                    f'<div style="color: #8b949e; font-size: 0.95rem; margin-top: 0.2rem;">Cuota Justa: <b>{fair_odds["draw"]:.2f}</b></div>'
                    f'</div>', unsafe_allow_html=True
                )
            with col_p3:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">Victoria {away_team}</div>'
                    f'<div class="metric-value" style="color: #f85149">{probs["away_win"] * 100:.1f}%</div>'
                    f'<div style="color: #8b949e; font-size: 0.95rem; margin-top: 0.2rem;">Cuota Justa: <b>{fair_odds["away_win"]:.2f}</b></div>'
                    f'</div>', unsafe_allow_html=True
                )
                
            st.markdown("<br>", unsafe_allow_html=True)

            # Métricas Extra Contextuales (Goles Esperados Combinados y Tarjetas Proyectadas por Altitud)
            defaults = get_advanced_global_defaults(adv_data)
            h_adv = get_team_advanced_stats(home_team, adv_data, defaults)
            a_adv = get_team_advanced_stats(away_team, adv_data, defaults)

            base_cards = h_adv["tarjetas_amarillas_por_partido"] + a_adv["tarjetas_amarillas_por_partido"]
            
            # Incorporar el multiplicador de árbitro central
            if match_url and lineup_data is not None:
                base_cards = base_cards * referee_multiplier
                ref_info = f" | Árbitro {scraped_referee}: x{referee_multiplier:.2f}"
            else:
                ref_info = ""

            if altitude > 1000.0:
                projected_cards = base_cards * 1.15
                card_suffix = f" (Fatiga por Hipoxia: +15% en {selected_city} a {altitude:.0f}m{ref_info})"
            else:
                projected_cards = base_cards
                card_suffix = f" (Sede a baja altitud: {altitude:.0f}m{ref_info})"

            col_x1, col_x2 = st.columns(2)
            with col_x1:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">Total de Goles Esperados Combinado</div>'
                    f'<div class="metric-value" style="color: #ffd700">{(l_home + l_away):.2f} xG</div>'
                    f'<div style="color: #8b949e; font-size: 0.95rem; margin-top: 0.2rem;">'
                    f'Cuotas Justas: L <b>{fair_odds["home_win"]:.2f}</b> | E <b>{fair_odds["draw"]:.2f}</b> | V <b>{fair_odds["away_win"]:.2f}</b>'
                    f'</div>'
                    f'</div>', unsafe_allow_html=True
                )
            with col_x2:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">Línea de Tarjetas Totales Proyectadas</div>'
                    f'<div class="metric-value" style="color: #e0a96d">{projected_cards:.2f}</div>'
                    f'<div style="color: #8b949e; font-size: 0.95rem; margin-top: 0.2rem;">'
                    f'Suma base: {base_cards:.2f}{card_suffix}'
                    f'</div>'
                    f'</div>', unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- SECCIÓN: INTEGRACIÓN DE CUOTAS DE CASAS DE APUESTAS Y VENTAJAS ---
            st.markdown("### 💰 Cuotas de Mercado y Ventajas Cuantitativas")
            col_b1, col_b2 = st.columns(2)
            
            with col_b1:
                st.markdown(
                    f'<div class="card" style="border-color: #58a6ff;">'
                    f'<h4> Pinnacle (Margen Real: 2.5%)</h4>'
                    f'<p style="color: #8b949e; font-size: 0.9rem;">El mercado más eficiente y profesional del mundo. Menor comisión, mayor dificultad para batir.</p>'
                    f'<hr style="margin: 0.5rem 0; border-color: rgba(48,54,65,0.8);">'
                    f'<b>Victoria {home_team}:</b> Cuota: <span style="color:#58a6ff; font-weight:bold;">{odds_pinnacle["home_win"]:.2f}</span> | EV: <span style="color:{"#3fb950" if ev_pinnacle["home_win"] > 0 else "#8b949e"};">{ev_pinnacle["home_win"]*100:+.1f}%</span> | CLV: <span style="color:{"#3fb950" if clv_pinnacle["home_win"] > 0 else "#8b949e"};">{clv_pinnacle["home_win"]*100:+.1f}%</span><br>'
                    f'<b>Empate:</b> Cuota: <span style="color:#58a6ff; font-weight:bold;">{odds_pinnacle["draw"]:.2f}</span> | EV: <span style="color:{"#3fb950" if ev_pinnacle["draw"] > 0 else "#8b949e"};">{ev_pinnacle["draw"]*100:+.1f}%</span> | CLV: <span style="color:{"#3fb950" if clv_pinnacle["draw"] > 0 else "#8b949e"};">{clv_pinnacle["draw"]*100:+.1f}%</span><br>'
                    f'<b>Victoria {away_team}:</b> Cuota: <span style="color:#58a6ff; font-weight:bold;">{odds_pinnacle["away_win"]:.2f}</span> | EV: <span style="color:{"#3fb950" if ev_pinnacle["away_win"] > 0 else "#8b949e"};">{ev_pinnacle["away_win"]*100:+.1f}%</span> | CLV: <span style="color:{"#3fb950" if clv_pinnacle["away_win"] > 0 else "#8b949e"};">{clv_pinnacle["away_win"]*100:+.1f}%</span>'
                    f'</div>', unsafe_allow_html=True
                )
                
            with col_b2:
                st.markdown(
                    f'<div class="card" style="border-color: #3fb950;">'
                    f'<h4> Bet365 (Margen Comercial: 5.0%)</h4>'
                    f'<p style="color: #8b949e; font-size: 0.9rem;">Bookmaker comercial/recreativo estándar. Mayor comisión (margen), pero cuotas más lentas en ajustarse.</p>'
                    f'<hr style="margin: 0.5rem 0; border-color: rgba(48,54,65,0.8);">'
                    f'<b>Victoria {home_team}:</b> Cuota: <span style="color:#3fb950; font-weight:bold;">{odds_bet365["home_win"]:.2f}</span> | EV: <span style="color:{"#3fb950" if ev_bet365["home_win"] > 0 else "#8b949e"};">{ev_bet365["home_win"]*100:+.1f}%</span> | CLV: <span style="color:{"#3fb950" if clv_bet365["home_win"] > 0 else "#8b949e"};">{clv_bet365["home_win"]*100:+.1f}%</span><br>'
                    f'<b>Empate:</b> Cuota: <span style="color:#3fb950; font-weight:bold;">{odds_bet365["draw"]:.2f}</span> | EV: <span style="color:{"#3fb950" if ev_bet365["draw"] > 0 else "#8b949e"};">{ev_bet365["draw"]*100:+.1f}%</span> | CLV: <span style="color:{"#3fb950" if clv_bet365["draw"] > 0 else "#8b949e"};">{clv_bet365["draw"]*100:+.1f}%</span><br>'
                    f'<b>Victoria {away_team}:</b> Cuota: <span style="color:#3fb950; font-weight:bold;">{odds_bet365["away_win"]:.2f}</span> | EV: <span style="color:{"#3fb950" if ev_bet365["away_win"] > 0 else "#8b949e"};">{ev_bet365["away_win"]*100:+.1f}%</span> | CLV: <span style="color:{"#3fb950" if clv_bet365["away_win"] > 0 else "#8b949e"};">{clv_bet365["away_win"]*100:+.1f}%</span>'
                    f'</div>', unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- SECCIÓN: GENERADOR DE PICKS AUTOMÁTICOS Y TABLA COMPARATIVA ---
            st.markdown("### 📊 Generador de Picks y Análisis de Ventaja (Edge)")
            
            # Calcular cuotas justas y ventajas (Edge) para los 7 mercados
            outcomes = [
                {
                    "name": f"Gana {home_team}",
                    "model_prob": probs["home_win"],
                    "market_odds": odds_house_home,
                },
                {
                    "name": "Empate",
                    "model_prob": probs["draw"],
                    "market_odds": odds_house_draw,
                },
                {
                    "name": f"Gana {away_team}",
                    "model_prob": probs["away_win"],
                    "market_odds": odds_house_away,
                },
                {
                    "name": "Over 2.5 Goles",
                    "model_prob": p_over,
                    "market_odds": odds_house_over,
                },
                {
                    "name": "Under 2.5 Goles",
                    "model_prob": p_under,
                    "market_odds": odds_house_under,
                },
                {
                    "name": "Ambos Anotan (Sí)",
                    "model_prob": p_btts_yes,
                    "market_odds": odds_house_btts_yes,
                },
                {
                    "name": "Ambos Anotan (No)",
                    "model_prob": p_btts_no,
                    "market_odds": odds_house_btts_no,
                }
            ]
            
            rows_html = ""
            for item in outcomes:
                prob = item["model_prob"]
                odds = item["market_odds"]
                fair_odds_val = 1.0 / prob if prob > 0 else 999.0
                implied_prob = 1.0 / odds if odds > 0 else 0.0
                edge = (prob * odds) - 1.0 if odds > 0 else 0.0
                
                is_pick = edge >= 0.03
                
                # Formatear valores
                model_pct = f"{prob * 100:.1f}%"
                market_pct_val = implied_prob * 100
                market_str = f"{market_pct_val:.1f}% ({odds:.2f})"
                fair_odds_str = f"{fair_odds_val:.2f}"
                edge_str = f"{edge * 100:+.1f}%"
                
                # Estilo de fila
                if is_pick:
                    row_style = "background-color: rgba(46, 160, 67, 0.12); border-left: 4px solid #3fb950;"
                    market_name = f"<span style='color: #3fb950; font-weight: 600;'>🔥 [PICK] {item['name']}</span>"
                    edge_style = "color: #3fb950; font-weight: bold;"
                else:
                    row_style = "border-bottom: 1px solid rgba(48, 54, 65, 0.5);"
                    market_name = item["name"]
                    if edge > 0:
                        edge_style = "color: #58a6ff;"
                    else:
                        edge_style = "color: #8b949e;"
                
                rows_html += f"""<tr style="{row_style}">
<td style="padding: 0.75rem 1rem; vertical-align: middle;">{market_name}</td>
<td style="padding: 0.75rem 1rem; vertical-align: middle;">{model_pct}</td>
<td style="padding: 0.75rem 1rem; vertical-align: middle;">{market_str}</td>
<td style="padding: 0.75rem 1rem; vertical-align: middle; font-weight: 600;">{fair_odds_str}</td>
<td style="padding: 0.75rem 1rem; vertical-align: middle; {edge_style}">{edge_str}</td>
</tr>"""
            
            table_html = f"""<div class="card" style="border: 1px solid rgba(88, 166, 255, 0.3); background: rgba(22, 27, 34, 0.8); padding: 1.5rem; margin-bottom: 1.5rem;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; gap: 10px;">
<h4 style="margin: 0; color: #58a6ff;">📋 Comparativa de Probabilidades y Picks de Valor</h4>
<span style="font-weight: 600; color: #ff8533; font-size: 1.1rem; background: rgba(255, 133, 51, 0.1); padding: 0.3rem 0.8rem; border-radius: 6px; border: 1px solid rgba(255, 133, 51, 0.3);">
xG Modelo: {home_team} {l_home:.2f} - {l_away:.2f} {away_team}
</span>
</div>
<div style="overflow-x: auto;">
<table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.95rem;">
<thead>
<tr style="border-bottom: 2px solid rgba(48, 54, 65, 0.8); color: #8b949e; font-weight: 600;">
<th style="padding: 0.75rem 1rem;">MERCADO</th>
<th style="padding: 0.75rem 1rem;">MODELO (%)</th>
<th style="padding: 0.75rem 1rem;">MERCADO (%) (CUOTA)</th>
<th style="padding: 0.75rem 1rem;">CUOTA JUSTA</th>
<th style="padding: 0.75rem 1rem;">EDGE (VENTAJA)</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
</div>
</div>"""
            st.markdown(table_html, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Distribución y Matriz de Marcadores
            col_left, col_right = st.columns([6, 5])
            
            with col_left:
                st.subheader("Matriz de Probabilidades de Marcadores Exactos")
                # Graficar matriz de calor con Plotly
                max_g = 6  # limitamos a 5-5 para visualización
                vis_matrix = matrix[:max_g, :max_g] * 100  # En porcentaje
                
                fig = px.imshow(
                    vis_matrix,
                    labels=dict(x=f"Goles de {away_team}", y=f"Goles de {home_team}", color="Probabilidad (%)"),
                    x=[str(i) for i in range(max_g)],
                    y=[str(i) for i in range(max_g)],
                    color_continuous_scale="Viridis",
                    aspect="auto",
                    text_auto=".1f"
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#f0f6fc",
                    title_text=f"Probabilidad de cada resultado (%)",
                    title_x=0.5
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col_right:
                st.subheader("Top 5 Marcadores Más Probables")
                
                # Crear tabla limpia
                score_data = []
                for idx, (gh, ga, p) in enumerate(most_likely):
                    # Identificar resultado
                    if gh > ga:
                        result_text = f"Victoria {home_team}"
                        badge_style = "prob-win"
                    elif gh < ga:
                        result_text = f"Victoria {away_team}"
                        badge_style = "prob-loss"
                    else:
                        result_text = "Empate"
                        badge_style = "prob-draw"
                        
                    score_data.append({
                        "Ranking": f"#{idx+1}",
                        "Marcador Exacto": f"**{gh} - {ga}**",
                        "Resultado": result_text,
                        "Probabilidad": f"{p * 100:.2f}%",
                        "_badge": badge_style
                    })
                
                # Mostrar en formato visual personalizado
                for row in score_data:
                    st.markdown(
                        f"<div class='card' style='margin-bottom:0.8rem; padding: 0.8rem 1.2rem; display:flex; justify-content:space-between; align-items:center;'>"
                        f"<div><span style='font-size:1.3rem;'>{row['Marcador Exacto']}</span>"
                        f" <span class='prob-badge {row['_badge']}' style='margin-left: 1rem;'>{row['Resultado']}</span></div>"
                        f"<div style='font-size:1.3rem; font-weight:800; color:#58a6ff'>{row['Probabilidad']}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                
                # Nota adicional
                st.info("💡 La probabilidad de los marcadores se calcula combinando las distribuciones de Poisson independientes para los goles de cada equipo basándose en sus rendimientos respectivos.")

                # --- BOTÓN DE GUARDAR PREDICCIÓN ---
                prediction_data = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "home_team": home_team,
                    "away_team": away_team,
                    "model_choice": model_choice,
                    "venue": f"{selected_city} ({altitude:.0f}m)",
                    "neutral": neutral,
                    "xg_home": round(float(l_home), 2),
                    "xg_away": round(float(l_away), 2),
                    "prob_home": round(float(probs["home_win"] * 100), 1),
                    "prob_draw": round(float(probs["draw"] * 100), 1),
                    "prob_away": round(float(probs["away_win"] * 100), 1),
                    "fair_odds_home": round(float(fair_odds["home_win"]), 2),
                    "fair_odds_draw": round(float(fair_odds["draw"]), 2),
                    "fair_odds_away": round(float(fair_odds["away_win"]), 2),
                    "over_25_prob": round(float(p_over * 100), 1),
                    "under_25_prob": round(float(p_under * 100), 1),
                    "btts_yes_prob": round(float(p_btts_yes * 100), 1),
                    "btts_no_prob": round(float(p_btts_no * 100), 1),
                    "referee": scraped_referee if scraped_referee else "No especificado",
                    "cards_projected": round(float(projected_cards), 2),
                    "starting_val_home": round(float(home_starting_val), 1) if home_starting_val else None,
                    "starting_val_away": round(float(away_starting_val), 1) if away_starting_val else None,
                    "odds_house_home": round(float(odds_house_home), 2),
                    "odds_house_draw": round(float(odds_house_draw), 2),
                    "odds_house_away": round(float(odds_house_away), 2),
                    "odds_house_over": round(float(odds_house_over), 2),
                    "odds_house_under": round(float(odds_house_under), 2),
                    "odds_house_btts_yes": round(float(odds_house_btts_yes), 2),
                    "odds_house_btts_no": round(float(odds_house_btts_no), 2),
                    "picks": [
                        item["name"] for item in outcomes 
                        if (float(item["model_prob"]) * float(item["market_odds"]) - 1.0) >= 0.03
                    ],
                    "top_scores": [f"{gh}-{ga} ({p*100:.1f}%)" for gh, ga, p in most_likely]
                }
                
                st.markdown("---")
                col_save1, col_save2, col_save3 = st.columns([1, 2, 1])
                with col_save2:
                    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                    if st.button("💾 Guardar esta Predicción en el Registro", use_container_width=True, type="primary"):
                        save_prediction_to_log(prediction_data)
                        st.success(f"✅ ¡Predicción **{home_team} vs {away_team}** guardada correctamente!")
                    st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# PESTAÑA: REGISTRO DE PREDICCIONES
# ──────────────────────────────────────────────────────────────────────────────
elif page == "📋 Registro de Predicciones":
    st.subheader("📋 Registro Histórico de Predicciones Guardadas")
    st.write(
        "Aquí puedes consultar todas las predicciones de partidos que has guardado. "
        "Se muestran las probabilidades de victoria, empates, cuotas justas, picks recomendados "
        "y los marcadores proyectados más probables."
    )
    
    log_file = "predictions_log.json"
    
    if not os.path.exists(log_file):
        st.info("Aún no tienes predicciones guardadas. ¡Realiza una predicción en la pestaña de partidos y presiona guardar!")
    else:
        import json
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception as e:
            st.error(f"Error al leer el archivo de registros: {e}")
            history = []
            
        if not history:
            st.info("Aún no tienes predicciones guardadas. ¡Realiza una predicción en la pestaña de partidos y presiona guardar!")
        else:
            # Botones superiores de limpieza y descarga
            col_act1, col_act2, col_act3 = st.columns([3, 3, 4])
            
            with col_act1:
                # Botón para borrar todo el historial
                if st.button("🗑️ Vaciar Todo el Registro", use_container_width=True):
                    if os.path.exists(log_file):
                        os.remove(log_file)
                    st.success("¡Registro vaciado correctamente!")
                    time.sleep(1)
                    st.rerun()
                    
            with col_act2:
                # Descargar JSON
                json_string = json.dumps(history, indent=4, ensure_ascii=False)
                st.download_button(
                    label="📥 Descargar JSON",
                    data=json_string,
                    file_name="historial_predicciones.json",
                    mime="application/json",
                    use_container_width=True
                )
                
            with col_act3:
                # Descargar CSV
                # Convertimos el historial a un formato plano simple para CSV
                flat_history = []
                for p in history:
                    flat_history.append({
                        "Fecha": p.get("timestamp"),
                        "Local": p.get("home_team"),
                        "Visitante": p.get("away_team"),
                        "Modelo": p.get("model_choice"),
                        "Sede": p.get("venue"),
                        "xG Local": p.get("xg_home"),
                        "xG Visitante": p.get("xg_away"),
                        "Prob Local (%)": p.get("prob_home"),
                        "Prob Empate (%)": p.get("prob_draw"),
                        "Prob Visitante (%)": p.get("prob_away"),
                        "Cuota Local": p.get("fair_odds_home"),
                        "Cuota Empate": p.get("fair_odds_draw"),
                        "Cuota Visitante": p.get("fair_odds_away"),
                        "Over 2.5 (%)": p.get("over_25_prob"),
                        "Under 2.5 (%)": p.get("under_25_prob"),
                        "Ambos Anotan Sí (%)": p.get("btts_yes_prob"),
                        "Ambos Anotan No (%)": p.get("btts_no_prob"),
                        "Picks": ", ".join(p.get("picks", [])),
                        "Marcadores": ", ".join(p.get("top_scores", []))
                    })
                df_csv = pd.DataFrame(flat_history)
                csv_data = df_csv.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    label="📥 Descargar CSV",
                    data=csv_data,
                    file_name="historial_predicciones.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Formatear tabla de resumen
            summary_list = []
            for idx, p in enumerate(history):
                summary_list.append({
                    "ID": idx + 1,
                    "Fecha y Hora": p.get("timestamp"),
                    "Partido": f"{p.get('home_team')} vs {p.get('away_team')}",
                    "Modelo": p.get("model_choice"),
                    "Sede": p.get("venue").split(" (")[0] if " (" in p.get("venue", "") else p.get("venue"),
                    "xG": f"{p.get('xg_home'):.2f} - {p.get('xg_away'):.2f}",
                    "Probabilidades (L/E/V)": f"{p.get('prob_home')}% / {p.get('prob_draw')}% / {p.get('prob_away')}%",
                    "Picks Recomendados": ", ".join(p.get("picks", []))
                })
            
            df_summary = pd.DataFrame(summary_list)
            st.dataframe(df_summary.set_index("ID"), use_container_width=True)
            
            st.markdown("---")
            st.subheader("🔍 Detalle del Partido Guardado")
            
            # Selector de partido específico
            match_options = [f"#{p['ID']}: {p['Partido']} ({p['Fecha y Hora']})" for p in summary_list]
            selected_match_option = st.selectbox(
                "Selecciona un partido del historial para ver el desglose completo:",
                match_options,
                index=len(match_options) - 1
            )
            
            if selected_match_option:
                selected_idx = int(selected_match_option.split(":")[0].replace("#", "")) - 1
                p = history[selected_idx]
                
                # Botón de eliminar este registro específico
                col_del1, col_del2 = st.columns([10, 2])
                with col_del2:
                    if st.button("❌ Eliminar este Registro", use_container_width=True):
                        history.pop(selected_idx)
                        if history:
                            with open(log_file, "w", encoding="utf-8") as f:
                                json.dump(history, f, indent=4, ensure_ascii=False)
                        else:
                            os.remove(log_file)
                        st.success("¡Registro eliminado!")
                        time.sleep(1)
                        st.rerun()
                
                # Visualización detallada de la predicción en formato de dashboard premium
                st.markdown(f"### {p['home_team']} vs {p['away_team']}")
                st.write(f"📅 **Fecha de guardado**: {p['timestamp']} | ⚙️ **Modelo utilizado**: {p['model_choice']} | 📍 **Sede**: {p['venue']}")
                
                # xG y Árbitro
                col_det1, col_det2, col_det3 = st.columns(3)
                with col_det1:
                    st.metric("Goles Esperados (xG) Local", f"{p['xg_home']:.2f}")
                with col_det2:
                    st.metric("Goles Esperados (xG) Visitante", f"{p['xg_away']:.2f}")
                with col_det3:
                    st.metric("Tarjetas Proyectadas / Árbitro", f"{p['cards_projected']:.2f}", help=f"Árbitro: {p['referee']}")
                
                # Probabilidades y Cuotas Justas
                st.markdown("#### 1X2 Probabilidades y Cuotas Justas")
                col_det_p1, col_det_p2, col_det_p3 = st.columns(3)
                with col_det_p1:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-label">Victoria {p["home_team"]}</div>'
                        f'<div class="metric-value" style="color: #3fb950">{p["prob_home"]:.1f}%</div>'
                        f'<div style="color: #8b949e; font-size: 0.95rem; margin-top: 0.2rem;">Cuota Justa: <b>{p["fair_odds_home"]:.2f}</b></div>'
                        f'</div>', unsafe_allow_html=True
                    )
                with col_det_p2:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-label">Empate</div>'
                        f'<div class="metric-value" style="color: #c9d1d9">{p["prob_draw"]:.1f}%</div>'
                        f'<div style="color: #8b949e; font-size: 0.95rem; margin-top: 0.2rem;">Cuota Justa: <b>{p["fair_odds_draw"]:.2f}</b></div>'
                        f'</div>', unsafe_allow_html=True
                    )
                with col_det_p3:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-label">Victoria {p["away_team"]}</div>'
                        f'<div class="metric-value" style="color: #f85149">{p["prob_away"]:.1f}%</div>'
                        f'<div style="color: #8b949e; font-size: 0.95rem; margin-top: 0.2rem;">Cuota Justa: <b>{p["fair_odds_away"]:.2f}</b></div>'
                        f'</div>', unsafe_allow_html=True
                    )
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Cuotas de Casa vs Cuota Justa (si existen en el registro)
                st.markdown("#### 💰 Comparativa de Cuotas y Ventaja (Edge)")
                col_det_odds1, col_det_odds2, col_det_odds3 = st.columns(3)
                
                odds_h = p.get("odds_house_home", p["fair_odds_home"])
                odds_d = p.get("odds_house_draw", p["fair_odds_draw"])
                odds_a = p.get("odds_house_away", p["fair_odds_away"])
                
                prob_h = p["prob_home"] / 100
                prob_d = p["prob_draw"] / 100
                prob_a = p["prob_away"] / 100
                
                edge_h = (prob_h * odds_h) - 1.0
                edge_d = (prob_d * odds_d) - 1.0
                edge_a = (prob_a * odds_a) - 1.0
                
                with col_det_odds1:
                    st.markdown(
                        f'<div class="metric-card" style="border-color: {"#3fb950" if edge_h >= 0.03 else "rgba(88, 166, 255, 0.2)"};">'
                        f'<div class="metric-label">Victoria {p["home_team"]}</div>'
                        f'<div style="font-size: 1.1rem; color: #8b949e; margin-top: 0.3rem;">'
                        f'Cuota Casa: <b>{odds_h:.2f}</b><br>'
                        f'Cuota Justa: <b>{p["fair_odds_home"]:.2f}</b><br>'
                        f'Ventaja: <span style="color: {"#3fb950" if edge_h > 0 else "#8b949e"}; font-weight: bold;">{edge_h*100:+.1f}%</span>'
                        f'</div>'
                        f'</div>', unsafe_allow_html=True
                    )
                with col_det_odds2:
                    st.markdown(
                        f'<div class="metric-card" style="border-color: {"#3fb950" if edge_d >= 0.03 else "rgba(88, 166, 255, 0.2)"};">'
                        f'<div class="metric-label">Empate</div>'
                        f'<div style="font-size: 1.1rem; color: #8b949e; margin-top: 0.3rem;">'
                        f'Cuota Casa: <b>{odds_d:.2f}</b><br>'
                        f'Cuota Justa: <b>{p["fair_odds_draw"]:.2f}</b><br>'
                        f'Ventaja: <span style="color: {"#3fb950" if edge_d > 0 else "#8b949e"}; font-weight: bold;">{edge_d*100:+.1f}%</span>'
                        f'</div>'
                        f'</div>', unsafe_allow_html=True
                    )
                with col_det_odds3:
                    st.markdown(
                        f'<div class="metric-card" style="border-color: {"#3fb950" if edge_a >= 0.03 else "rgba(88, 166, 255, 0.2)"};">'
                        f'<div class="metric-label">Victoria {p["away_team"]}</div>'
                        f'<div style="font-size: 1.1rem; color: #8b949e; margin-top: 0.3rem;">'
                        f'Cuota Casa: <b>{odds_a:.2f}</b><br>'
                        f'Cuota Justa: <b>{p["fair_odds_away"]:.2f}</b><br>'
                        f'Ventaja: <span style="color: {"#3fb950" if edge_a > 0 else "#8b949e"}; font-weight: bold;">{edge_a*100:+.1f}%</span>'
                        f'</div>'
                        f'</div>', unsafe_allow_html=True
                    )
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Mercados secundarios
                col_det_sec1, col_det_sec2 = st.columns(2)
                with col_det_sec1:
                    over_odds = f"{100/p['over_25_prob']:.2f}" if p.get('over_25_prob', 0) > 0 else "999.00"
                    under_odds = f"{100/p['under_25_prob']:.2f}" if p.get('under_25_prob', 0) > 0 else "999.00"
                    over_under_html = f"""
                    <div class="card" style="min-height: 180px;">
                        <h4 style="margin-top: 0; color: #58a6ff;">⚽ Mercado Over/Under 2.5 Goles</h4>
                        <p style="margin: 0.5rem 0; font-size: 1.05rem;">📈 <b>Over 2.5 Goles</b>: {p['over_25_prob']:.1f}% (Cuota Justa: <b>{over_odds}</b>)</p>
                        <p style="margin: 0.5rem 0; font-size: 1.05rem;">📉 <b>Under 2.5 Goles</b>: {p['under_25_prob']:.1f}% (Cuota Justa: <b>{under_odds}</b>)</p>
                    </div>
                    """
                    st.markdown(over_under_html, unsafe_allow_html=True)
                    
                with col_det_sec2:
                    btts_yes_odds = f"{100/p['btts_yes_prob']:.2f}" if p.get('btts_yes_prob', 0) > 0 else "999.00"
                    btts_no_odds = f"{100/p['btts_no_prob']:.2f}" if p.get('btts_no_prob', 0) > 0 else "999.00"
                    btts_html = f"""
                    <div class="card" style="min-height: 180px;">
                        <h4 style="margin-top: 0; color: #ff8533;">🤝 Mercado Ambos Anotan (BTTS)</h4>
                        <p style="margin: 0.5rem 0; font-size: 1.05rem;">✅ <b>Ambos Anotan (Sí)</b>: {p['btts_yes_prob']:.1f}% (Cuota Justa: <b>{btts_yes_odds}</b>)</p>
                        <p style="margin: 0.5rem 0; font-size: 1.05rem;">❌ <b>Ambos Anotan (No)</b>: {p['btts_no_prob']:.1f}% (Cuota Justa: <b>{btts_no_odds}</b>)</p>
                    </div>
                    """
                    st.markdown(btts_html, unsafe_allow_html=True)
                
                # Picks de Valor y Marcadores
                col_det_pks, col_det_scs = st.columns(2)
                with col_det_pks:
                    picks_list_html = ""
                    if p.get("picks"):
                        for pick in p["picks"]:
                            picks_list_html += f'<p style="margin: 0.5rem 0; font-size: 1.05rem; color: #3fb950;">🟢 <b>{pick}</b></p>'
                    else:
                        picks_list_html = '<p style="margin: 0.5rem 0; font-size: 1.05rem; color: #8b949e;">No se identificaron picks con ventaja suficiente.</p>'
                        
                    picks_html = f"""
                    <div class="card" style="border-color: #3fb950; min-height: 250px;">
                        <h4 style="margin-top: 0; color: #3fb950;">🔥 Picks de Valor Identificados</h4>
                        {picks_list_html}
                    </div>
                    """
                    st.markdown(picks_html, unsafe_allow_html=True)
                    
                with col_det_scs:
                    scores_list_html = ""
                    for score in p["top_scores"]:
                        scores_list_html += f'<p style="margin: 0.5rem 0; font-size: 1.05rem; color: #f0f6fc;">⚽ <b>{score}</b></p>'
                        
                    scores_html = f"""
                    <div class="card" style="min-height: 250px;">
                        <h4 style="margin-top: 0; color: #58a6ff;">🔢 Marcadores Más Probables</h4>
                        {scores_list_html}
                    </div>
                    """
                    st.markdown(scores_html, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# PESTAÑA 3: SIMULAR MUNDIAL 2026
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🏆 Simular Mundial 2026":
    if not st.session_state.models_loaded:
        st.warning("⚠️ Debes entrenar los modelos primero. Ve a la pestaña '🏠 Inicio y Modelos'.")
    else:
        st.subheader("Simulador Completo del Mundial FIFA 2026")
        st.write(
            "Simula las fases de grupos y eliminatorias del Mundial usando el modelo configurado. "
            "Se aplica el formato real de 48 equipos distribuidos en 12 grupos de 4, donde clasifican los "
            "2 primeros de cada grupo y los 8 mejores terceros a la fase de Dieciseisavos de Final (R32)."
        )
        
        sim_model_choice = st.selectbox("Modelo a utilizar para la simulación del torneo:", ["XGBoost (Recomendado)", "Bayesiano MCMC"])
        model_obj = st.session_state.xgb_model if sim_model_choice == "XGBoost (Recomendado)" else st.session_state.bayes_model
        model_key = "xgboost" if sim_model_choice == "XGBoost (Recomendado)" else "bayesian"
        
        if st.button("🎲 Simular Mundial Completo 🏆"):
            # Realizar simulación
            with st.spinner("Simulando partidos de fase de grupos..."):
                sim_res = simulate_group_stage(model_obj, df, model_type=model_key)
                standings = sim_res["standings"]
                qualified = sim_res["qualified_teams"]
                
            with st.spinner("Simulando llaves de eliminación directa..."):
                bracket_res = simulate_tournament_bracket(model_obj, qualified, df, model_type=model_key)
                
            # Guardar en session_state para persistencia
            st.session_state.sim_run = True
            st.session_state.standings = standings
            st.session_state.bracket_res = bracket_res
            st.session_state.group_matches = sim_res["matches"]
            st.session_state.qualified_count = len(qualified)
            st.session_state.sim_model_used = sim_model_choice
            st.success("¡Simulación completada con éxito!")
            
        if st.session_state.get("sim_run", False):
            # Banner de Campeón
            bracket = st.session_state.bracket_res
            st.markdown(
                f"<div style='text-align: center; padding: 2rem; border-radius: 12px; "
                f"background: linear-gradient(135deg, #ffd700 0%, #ff8c00 100%); margin-bottom: 2rem; border: 2px solid #ffd700; box-shadow: 0 10px 30px rgba(255,215,0,0.2)'>"
                f"<h1 style='color: #000; font-size: 3rem; margin:0;'>🏆 ¡{bracket['campeon'].upper()} ES CAMPEÓN! 🏆</h1>"
                f"<h4 style='color: #222; margin: 0.5rem 0 0 0;'>Subcampeón: {bracket['subcampeon']} | Tercer Lugar: {bracket['tercero']}</h4>"
                f"</div>",
                unsafe_allow_html=True
            )
            
            # Selector de fases
            stage = st.selectbox("Selecciona la fase del torneo que deseas ver:", 
                                 ["🏆 Podio y Bracket Final", "📋 Posiciones de Grupos", "⚔️ Resultados de Grupos"])
            
            if stage == "🏆 Podio y Bracket Final":
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Gran Final y 3er Puesto")
                    
                    fin = bracket["Final"]
                    terc = bracket["3rd"]
                    
                    st.markdown("**FINAL**")
                    st.markdown(f"### {fin['local']} {fin['goles_local']} - {fin['goles_visitante']} {fin['visitante']} {fin['penales_str']}")
                    st.markdown(f"**Ganador: {fin['ganador']}**")
                    st.markdown("---")
                    st.markdown("**TERCER PUESTO**")
                    st.markdown(f"#### {terc['local']} {terc['goles_local']} - {terc['goles_visitante']} {terc['visitante']} {terc['penales_str']}")
                    st.markdown(f"**Ganador: {terc['ganador']}**")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Semifinales")
                    for idx, sf in enumerate(bracket["SF"]):
                        st.markdown(f"**Semifinal {idx+1}**")
                        st.markdown(f"{sf['local']} {sf['goles_local']} - {sf['goles_visitante']} {sf['visitante']} {sf['penales_str']}")
                        st.markdown(f"Ganador: **{sf['ganador']}**")
                        if idx < 1: st.markdown("---")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col_b2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Cuartos de Final (QF)")
                    for idx, qf in enumerate(bracket["QF"]):
                        st.markdown(f"**Cuartos {idx+1}**")
                        st.markdown(f"{qf['local']} {qf['goles_local']} - {qf['goles_visitante']} {qf['visitante']} {qf['penales_str']} → **{qf['ganador']}**")
                        if idx < 3: st.markdown("---")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                # Mostrar R16 y R32 colapsables
                with st.expander("🔍 Ver Llaves de Octavos de Final (R16) y Dieciseisavos (R32)"):
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        st.subheader("Octavos de Final")
                        for idx, r16 in enumerate(bracket["R16"]):
                            st.write(f"Match {idx+1}: {r16['local']} {r16['goles_local']} - {r16['goles_visitante']} {r16['visitante']} {r16['penales_str']} → **{r16['ganador']}**")
                    with col_r2:
                        st.subheader("Dieciseisavos de Final (R32)")
                        for idx, r32 in enumerate(bracket["R32"]):
                            st.write(f"Match {idx+1}: {r32['local']} {r32['goles_local']} - {r32['goles_visitante']} {r32['visitante']} {r32['penales_str']} → **{r32['ganador']}**")
                            
            elif stage == "📋 Posiciones de Grupos":
                st.subheader("Tabla de Standings por Grupo")
                st.write("Los 2 mejores de cada grupo y los 8 mejores terceros avanzaron de ronda.")
                
                # Mostrar en cuadricula
                for row_idx in range(4):  # 4 filas x 3 columnas = 12 grupos
                    col_g1, col_g2, col_g3 = st.columns(3)
                    groups_subset = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
                    
                    g1_name = groups_subset[row_idx * 3]
                    g2_name = groups_subset[row_idx * 3 + 1]
                    g3_name = groups_subset[row_idx * 3 + 2]
                    
                    with col_g1:
                        st.markdown(f"#### Grupo {g1_name}")
                        st.dataframe(st.session_state.standings[g1_name].set_index("Equipo"), use_container_width=True)
                    with col_g2:
                        st.markdown(f"#### Grupo {g2_name}")
                        st.dataframe(st.session_state.standings[g2_name].set_index("Equipo"), use_container_width=True)
                    with col_g3:
                        st.markdown(f"#### Grupo {g3_name}")
                        st.dataframe(st.session_state.standings[g3_name].set_index("Equipo"), use_container_width=True)
                        
            elif stage == "⚔️ Resultados de Grupos":
                st.subheader("Historial de partidos jugados en Fase de Grupos")
                sel_group = st.selectbox("Filtrar por grupo:", ["Todos", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"])
                
                group_matches_df = pd.DataFrame(st.session_state.group_matches)
                if sel_group != "Todos":
                    group_matches_df = group_matches_df[group_matches_df["grupo"] == sel_group]
                    
                st.dataframe(
                    group_matches_df.rename(columns={
                        "grupo": "Grupo",
                        "local": "Local",
                        "visitante": "Visitante",
                        "goles_local": "Goles Local",
                        "goles_visitante": "Goles Visitante"
                    }), 
                    use_container_width=True, 
                    hide_index=True
                )

# ──────────────────────────────────────────────────────────────────────────────
# PESTAÑA 4: FUERZA DE EQUIPOS
# ──────────────────────────────────────────────────────────────────────────────
elif page == "📊 Rankings e Intensidad":
    if not st.session_state.models_loaded:
        st.warning("⚠️ Debes entrenar los modelos primero. Ve a la pestaña '🏠 Inicio y Modelos'.")
    else:
        st.subheader("Análisis Global de Fuerzas e Intensidades de Equipos")
        st.write(
            "Visualiza la fuerza de ataque y defensa calculada por el modelo Bayesiano jerárquico "
            "sobre todos los equipos del dataset. Los valores representan desviaciones logarítmicas de goles respecto al promedio."
        )
        
        # Extraer ranking
        rankings = st.session_state.bayes_model.get_team_rankings()
        df_ranks = pd.DataFrame(rankings)
        
        # Filtrar solo equipos clasificados al mundial si el usuario quiere
        only_wc = st.checkbox("Filtrar únicamente equipos clasificados al Mundial 2026", value=True)
        if only_wc:
            df_ranks = df_ranks[df_ranks["team"].isin(WORLD_CUP_2026_TEAMS)].copy().reset_index(drop=True)
            df_ranks.index = df_ranks.index + 1
            
        col_rank_left, col_rank_right = st.columns([5, 7])
        
        with col_rank_left:
            st.write("### Tabla de Ranking Global (Ataque - Defensa)")
            st.dataframe(
                df_ranks.rename(columns={
                    "team": "Equipo",
                    "attack": "Fuerza de Ataque",
                    "defense": "Fuerza de Defensa (Menor es Mejor)",
                    "overall": "Fuerza Combinada"
                })[["Equipo", "Fuerza de Ataque", "Fuerza de Defensa (Menor es Mejor)", "Fuerza Combinada"]],
                use_container_width=True
            )
            
        with col_rank_right:
            st.write("### Gráfico de Dispersión: Ataque vs Defensa")
            st.write("El cuadrante superior derecho representa equipos ofensivos con buena defensa.")
            
            # Graficar
            fig_scatter = px.scatter(
                df_ranks,
                x="attack",
                y="defense",
                text="team",
                labels={"attack": "Fuerza de Ataque (Más a la derecha = Mejor ataque)", 
                        "defense": "Fuerza de Defensa (Más abajo = Mejor defensa)"},
                title="Mapa de Fuerzas del Mundial 2026",
                hover_data=["overall"]
            )
            
            # Invertimos eje Y porque en defensa menor valor es mejor
            fig_scatter.update_yaxes(autorange="reversed")
            
            fig_scatter.update_traces(
                textposition='top center',
                marker=dict(size=12, color=df_ranks['overall'], colorscale='Viridis', showscale=True)
            )
            
            fig_scatter.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#f0f6fc",
                height=600
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# PESTAÑA 5: DIAGNÓSTICOS DEL MODELO
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🔬 Diagnósticos y SHAP":
    if not st.session_state.models_loaded:
        st.warning("⚠️ Debes entrenar los modelos primero. Ve a la pestaña '🏠 Inicio y Modelos'.")
    else:
        st.subheader("Métricas y Diagnósticos de los Modelos")
        
        tab_diag_xgb, tab_diag_bayes = st.tabs(["🌲 XGBoost Metrics & Importancia", "🧮 MCMC Bayesiano Convergencia"])
        
        with tab_diag_xgb:
            st.write("### Importancia de Características (Feature Importance)")
            st.write("Muestra qué variables históricas o del contexto son más decisivas para predecir los goles.")
            
            # Obtener importancia
            imp_df = st.session_state.xgb_model.get_feature_importance()
            
            if not imp_df.empty:
                fig_imp = px.bar(
                    imp_df,
                    x="importance_avg",
                    y="feature",
                    orientation="h",
                    labels={"importance_avg": "Importancia Media", "feature": "Variable"},
                    title="Importancia de Variables en XGBoost (Promedio Local & Visitante)",
                    color="importance_avg",
                    color_continuous_scale="Cividis"
                )
                fig_imp.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#f0f6fc",
                    yaxis={"categoryorder": "total ascending"}
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.warning("No hay datos de importancia disponibles.")
                
            # Métricas de validación cruzada
            st.write("### Evaluación con Validación Cruzada (5-Fold CV)")
            col_cv1, col_cv2 = st.columns(2)
            
            # Obtener scores
            cv_home = st.session_state.xgb_model.cv_scores_home
            cv_away = st.session_state.xgb_model.cv_scores_away
            
            with col_cv1:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">MAE Goles Local (CV)</div>'
                    f'<div class="metric-value">{-cv_home.mean():.3f}</div>'
                    f'<div class="metric-label">Desviación: {cv_home.std():.3f}</div>'
                    f'</div>', unsafe_allow_html=True
                )
            with col_cv2:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">MAE Goles Visitante (CV)</div>'
                    f'<div class="metric-value">{-cv_away.mean():.3f}</div>'
                    f'<div class="metric-label">Desviación: {cv_away.std():.3f}</div>'
                    f'</div>', unsafe_allow_html=True
                )
                
        with tab_diag_bayes:
            st.write("### Diagnósticos de Muestreo MCMC")
            st.write("Monitoreo de convergencia de parámetros base (intercepto, ventaja local, varianzas de ataque/defensa).")
            
            # Obtener diagnósticos
            diags = st.session_state.bayes_model.get_diagnostics()
            
            if diags:
                st.write(f"**Convergencia de cadenas (R-hat < 1.05):** {'✅ Sí, converge' if diags['rhat_ok'] else '⚠️ Posible no convergencia'}")
                st.dataframe(diags["summary"])
                st.info(
                    "💡 La métrica R-hat mide la convergencia de cadenas múltiples de MCMC. "
                    "Un valor cercano a 1.00 (menor a 1.05) indica que el muestreador ha alcanzado el equilibrio "
                    "y las muestras representan fielmente la distribución posterior."
                )
            else:
                st.info("Los diagnósticos detallados no están disponibles en la caché del posterior resumido.")
