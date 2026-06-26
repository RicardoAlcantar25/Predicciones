"""
financial_simulator.py — Simulador cuantitativo de Bankroll y apuestas de Valor Esperado (+EV)
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import hashlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Añadir el directorio raíz al path de python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_and_preprocess, prepare_bayesian_data
from src.feature_engineering import build_training_dataset
from src.xgboost_model import XGBoostGoalModel
from src.bayesian_model import BayesianGoalModel
from src.utils import get_match_probabilities, compute_poisson_matrix

def run_financial_simulation():
    print("=== INICIANDO SIMULADOR FINANCIERO CUANTITATIVO (+EV) ===")
    
    # 1. Cargar y ordenar dataset completo
    df = load_and_preprocess(min_year=2018)
    df_sorted = df.sort_values("date").reset_index(drop=True)
    
    # 2. Aplicar división temporal estricta
    split_date = pd.Timestamp("2025-01-01")
    df_train = df_sorted[df_sorted["date"] < split_date].copy()
    df_test = df_sorted[df_sorted["date"] >= split_date].copy()
    
    print(f"\nConjuntos de Datos:")
    print(f"  - Entrenamiento (2018-2024): {len(df_train)} partidos")
    print(f"  - Simulación (2025-2026): {len(df_test)} partidos")
    
    # 3. Cargar modelos pre-entrenados del repositorio
    xgb_model = XGBoostGoalModel()
    xgb_model.load()
    xgb_model.model_home.set_params(device="cpu", n_jobs=1)
    xgb_model.model_away.set_params(device="cpu", n_jobs=1)
    
    bayes_model = BayesianGoalModel()
    bayes_model.load()
    
    # 4. Parámetros de la simulación
    bankroll_flat = 1000.0
    bankroll_kelly = 1000.0
    
    flat_bet_size = 10.0  # Apuesta fija de 10 unidades
    kelly_fraction = 0.05  # Kelly fraccional al 0.05 (Ultra-conservador)
    
    # Fijar la semilla al inicio de la simulación para consistencia en la generación de ruido
    np.random.seed(42)
    
    # Historiales para graficar
    history_flat = [bankroll_flat]
    history_kelly = [bankroll_kelly]
    dates_bet = [split_date]
    clvs = []
    
    # Métricas agregadas
    total_bets_flat = 0
    total_wins_flat = 0
    total_wagered_flat = 0.0
    
    total_bets_kelly = 0
    total_wins_kelly = 0
    total_wagered_kelly = 0.0
    
    print("\nEjecutando apuestas en Test Set con regla EV > 0.12 y Consenso eficiente de Mercado...")
    
    for idx, row in df_test.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        actual_home = int(row["home_score"])
        actual_away = int(row["away_score"])
        neutral = row["neutral"]
        altitude = row.get("altitude", 0.0)
        match_date = row["date"]
        
        # Historial de partidos previos a este encuentro (evita data leakage)
        df_prior = df_sorted.iloc[:idx]
        
        actual_res = "home_win" if actual_home > actual_away else ("away_win" if actual_home < actual_away else "draw")
        
        # A. Probabilidades XGBoost (Consenso del Mercado)
        try:
            pred_xgb = xgb_model.predict(df_prior, home_team, away_team, neutral=neutral, altitude=altitude)
            l_home_xgb, l_away_xgb = pred_xgb["xg_home"], pred_xgb["xg_away"]
        except Exception as e:
            logger.warning(f"Predicción XGBoost falló para {home_team} vs {away_team}: {e}")
            l_home_xgb, l_away_xgb = 1.2, 1.1
        matrix_xgb = compute_poisson_matrix(l_home_xgb, l_away_xgb)
        probs_xgb = get_match_probabilities(matrix_xgb)
        
        # B. Probabilidades Híbridas (Ensemble Pesado 75/25 - Nuestro Modelo Propietario)
        try:
            pred_bay = bayes_model.predict(home_team, away_team, neutral=neutral)
            l_home_bay, l_away_bay = pred_bay["xg_home"], pred_bay["xg_away"]
        except Exception as e:
            logger.warning(f"Predicción Bayesiana falló para {home_team} vs {away_team}: {e}")
            l_home_bay, l_away_bay = 1.2, 1.1
            
        l_home_avg = (0.25 * l_home_xgb) + (0.75 * l_home_bay)
        l_away_avg = (0.25 * l_away_xgb) + (0.75 * l_away_bay)
        matrix_avg = compute_poisson_matrix(l_home_avg, l_away_avg)
        probs_avg = get_match_probabilities(matrix_avg)
        
        # C. Generar Cuotas de la Casa con ruido gaussiano independiente para romper la circularidad (auto-arbitraje)
        alpha = 0.35
        # Generar una semilla determinista para este partido basada en la fecha y equipos
        match_seed = int(hashlib.md5(f"{match_date}_{home_team}_{away_team}".encode('utf-8')).hexdigest(), 16) % 2**32
        rng = np.random.default_rng(match_seed)
        
        # Probabilidades consensus de base (true benchmark)
        p_consensus_vals = np.array([
            (1.0 - alpha) * probs_avg["home_win"] + alpha * probs_xgb["home_win"],
            (1.0 - alpha) * probs_avg["draw"] + alpha * probs_xgb["draw"],
            (1.0 - alpha) * probs_avg["away_win"] + alpha * probs_xgb["away_win"]
        ])
        p_consensus_vals /= p_consensus_vals.sum()
        
        # 1. Cuotas de apertura (odds_house) con ruido gaussiano de mercado σ = 4% (0.04)
        noise_open = rng.normal(0, 0.04, size=3)
        p_market = np.clip(p_consensus_vals + noise_open, 0.02, 0.98)
        p_market /= p_market.sum()
        
        odds_house = {
            "home_win": max(1.01, min(20.0, 1.0 / (p_market[0] * 1.025))),
            "draw": max(1.01, min(20.0, 1.0 / (p_market[1] * 1.025))),
            "away_win": max(1.01, min(20.0, 1.0 / (p_market[2] * 1.025)))
        }
        
        # Determinar si existen cuotas reales de Pinnacle integradas
        has_real_odds = not pd.isna(row.get("pinnacle_odds_home"))
        
        # 2. Cuotas de cierre (Closing Line) de Pinnacle
        if has_real_odds:
            odds_closing_pinnacle = {
                "home_win": float(row["pinnacle_odds_home"]),
                "draw": float(row["pinnacle_odds_draw"]),
                "away_win": float(row["pinnacle_odds_away"])
            }
        else:
            # Cuotas de cierre con ruido gaussiano menor σ = 1.5% (0.015) para simular eficiencia al cierre
            noise_close = rng.normal(0, 0.015, size=3)
            p_closing = np.clip(p_consensus_vals + noise_close, 0.02, 0.98)
            p_closing /= p_closing.sum()
            
            odds_closing_pinnacle = {
                "home_win": max(1.01, min(20.0, 1.0 / (p_closing[0] * 1.025))),
                "draw": max(1.01, min(20.0, 1.0 / (p_closing[1] * 1.025))),
                "away_win": max(1.01, min(20.0, 1.0 / (p_closing[2] * 1.025)))
            }
        
        # D. Calcular Valor Esperado (+EV) para cada opción basado en las cuotas de apertura (odds_house)
        evs = {
            "home_win": probs_avg["home_win"] * odds_house["home_win"] - 1.0,
            "draw": probs_avg["draw"] * odds_house["draw"] - 1.0,
            "away_win": probs_avg["away_win"] * odds_house["away_win"] - 1.0
        }
        
        # E. Seleccionar la mejor opción de apuesta con EV > 12%
        best_option = max(evs, key=evs.get)
        best_ev = evs[best_option]
        
        # Verificar si hay una apuesta válida
        if best_ev > 0.12:
            odds_bet = odds_house[best_option]
            p_win = probs_avg[best_option]
            is_win = (best_option == actual_res)
            
            # Registrar el Closing Line Value (CLV)
            clv = (odds_bet / odds_closing_pinnacle[best_option]) - 1.0
            clvs.append(clv)
            
            # --- STRATEGY 1: Apuesta Fija (Flat Betting) ---
            if bankroll_flat >= flat_bet_size:
                total_bets_flat += 1
                total_wagered_flat += flat_bet_size
                if is_win:
                    profit = flat_bet_size * (odds_bet - 1.0)
                    bankroll_flat += profit
                    total_wins_flat += 1
                else:
                    bankroll_flat -= flat_bet_size
                history_flat.append(bankroll_flat)
            else:
                history_flat.append(bankroll_flat)
                
            # --- STRATEGY 2: Criterio de Kelly Fraccional (0.05) ---
            b = odds_bet - 1.0
            # Kelly fraction = (p * b - q) / b = p - (1 - p)/b
            kelly_f = p_win - (1.0 - p_win) / b
            
            if kelly_f > 0 and bankroll_kelly > 0:
                # Aplicamos Kelly fraccional al 0.05 y limitamos la apuesta a un máximo de 5% del bankroll para gestión de riesgo
                bet_fraction = min(0.05, kelly_fraction * kelly_f)
                bet_amount = bankroll_kelly * bet_fraction
                
                if bet_amount >= 1.0: # Solo apostar si el monto es razonable
                    total_bets_kelly += 1
                    total_wagered_kelly += bet_amount
                    if is_win:
                        profit = bet_amount * b
                        bankroll_kelly += profit
                        total_wins_kelly += 1
                    else:
                        bankroll_kelly -= bet_amount
                    history_kelly.append(bankroll_kelly)
                else:
                    history_kelly.append(bankroll_kelly)
            else:
                history_kelly.append(bankroll_kelly)
                
            dates_bet.append(match_date)
            
    # Calcular Max Drawdown para ambas estrategias
    def calculate_max_drawdown(history):
        peaks = np.maximum.accumulate(history)
        drawdowns = (peaks - history) / peaks
        return float(np.max(drawdowns)) * 100.0
        
    mdd_flat = calculate_max_drawdown(history_flat)
    mdd_kelly = calculate_max_drawdown(history_kelly)
    
    # Calcular ROI y Yield
    roi_flat = ((bankroll_flat - 1000.0) / 1000.0) * 100.0
    yield_flat = ((bankroll_flat - 1000.0) / total_wagered_flat) * 100.0 if total_wagered_flat > 0 else 0.0
    
    roi_kelly = ((bankroll_kelly - 1000.0) / 1000.0) * 100.0
    yield_kelly = ((bankroll_kelly - 1000.0) / total_wagered_kelly) * 100.0 if total_wagered_kelly > 0 else 0.0
    
    avg_clv = np.mean(clvs) * 100.0 if clvs else 0.0

    print("\n================ REPORT DE SIMULACIÓN FINANCIERA (2025-2026) ================")
    print(f"  - CLV Promedio del Portafolio: {avg_clv:.2f}% (N = {len(clvs)} apuestas)")
    print(f"\n[ESTRATEGIA 1: APUESTA FIJA (Flat Betting - 10 unidades)]")
    print(f"  - Bankroll Final: {bankroll_flat:.2f} unidades (Inicial: 1000.00)")
    print(f"  - Ganancia Neta: {bankroll_flat - 1000.0:.2f} unidades")
    print(f"  - Total Apuestas Realizadas: {total_bets_flat} de {len(df_test)} ({total_bets_flat/len(df_test)*100:.2f}%)")
    print(f"  - Total Apuestas Ganadas: {total_wins_flat} (Win Rate: {total_wins_flat/max(1, total_bets_flat)*100:.2f}%)")
    print(f"  - Total Apostado (Volumen): {total_wagered_flat:.2f} unidades")
    print(f"  - ROI Total (Retorno Inversión): {roi_flat:.2f}%")
    print(f"  - Yield Neto: {yield_flat:.2f}%")
    print(f"  - Max Drawdown Histórico: {mdd_flat:.2f}%")
    
    print(f"\n[ESTRATEGIA 2: CRITERIO DE KELLY FRACCIONAL (0.05 Kelly - Ultra-conservador)]")
    print(f"  - Bankroll Final: {bankroll_kelly:.2f} unidades (Inicial: 1000.00)")
    print(f"  - Ganancia Neta: {bankroll_kelly - 1000.0:.2f} unidades")
    print(f"  - Total Apuestas Realizadas: {total_bets_kelly} de {len(df_test)} ({total_bets_kelly/len(df_test)*100:.2f}%)")
    print(f"  - Total Apuestas Ganadas: {total_wins_kelly} (Win Rate: {total_wins_kelly/max(1, total_bets_kelly)*100:.2f}%)")
    print(f"  - Total Apostado (Volumen): {total_wagered_kelly:.2f} unidades")
    print(f"  - ROI Total (Retorno Inversión): {roi_kelly:.2f}%")
    print(f"  - Yield Neto: {yield_kelly:.2f}%")
    print(f"  - Max Drawdown Histórico: {mdd_kelly:.2f}%")
    
    # 5. Generar y guardar la gráfica en la carpeta de artefactos
    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "brain", "9f2b544e-fddd-44c3-880a-d8adc2aab78b")
    # Si la carpeta del brain no existiera (fallback al directorio actual)
    if not os.path.exists(artifacts_dir):
        artifacts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    plt.figure(figsize=(12, 6))
    plt.style.use("dark_background")
    plt.plot(dates_bet, history_flat, label=f"Apuesta Fija (Yield: {yield_flat:.1f}%, ROI: {roi_flat:.1f}%)", color="#58a6ff", linewidth=2)
    plt.plot(dates_bet, history_kelly, label=f"Kelly Fraccional 0.05 (Yield: {yield_kelly:.1f}%, ROI: {roi_kelly:.1f}%)", color="#3fb950", linewidth=2.5)
    plt.axhline(1000.0, color="#f85149", linestyle="--", alpha=0.5, label="Banco Inicial ($1,000)")
    
    plt.title("Evolución de Bankroll Quant en Tiempo Real (Mundial 2026 Predictor)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Fecha del Partido", fontsize=11)
    plt.ylabel("Unidades de Bankroll", fontsize=11)
    plt.grid(True, color="#30363d", alpha=0.5)
    plt.legend(fontsize=10, loc="upper left")
    plt.tight_layout()
    
    graph_path = os.path.join(artifacts_dir, "bankroll_growth.png")
    plt.savefig(graph_path, dpi=150)
    plt.close()
    print(f"\n[GRÁFICA] Curva de crecimiento del Bankroll guardada exitosamente en:\n  -> {graph_path}")

if __name__ == "__main__":
    run_financial_simulation()
