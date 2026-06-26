"""
backtest_temporal.py — Auditoría de Data Leakage mediante validación temporal fuera de muestra (Out-of-Sample).
"""

import sys
import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Añadir el directorio raíz al path de python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_and_preprocess, prepare_bayesian_data
from src.feature_engineering import build_training_dataset
from src.xgboost_model import XGBoostGoalModel
from src.bayesian_model import BayesianGoalModel
from src.utils import get_match_probabilities, compute_poisson_matrix, get_most_likely_scores

def calculate_log_loss(probs_list, actual_list):
    """
    Calcula Log Loss multicategoría (1X2) para una lista de predicciones.
    Clases: 0: home_win, 1: draw, 2: away_win
    """
    eps = 1e-15
    losses = []
    for probs, actual in zip(probs_list, actual_list):
        p = np.clip([probs["home_win"], probs["draw"], probs["away_win"]], eps, 1 - eps)
        # Normalizar para asegurar que sumen 1 tras el clip
        p = p / np.sum(p)
        
        y = [0, 0, 0]
        if actual == "home_win":
            y[0] = 1
        elif actual == "draw":
            y[1] = 1
        else:
            y[2] = 1
            
        loss = -np.sum(np.array(y) * np.log(p))
        losses.append(loss)
    return np.mean(losses)

def calculate_brier_score(probs_list, actual_list):
    """
    Calcula Brier Score multicategoría (1X2).
    Clases: 0: home_win, 1: draw, 2: away_win
    """
    scores = []
    for probs, actual in zip(probs_list, actual_list):
        p = np.array([probs["home_win"], probs["draw"], probs["away_win"]])
        y = np.array([0, 0, 0])
        if actual == "home_win":
            y[0] = 1
        elif actual == "draw":
            y[1] = 1
        else:
            y[2] = 1
            
        score = np.sum((p - y) ** 2)
        scores.append(score)
    return np.mean(scores)

def main():
    print("=== INICIANDO AUDITORÍA DE DATA LEAKAGE Y VALIDACIÓN TEMPORAL ===")
    
    # 1. Cargar y ordenar dataset completo
    df = load_and_preprocess(min_year=2018)
    df_sorted = df.sort_values("date").reset_index(drop=True)
    
    # 2. Aplicar división temporal estricta
    split_date = pd.Timestamp("2025-01-01")
    df_train = df_sorted[df_sorted["date"] < split_date].copy()
    df_test = df_sorted[df_sorted["date"] >= split_date].copy()
    
    print(f"\nDivisión Temporal:")
    print(f"  - TRAIN SET (2018-2024): {len(df_train)} partidos")
    print(f"  - TEST SET  (2025-2026): {len(df_test)} partidos")
    
    if len(df_test) == 0:
        print("Error: No hay partidos registrados en el test set (2025-2026).")
        return
        
    # 3. Construir features de entrenamiento
    print("\n[1/3] Construyendo features del conjunto de entrenamiento...")
    train_features = build_training_dataset(df_train)
    
    # 4. Entrenar modelos con el conjunto aislado (In-Sample 2018-2024)
    print("\n[2/3] Entrenando modelos con datos aislados de entrenamiento...")
    
    xgb_model = XGBoostGoalModel()
    xgb_metrics = xgb_model.fit(train_features)
    # Forzar CPU inmediatamente después del entrenamiento para evitar bloqueos
    xgb_model.model_home.set_params(device="cpu", n_jobs=1)
    xgb_model.model_away.set_params(device="cpu", n_jobs=1)
    
    bayes_model = BayesianGoalModel()
    bayes_data = prepare_bayesian_data(df_train)
    bayes_model.fit(bayes_data, method="map")
    
    # 5. Evaluar en el conjunto de prueba (Validación Ciega Out-of-Sample)
    print("\n[3/3] Evaluando predicciones ciegas en el conjunto de prueba (2025-2026)...")
    
    results = []
    
    for idx, row in df_test.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        actual_home = int(row["home_score"])
        actual_away = int(row["away_score"])
        neutral = row["neutral"]
        altitude = row.get("altitude", 0.0)
        
        # Historial de partidos previos a este encuentro (evita data leakage de partidos del futuro)
        df_prior = df_sorted.iloc[:idx]
        
        actual_res = "home_win" if actual_home > actual_away else ("away_win" if actual_home < actual_away else "draw")
        actual_goals = actual_home + actual_away
        
        # A. Predicción XGBoost
        try:
            pred_xgb = xgb_model.predict(df_prior, home_team, away_team, neutral=neutral, altitude=altitude)
            l_home_xgb, l_away_xgb = pred_xgb["xg_home"], pred_xgb["xg_away"]
        except Exception as e:
            logger.warning(f"Predicción XGBoost falló para {home_team} vs {away_team}: {e}")
            l_home_xgb, l_away_xgb = 1.2, 1.1 # fallback
            
        matrix_xgb = compute_poisson_matrix(l_home_xgb, l_away_xgb)
        probs_xgb = get_match_probabilities(matrix_xgb)
        pred_res_xgb = max(probs_xgb, key=probs_xgb.get)
        most_likely_xgb = get_most_likely_scores(matrix_xgb, top_n=1)[0]
        pred_home_xgb, pred_away_xgb = most_likely_xgb[0], most_likely_xgb[1]
        
        # B. Predicción Bayesiana
        try:
            pred_bay = bayes_model.predict(home_team, away_team, neutral=neutral)
            l_home_bay, l_away_bay = pred_bay["xg_home"], pred_bay["xg_away"]
        except Exception as e:
            logger.warning(f"Predicción Bayesiana falló para {home_team} vs {away_team}: {e}")
            l_home_bay, l_away_bay = 1.2, 1.1
            
        matrix_bay = compute_poisson_matrix(l_home_bay, l_away_bay)
        probs_bay = get_match_probabilities(matrix_bay)
        pred_res_bay = max(probs_bay, key=probs_bay.get)
        most_likely_bay = get_most_likely_scores(matrix_bay, top_n=1)[0]
        pred_home_bay, pred_away_bay = most_likely_bay[0], most_likely_bay[1]
        
        # C. Predicción Híbrida (Ensemble Pesado: 75% Bayes + 25% XGBoost)
        l_home_avg = (0.25 * l_home_xgb) + (0.75 * l_home_bay)
        l_away_avg = (0.25 * l_away_xgb) + (0.75 * l_away_bay)
        matrix_avg = compute_poisson_matrix(l_home_avg, l_away_avg)
        probs_avg = get_match_probabilities(matrix_avg)
        pred_res_avg = max(probs_avg, key=probs_avg.get)
        most_likely_avg = get_most_likely_scores(matrix_avg, top_n=1)[0]
        pred_home_avg, pred_away_avg = most_likely_avg[0], most_likely_avg[1]
        
        results.append({
            "actual_res": actual_res,
            "actual_home": actual_home,
            "actual_away": actual_away,
            "actual_goals": actual_goals,
            # XGBoost
            "probs_xgb": probs_xgb,
            "pred_res_xgb": pred_res_xgb,
            "pred_home_xgb": pred_home_xgb,
            "pred_away_xgb": pred_away_xgb,
            "xg_home_xgb": l_home_xgb,
            "xg_away_xgb": l_away_xgb,
            # Bayes
            "probs_bay": probs_bay,
            "pred_res_bay": pred_res_bay,
            "pred_home_bay": pred_home_bay,
            "pred_away_bay": pred_away_bay,
            "xg_home_bay": l_home_bay,
            "xg_away_bay": l_away_bay,
            # Hybrid
            "probs_avg": probs_avg,
            "pred_res_avg": pred_res_avg,
            "pred_home_avg": pred_home_avg,
            "pred_away_avg": pred_away_avg,
            "xg_home_avg": l_home_avg,
            "xg_away_avg": l_away_avg,
        })
        
    res_df = pd.DataFrame(results)
    
    # 6. Recalcular Métricas Reales Out-of-Sample (OOS)
    print("\n================ METRICAS TEMPORALES OUT-OF-SAMPLE (2025-2026) ================")
    
    for label, prefix in [("XGBOOST", "xgb"), ("BAYESIANO MCMC", "bay"), ("HIBRIDO (PROMEDIO)", "avg")]:
        # Global 1X2
        glob_acc = (res_df["actual_res"] == res_df[f"pred_res_{prefix}"]).mean()
        
        # 1X2 para partidos con Over 2.8 xG combinado proyectado
        over_2_8_mask = (res_df[f"xg_home_{prefix}"] + res_df[f"xg_away_{prefix}"]) > 2.8
        over_2_8_df = res_df[over_2_8_mask]
        if len(over_2_8_df) > 0:
            over_2_8_acc = (over_2_8_df["actual_res"] == over_2_8_df[f"pred_res_{prefix}"]).mean()
        else:
            over_2_8_acc = 0.0
            
        # Marcador Exacto para partidos cerrados (actual total goles <= 2.0 OR xG combinado <= 2.0)
        closed_mask = (res_df["actual_goals"] <= 2.0) | ((res_df[f"xg_home_{prefix}"] + res_df[f"xg_away_{prefix}"]) <= 2.0)
        closed_df = res_df[closed_mask]
        if len(closed_df) > 0:
            closed_exact_acc = ((closed_df["actual_home"] == closed_df[f"pred_home_{prefix}"]) & 
                                (closed_df["actual_away"] == closed_df[f"pred_away_{prefix}"])).mean()
        else:
            closed_exact_acc = 0.0
            
        # Global Marcador Exacto
        glob_exact_acc = ((res_df["actual_home"] == res_df[f"pred_home_{prefix}"]) & 
                           (res_df["actual_away"] == res_df[f"pred_away_{prefix}"])).mean()
            
        # Métricas Probabilísticas
        logloss = calculate_log_loss(res_df[f"probs_{prefix}"].values, res_df["actual_res"].values)
        brier = calculate_brier_score(res_df[f"probs_{prefix}"].values, res_df["actual_res"].values)
        
        print(f"\n--- MODELO: {label} ---")
        print(f"  1X2 Precisión Global: {glob_acc * 100:.2f}% (N={len(res_df)})")
        print(f"  1X2 Precisión en Goleadas (Proyectado > 2.8 xG): {over_2_8_acc * 100:.2f}% (N={len(over_2_8_df)})")
        print(f"  Acierto Marcador Exacto Global: {glob_exact_acc * 100:.2f}%")
        print(f"  Acierto Marcador Exacto Cerrados (xG o Real <= 2.0): {closed_exact_acc * 100:.2f}% (N={len(closed_df)})")
        print(f"  Log Loss Multiclase (1X2): {logloss:.4f}")
        print(f"  Brier Score Multiclase (1X2): {brier:.4f}")

if __name__ == "__main__":
    main()
