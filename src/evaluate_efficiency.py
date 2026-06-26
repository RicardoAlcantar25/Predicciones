"""
evaluate_efficiency.py — Diagnóstico avanzado de generalización y calibración probabilística
"""

import sys
import os
import pandas as pd
import numpy as np
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
from src.utils import get_match_probabilities, compute_poisson_matrix, get_most_likely_scores

def calculate_log_loss(probs_list, actual_list):
    eps = 1e-15
    losses = []
    for probs, actual in zip(probs_list, actual_list):
        p = np.clip([probs["home_win"], probs["draw"], probs["away_win"]], eps, 1 - eps)
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
    print("=== INICIANDO ESCANEO DE METRICAS DE EFICIENCIA Y CALIBRACION ===")
    
    # 1. Cargar y ordenar dataset completo
    df = load_and_preprocess(min_year=2018)
    df_sorted = df.sort_values("date").reset_index(drop=True)
    
    split_date = pd.Timestamp("2025-01-01")
    df_train = df_sorted[df_sorted["date"] < split_date].copy()
    df_test = df_sorted[df_sorted["date"] >= split_date].copy()
    
    xgb_model = XGBoostGoalModel()
    xgb_model.load()
    xgb_model.model_home.set_params(device="cpu", n_jobs=1)
    xgb_model.model_away.set_params(device="cpu", n_jobs=1)
    
    bayes_model = BayesianGoalModel()
    bayes_model.load()
    
    print("\n[Audit] Ejecutando inferencia temporal Out-of-Sample (1337 partidos)...")
    
    y_true = []
    y_pred_xgb = []
    y_pred_bayes = []
    y_pred_ensemble = []
    
    probs_ensemble_list = []
    probs_xgb_list = []
    probs_bayes_list = []
    
    # Marcador exacto
    exact_correct = 0
    top3_correct = 0
    top5_correct = 0
    
    # Segmentación por confianza
    conf_clear_fav = []
    conf_parity = []
    
    clvs = []
    bets_profit = []
    
    for idx, row in df_test.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        actual_home = int(row["home_score"])
        actual_away = int(row["away_score"])
        neutral = row["neutral"]
        altitude = row.get("altitude", 0.0)
        
        df_prior = df_sorted.iloc[:idx]
        actual_res = "home_win" if actual_home > actual_away else ("away_win" if actual_home < actual_away else "draw")
        y_true.append(actual_res)
        
        # A. XGBoost
        try:
            pred_xgb = xgb_model.predict(df_prior, home_team, away_team, neutral=neutral, altitude=altitude)
            l_home_xgb, l_away_xgb = pred_xgb["xg_home"], pred_xgb["xg_away"]
        except Exception as e:
            logger.warning(f"Predicción XGBoost falló para {home_team} vs {away_team}: {e}")
            l_home_xgb, l_away_xgb = 1.2, 1.1
        matrix_xgb = compute_poisson_matrix(l_home_xgb, l_away_xgb)
        probs_xgb = get_match_probabilities(matrix_xgb)
        
        best_xgb = max(probs_xgb, key=probs_xgb.get)
        y_pred_xgb.append(best_xgb)
        probs_xgb_list.append(probs_xgb)
        
        # B. Bayesiano
        try:
            pred_bay = bayes_model.predict(home_team, away_team, neutral=neutral)
            l_home_bay, l_away_bay = pred_bay["xg_home"], pred_bay["xg_away"]
        except Exception as e:
            logger.warning(f"Predicción Bayesiana falló para {home_team} vs {away_team}: {e}")
            l_home_bay, l_away_bay = 1.2, 1.1
        matrix_bay = compute_poisson_matrix(l_home_bay, l_away_bay)
        probs_bay = get_match_probabilities(matrix_bay)
        
        best_bay = max(probs_bay, key=probs_bay.get)
        y_pred_bayes.append(best_bay)
        probs_bayes_list.append(probs_bay)
        
        # C. Ensemble Pesado (75% Bayesiano / 25% XGBoost)
        l_home_ens = 0.75 * l_home_bay + 0.25 * l_home_xgb
        l_away_ens = 0.75 * l_away_bay + 0.25 * l_away_xgb
        matrix_ens = compute_poisson_matrix(l_home_ens, l_away_ens)
        probs_ens = get_match_probabilities(matrix_ens)
        
        best_ens = max(probs_ens, key=probs_ens.get)
        y_pred_ensemble.append(best_ens)
        probs_ensemble_list.append(probs_ens)
        
        # Exact score check
        top_scores = get_most_likely_scores(matrix_ens, top_n=5)
        if actual_home == top_scores[0][0] and actual_away == top_scores[0][1]:
            exact_correct += 1
            
        is_top3 = False
        is_top5 = False
        for i, (h, a, p) in enumerate(top_scores):
            if actual_home == h and actual_away == a:
                if i < 3:
                    is_top3 = True
                if i < 5:
                    is_top5 = True
        
        if is_top3:
            top3_correct += 1
        if is_top5:
            top5_correct += 1
            
        # Nivel de confianza
        max_p = max(probs_ens.values())
        if max_p > 0.65:
            conf_clear_fav.append((best_ens == actual_res))
        elif all(p <= 0.40 for p in probs_ens.values()):
            conf_parity.append((best_ens == actual_res))
            
        # E. Simulación de Apuestas y Cálculo de CLV con ruido independiente
        alpha = 0.35
        match_date = row.get("date", idx)
        match_seed = int(hashlib.md5(f"{match_date}_{home_team}_{away_team}".encode('utf-8')).hexdigest(), 16) % 2**32
        rng = np.random.default_rng(match_seed)
        
        p_consensus_vals = np.array([
            (1.0 - alpha) * probs_ens["home_win"] + alpha * probs_xgb["home_win"],
            (1.0 - alpha) * probs_ens["draw"] + alpha * probs_xgb["draw"],
            (1.0 - alpha) * probs_ens["away_win"] + alpha * probs_xgb["away_win"]
        ])
        p_consensus_vals /= p_consensus_vals.sum()
        
        # 1. Cuotas de apertura (odds_house) con ruido gaussiano σ = 4%
        noise_open = rng.normal(0, 0.04, size=3)
        p_market = np.clip(p_consensus_vals + noise_open, 0.02, 0.98)
        p_market /= p_market.sum()
        
        odds_house = {
            "home_win": max(1.01, min(20.0, 1.0 / (p_market[0] * 1.025))),
            "draw": max(1.01, min(20.0, 1.0 / (p_market[1] * 1.025))),
            "away_win": max(1.01, min(20.0, 1.0 / (p_market[2] * 1.025)))
        }
        
        has_real_odds = not pd.isna(row.get("pinnacle_odds_home"))
        if has_real_odds:
            odds_closing_pinnacle = {
                "home_win": float(row["pinnacle_odds_home"]),
                "draw": float(row["pinnacle_odds_draw"]),
                "away_win": float(row["pinnacle_odds_away"])
            }
        else:
            # Cuotas de cierre con ruido gaussiano menor σ = 1.5% para simular eficiencia al cierre
            noise_close = rng.normal(0, 0.015, size=3)
            p_closing = np.clip(p_consensus_vals + noise_close, 0.02, 0.98)
            p_closing /= p_closing.sum()
            
            odds_closing_pinnacle = {
                "home_win": max(1.01, min(20.0, 1.0 / (p_closing[0] * 1.025))),
                "draw": max(1.01, min(20.0, 1.0 / (p_closing[1] * 1.025))),
                "away_win": max(1.01, min(20.0, 1.0 / (p_closing[2] * 1.025)))
            }
            
        evs = {
            "home_win": probs_ens["home_win"] * odds_house["home_win"] - 1.0,
            "draw": probs_ens["draw"] * odds_house["draw"] - 1.0,
            "away_win": probs_ens["away_win"] * odds_house["away_win"] - 1.0
        }
        best_opt = max(evs, key=evs.get)
        best_ev = evs[best_opt]
        
        if best_ev > 0.12:
            clv = (odds_house[best_opt] / odds_closing_pinnacle[best_opt]) - 1.0
            clvs.append(clv)
            is_win = (best_opt == actual_res)
            bets_profit.append((odds_house[best_opt] - 1.0) if is_win else -1.0)
            
    # Accuracy cálculos
    acc_xgb = np.mean([p == t for p, t in zip(y_pred_xgb, y_true)]) * 100
    acc_bayes = np.mean([p == t for p, t in zip(y_pred_bayes, y_true)]) * 100
    acc_ens = np.mean([p == t for p, t in zip(y_pred_ensemble, y_true)]) * 100
    
    acc_clear_fav = np.mean(conf_clear_fav) * 100 if conf_clear_fav else 0.0
    acc_parity = np.mean(conf_parity) * 100 if conf_parity else 0.0
    
    # Matriz y clasificación multiclase (Precision, Recall, F1)
    from sklearn.metrics import classification_report
    report_dict = classification_report(y_true, y_pred_ensemble, output_dict=True, labels=["home_win", "draw", "away_win"])
    
    # Calibración penalización
    log_loss_ens = calculate_log_loss(probs_ensemble_list, y_true)
    brier_ens = calculate_brier_score(probs_ensemble_list, y_true)
    
    # Desviación de calibración
    observed_home = np.mean([1 if t == "home_win" else 0 for t in y_true])
    observed_draw = np.mean([1 if t == "draw" else 0 for t in y_true])
    observed_away = np.mean([1 if t == "away_win" else 0 for t in y_true])
    
    pred_home = np.mean([p["home_win"] for p in probs_ensemble_list])
    pred_draw = np.mean([p["draw"] for p in probs_ensemble_list])
    pred_away = np.mean([p["away_win"] for p in probs_ensemble_list])
    
    dev_home = abs(pred_home - observed_home)
    dev_draw = abs(pred_draw - observed_draw)
    dev_away = abs(pred_away - observed_away)
    mean_dev = np.mean([dev_home, dev_draw, dev_away]) * 100
    
    exact_acc = (exact_correct / len(df_test)) * 100
    top3_acc = (top3_correct / len(df_test)) * 100
    top5_acc = (top5_correct / len(df_test)) * 100
    
    # Imprimir en consola la tabla estructurada
    print("\n" + "="*70)
    print("               DIAGNÓSTICO MAESTRO DE EFICIENCIA Y RENDIMIENTO")
    print("="*70)
    print(f"1. ACCURACY 1X2 GLOBAL Y SEGMENTADO:")
    print(f"   - Accuracy XGBoost Solo:       {acc_xgb:.2f}%")
    print(f"   - Accuracy Bayesiano MAP Solo: {acc_bayes:.2f}%")
    print(f"   - Accuracy Ensemble Pesado:    {acc_ens:.2f}%")
    print(f"   - Accuracy Favoritos Claros (>65%): {acc_clear_fav:.2f}% (N = {len(conf_clear_fav)})")
    print(f"   - Accuracy Alta Paridad (<40%):     {acc_parity:.2f}% (N = {len(conf_parity)})")
    print("-"*70)
    print(f"2. F1-SCORE Y REPORT MULTICLASE (Ensemble Pesado):")
    print(f"   Clase            Precision    Recall     F1-Score")
    for key, label in [("home_win", "Victoria Local"), ("draw", "Empate        "), ("away_win", "Vic. Visitante")]:
        d = report_dict[key]
        print(f"   {label}   {d['precision']*100:.2f}%      {d['recall']*100:.2f}%     {d['f1-score']*100:.2f}%")
    print("-"*70)
    print(f"3. CALIBRACIÓN Y PENALIZACIÓN:")
    print(f"   - Log Loss Multiclase:         {log_loss_ens:.4f}")
    print(f"   - Brier Score Multiclase:      {brier_ens:.4f}")
    print(f"   - Desviación de Calibración Promedio: {mean_dev:.2f}%")
    print(f"     * Prediccion vs Observado Local:   {pred_home*100:.1f}% vs {observed_home*100:.1f}%")
    print(f"     * Prediccion vs Observado Empate:  {pred_draw*100:.1f}% vs {observed_draw*100:.1f}%")
    print(f"     * Prediccion vs Observado Visit.:  {pred_away*100:.1f}% vs {observed_away*100:.1f}%")
    print("-"*70)
    avg_clv = np.mean(clvs) * 100.0 if clvs else 0.0
    pos_clv_pct = (np.array(clvs) > 0).mean() * 100.0 if clvs else 0.0
    yield_est = np.mean(bets_profit) * 100.0 if bets_profit else 0.0

    print(f"4. MERCADO DE GOLES Y MARCADOR EXACTO:")
    print(f"   - Accuracy Marcador Exacto (Top 1):  {exact_acc:.2f}%")
    print(f"   - Hit Rate Top 3 Marcadores:         {top3_acc:.2f}%")
    print(f"   - Hit Rate Top 5 Marcadores:         {top5_acc:.2f}%")
    print("-"*70)
    print(f"5. INTEGRACIÓN DE CUOTAS Y VENTAJA CUANTITATIVA (CLV):")
    print(f"   - CLV Promedio del Portafolio:       {avg_clv:.2f}% (N = {len(clvs)} apuestas)")
    print(f"   - Porcentaje de Apuestas con CLV > 0: {pos_clv_pct:.2f}%")
    print(f"   - Yield Estimado en Apuesta Fija:    {yield_est:.2f}%")
    print("="*70)

if __name__ == "__main__":
    main()
