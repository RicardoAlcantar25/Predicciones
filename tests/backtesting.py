import sys
import os
import pandas as pd
import numpy as np

# Add project directory to python path
sys.path.append(r"C:\Users\richi\.gemini\antigravity-ide\scratch\world-cup-2026-predictor")

from src.data_loader import load_and_preprocess
from src.xgboost_model import XGBoostGoalModel
from src.bayesian_model import BayesianGoalModel
from src.utils import get_match_probabilities, compute_poisson_matrix, get_most_likely_scores

def run_backtest(n_matches=100):
    print(f"=== INICIANDO BACKTESTING DE MODELOS EN LOS ULTIMOS {n_matches} PARTIDOS ===")
    
    # 1. Cargar datos
    df = load_and_preprocess()
    
    # Ordenar por fecha y tomar los últimos partidos jugados
    df_sorted = df.sort_values("date").reset_index(drop=True)
    # Tomamos los últimos n_matches jugados
    test_set = df_sorted.tail(n_matches).copy()
    
    print(f"Muestra de prueba seleccionada desde: {test_set['date'].min().strftime('%Y-%m-%d')} hasta {test_set['date'].max().strftime('%Y-%m-%d')}")
    
    # 2. Cargar modelos
    xgb_model = XGBoostGoalModel()
    xgb_model.load()
    
    bayes_model = BayesianGoalModel()
    bayes_model.load()
    
    # 3. Listas para recolectar resultados
    results = []
    
    for idx, row in test_set.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        actual_home = int(row["home_score"])
        actual_away = int(row["away_score"])
        neutral = row["neutral"]
        altitude = row.get("altitude", 0.0)
        
        # Determinar resultado real
        if actual_home > actual_away:
            actual_res = "home_win"
        elif actual_home < actual_away:
            actual_res = "away_win"
        else:
            actual_res = "draw"
            
        actual_total_goals = actual_home + actual_away
        
        # --- Predicción XGBoost ---
        try:
            pred_xgb = xgb_model.predict(df_sorted, home_team, away_team, neutral=neutral, altitude=altitude)
            l_home_xgb, l_away_xgb = pred_xgb["xg_home"], pred_xgb["xg_away"]
        except Exception:
            l_home_xgb, l_away_xgb = 1.2, 1.1 # Default fallback
            
        matrix_xgb = compute_poisson_matrix(l_home_xgb, l_away_xgb)
        probs_xgb = get_match_probabilities(matrix_xgb)
        pred_res_xgb = max(probs_xgb, key=probs_xgb.get)
        # Marcador exacto más probable
        most_likely_xgb = get_most_likely_scores(matrix_xgb, top_n=1)[0]
        pred_home_xgb, pred_away_xgb = most_likely_xgb[0], most_likely_xgb[1]
        
        # --- Predicción Bayesiana ---
        try:
            pred_bay = bayes_model.predict(home_team, away_team, neutral=neutral)
            l_home_bay, l_away_bay = pred_bay["xg_home"], pred_bay["xg_away"]
        except Exception:
            l_home_bay, l_away_bay = 1.2, 1.1
            
        matrix_bay = compute_poisson_matrix(l_home_bay, l_away_bay)
        probs_bay = get_match_probabilities(matrix_bay)
        pred_res_bay = max(probs_bay, key=probs_bay.get)
        most_likely_bay = get_most_likely_scores(matrix_bay, top_n=1)[0]
        pred_home_bay, pred_away_bay = most_likely_bay[0], most_likely_bay[1]
        
        # --- Predicción Híbrida (Promedio) ---
        l_home_avg = (l_home_xgb + l_home_bay) / 2
        l_away_avg = (l_away_xgb + l_away_bay) / 2
        matrix_avg = compute_poisson_matrix(l_home_avg, l_away_avg)
        probs_avg = get_match_probabilities(matrix_avg)
        pred_res_avg = max(probs_avg, key=probs_avg.get)
        most_likely_avg = get_most_likely_scores(matrix_avg, top_n=1)[0]
        pred_home_avg, pred_away_avg = most_likely_avg[0], most_likely_avg[1]
        
        results.append({
            "home": home_team,
            "away": away_team,
            "actual_home": actual_home,
            "actual_away": actual_away,
            "actual_res": actual_res,
            "actual_goals": actual_total_goals,
            # XGBoost
            "xg_home_xgb": l_home_xgb,
            "xg_away_xgb": l_away_xgb,
            "pred_res_xgb": pred_res_xgb,
            "pred_home_xgb": pred_home_xgb,
            "pred_away_xgb": pred_away_xgb,
            # Bayes
            "xg_home_bay": l_home_bay,
            "xg_away_bay": l_away_bay,
            "pred_res_bay": pred_res_bay,
            "pred_home_bay": pred_home_bay,
            "pred_away_bay": pred_away_bay,
            # Hybrid
            "xg_home_avg": l_home_avg,
            "xg_away_avg": l_away_avg,
            "pred_res_avg": pred_res_avg,
            "pred_home_avg": pred_home_avg,
            "pred_away_avg": pred_away_avg
        })
        
    res_df = pd.DataFrame(results)
    
    # 4. Calcular Métricas
    print("\n--- METRICAS OBTENIDAS ---")
    
    for model_name, prefix in [("XGBoost", "xgb"), ("Bayesiano", "bay"), ("Híbrido (Promedio)", "avg")]:
        trend_acc = (res_df["actual_res"] == res_df[f"pred_res_{prefix}"]).mean()
        score_acc = ((res_df["actual_home"] == res_df[f"pred_home_{prefix}"]) & 
                     (res_df["actual_away"] == res_df[f"pred_away_{prefix}"])).mean()
        mae_home = np.abs(res_df["actual_home"] - res_df[f"xg_home_{prefix}"]).mean()
        mae_away = np.abs(res_df["actual_away"] - res_df[f"xg_away_{prefix}"]).mean()
        mae_combined = (mae_home + mae_away) / 2
        
        print(f"\n[{model_name.upper()}]:")
        print(f"  Precisión Tendencia (1X2): {trend_acc * 100:.2f}%")
        print(f"  Precisión Marcador Exacto: {score_acc * 100:.2f}%")
        print(f"  Desviación Goles (MAE): {mae_combined:.4f} (Local: {mae_home:.3f} | Visita: {mae_away:.3f})")
        
    # 5. Diagnóstico de Sesgo y Escenarios Especiales (Usando el modelo Híbrido)
    print("\n--- DIAGNOSTICO DE SESGOS Y RENDIMIENTO ESPECIFICO (MODELO HIBRIDO) ---")
    
    # Escenario 1: Partidos de Baja Anotación (Actual total goles <= 2.0)
    low_scoring = res_df[res_df["actual_goals"] <= 2]
    low_scoring_acc = (low_scoring["actual_res"] == low_scoring["pred_res_avg"]).mean()
    low_scoring_score = ((low_scoring["actual_home"] == low_scoring["pred_home_avg"]) & 
                         (low_scoring["actual_away"] == low_scoring["pred_away_avg"])).mean()
    print(f"Partidos de Baja Anotación (Actual <= 2 goles) [N={len(low_scoring)}]:")
    print(f"  Acierto de Tendencia (1X2): {low_scoring_acc * 100:.2f}%")
    print(f"  Acierto de Marcador Exacto: {low_scoring_score * 100:.2f}%")
    
    # Escenario 2: Partidos de Alta Anotación (Actual total goles > 3.0)
    high_scoring = res_df[res_df["actual_goals"] > 3]
    high_scoring_acc = (high_scoring["actual_res"] == high_scoring["pred_res_avg"]).mean()
    high_scoring_score = ((high_scoring["actual_home"] == high_scoring["pred_home_avg"]) & 
                          (high_scoring["actual_away"] == high_scoring["pred_away_avg"])).mean()
    print(f"Partidos de Alta Anotación (Actual > 3 goles) [N={len(high_scoring)}]:")
    print(f"  Acierto de Tendencia (1X2): {high_scoring_acc * 100:.2f}%")
    print(f"  Acierto de Marcador Exacto: {high_scoring_score * 100:.2f}%")
    
    # Escenario 3: Subestimación de Goleadas / Alta Puntuación (Over 3.5 goles)
    over_3_5 = res_df[res_df["actual_goals"] >= 4]
    avg_pred_total = (over_3_5["xg_home_avg"] + over_3_5["xg_away_avg"]).mean()
    avg_actual_total = over_3_5["actual_goals"].mean()
    print(f"Partidos con Over 3.5 goles [N={len(over_3_5)}]:")
    print(f"  Goles Promedio Reales: {avg_actual_total:.2f}")
    print(f"  Goles Promedio Proyectados (xG combinado): {avg_pred_total:.2f}")
    print(f"  Subestimación Promedio en Goleadas: {avg_actual_total - avg_pred_total:.2f} goles")
    
    # Escenario 4: Porterías en Cero (Clean Sheets de algún equipo)
    clean_sheets = res_df[(res_df["actual_home"] == 0) | (res_df["actual_away"] == 0)]
    clean_sheets_acc = (clean_sheets["actual_res"] == clean_sheets["pred_res_avg"]).mean()
    print(f"Partidos con alguna Portería en Cero (Clean Sheet) [N={len(clean_sheets)}]:")
    print(f"  Acierto de Tendencia (1X2): {clean_sheets_acc * 100:.2f}%")

if __name__ == "__main__":
    n = 100
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    run_backtest(n)
