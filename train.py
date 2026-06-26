"""
train.py — Script para entrenar y guardar los modelos predictivos del Mundial 2026.
"""

import os
import time
from src.data_loader import load_and_preprocess, prepare_bayesian_data
from src.feature_engineering import build_training_dataset
from src.xgboost_model import XGBoostGoalModel
from src.bayesian_model import BayesianGoalModel

def main():
    start_time = time.time()
    print("=== INICIANDO ENTRENAMIENTO DE MODELOS DEL MUNDIAL 2026 ===")
    
    # 1. Cargar datos
    print("\n[1/4] Cargando y preprocesando datos históricos...")
    df = load_and_preprocess(min_year=2018)
    
    # 2. Entrenar XGBoost
    print("\n[2/4] Preparando features y entrenando XGBoost Poisson...")
    train_df = build_training_dataset(df)
    xgb_model = XGBoostGoalModel()
    xgb_metrics = xgb_model.fit(train_df)
    xgb_model.save()
    
    # 3. Entrenar Modelo Bayesiano
    print("\n[3/4] Preparando datos y estimando Modelo Bayesiano...")
    print("Nota: Estimando mediante Maximo A Posteriori (MAP) para ejecucion instantanea en Windows.")
    bayes_data = prepare_bayesian_data(df)
    bayes_model = BayesianGoalModel()
    # Usamos estimacion por MAP para que sea instantaneo y preciso en cualquier Windows
    bayes_model.fit(bayes_data, method="map")
    bayes_model.save()
    
    # 4. Finalizado
    elapsed = time.time() - start_time
    print(f"\n[4/4] ¡Entrenamiento completado con éxito en {elapsed/60:.2f} minutos!")
    print("=== MODELOS GUARDADOS EN LA CARPETA 'models/' ===")

if __name__ == "__main__":
    main()
