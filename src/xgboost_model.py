"""
xgboost_model.py — Modelo XGBoost para predecir goles + Matriz de Poisson.

Entrena dos modelos separados:
- model_home: predice goles del equipo local
- model_away: predice goles del equipo visitante

Los goles predichos se usan como λ para generar la matriz de marcadores
con distribución de Poisson.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import pickle
import os
from src.utils import compute_poisson_matrix
from src.feature_engineering import build_features_for_match

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
XGBOOST_MODEL_FILE = os.path.join(MODEL_DIR, "xgboost_models.pkl")

FEATURE_COLUMNS = [
    "home_avg_goals_scored",
    "home_avg_goals_conceded",
    "home_win_rate",
    "home_goal_diff",
    "home_streak",
    "away_avg_goals_scored",
    "away_avg_goals_conceded",
    "away_win_rate",
    "away_goal_diff",
    "away_streak",
    "h2h_home_win_pct",
    "h2h_away_win_pct",
    "h2h_avg_goals_home",
    "h2h_avg_goals_away",
    "is_neutral",
    "attack_diff",
    "defense_diff",
    # Advanced features
    "home_valor_plantilla_mde",
    "home_promedio_xg_eliminatorias",
    "home_tarjetas_amarillas_por_partido",
    "home_dependencia_estrella",
    "away_valor_plantilla_mde",
    "away_promedio_xg_eliminatorias",
    "away_tarjetas_amarillas_por_partido",
    "away_dependencia_estrella",
    "altitude",
]


class XGBoostGoalModel:
    """
    Modelo XGBoost que predice goles por equipo y genera matrices de Poisson.
    """

    def __init__(self):
        self.model_home = None
        self.model_away = None
        self.is_fitted = False
        self.cv_scores_home = None
        self.cv_scores_away = None

    def fit(self, train_df: pd.DataFrame) -> dict:
        """
        Entrena los dos modelos XGBoost.

        Args:
            train_df: DataFrame del build_training_dataset()
        """
        print(f"Entrenando modelos XGBoost con {len(train_df)} muestras...")

        X = train_df[FEATURE_COLUMNS].values
        y_home = train_df["home_goals"].values
        y_away = train_df["away_goals"].values
        weights = train_df["weight"].values

        # Parámetros optimizados para predicción de goles con aceleración GPU CUDA
        params = {
            "objective": "count:poisson",
            "max_depth": 5,
            "learning_rate": 0.08,
            "n_estimators": 300,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbosity": 0,
            "tree_method": "hist",
            "device": "cuda",
            "n_jobs": 1,
        }

        # Modelo para goles de local
        print("   -> Entrenando modelo de goles local...")
        self.model_home = xgb.XGBRegressor(**params)
        self.model_home.fit(X, y_home, sample_weight=weights)

        # Modelo para goles de visitante
        print("   -> Entrenando modelo de goles visitante...")
        self.model_away = xgb.XGBRegressor(**params)
        self.model_away.fit(X, y_away, sample_weight=weights)

        self.is_fitted = True
        print("Modelos XGBoost entrenados exitosamente")

        # Cross-validation
        print("   -> Evaluando con cross-validation...")
        self.cv_scores_home = cross_val_score(
            xgb.XGBRegressor(**params), X, y_home,
            cv=5, scoring="neg_mean_absolute_error"
        )
        self.cv_scores_away = cross_val_score(
            xgb.XGBRegressor(**params), X, y_away,
            cv=5, scoring="neg_mean_absolute_error"
        )

        results = {
            "mae_home": -self.cv_scores_home.mean(),
            "mae_away": -self.cv_scores_away.mean(),
            "mae_home_std": self.cv_scores_home.std(),
            "mae_away_std": self.cv_scores_away.std(),
        }
        print(f"   MAE goles local: {results['mae_home']:.3f} ± {results['mae_home_std']:.3f}")
        print(f"   MAE goles visitante: {results['mae_away']:.3f} ± {results['mae_away_std']:.3f}")

        return results

    def predict(self, df: pd.DataFrame, home_team: str, away_team: str,
                neutral: bool = True, altitude: float = 0.0,
                home_starting_val: float = None, away_starting_val: float = None,
                home_lineup: list = None, away_lineup: list = None) -> dict:
        """
        Predice goles y genera la matriz de Poisson para un partido.

        Args:
            df: DataFrame completo de partidos (para calcular features)
            home_team: nombre del equipo local
            away_team: nombre del equipo visitante
            neutral: si el partido es en cancha neutral
            altitude: la altitud de la sede en metros
            home_starting_val: valor del once titular local (opcional)
            away_starting_val: valor del once titular visitante (opcional)
            home_lineup: alineación titular local (opcional)
            away_lineup: alineación titular visitante (opcional)
        """
        if not self.is_fitted:
            raise RuntimeError("Modelos no entrenados. Llama a fit() primero.")

        # Construir features
        features = build_features_for_match(
            df, home_team, away_team, neutral, altitude, 
            home_starting_val, away_starting_val,
            home_lineup, away_lineup
        )
        X = np.array([[features[col] for col in FEATURE_COLUMNS]])

        # Predicciones
        lambda_home = float(self.model_home.predict(X)[0])
        lambda_away = float(self.model_away.predict(X)[0])

        # Asegurar valores positivos razonables
        lambda_home = max(0.1, min(lambda_home, 6.0))
        lambda_away = max(0.1, min(lambda_away, 6.0))

        # Matriz de Poisson
        matrix = compute_poisson_matrix(lambda_home, lambda_away)

        return {
            "xg_home": lambda_home,
            "xg_away": lambda_away,
            "matrix": matrix,
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """Devuelve la importancia de features combinada de ambos modelos."""
        if not self.is_fitted:
            return pd.DataFrame()

        imp_home = self.model_home.feature_importances_
        imp_away = self.model_away.feature_importances_

        df = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance_home": imp_home,
            "importance_away": imp_away,
            "importance_avg": (imp_home + imp_away) / 2,
        }).sort_values("importance_avg", ascending=False)

        return df

    def save(self, path: str = None):
        """Guarda los modelos entrenados."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        save_path = path or XGBOOST_MODEL_FILE

        data = {
            "model_home": self.model_home,
            "model_away": self.model_away,
            "is_fitted": self.is_fitted,
            "cv_scores_home": self.cv_scores_home,
            "cv_scores_away": self.cv_scores_away,
        }

        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Modelos XGBoost guardados en {save_path}")

    def load(self, path: str = None):
        """Carga modelos previamente entrenados."""
        load_path = path or XGBOOST_MODEL_FILE
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No se encontraron modelos en {load_path}")

        with open(load_path, "rb") as f:
            data = pickle.load(f)

        self.model_home = data["model_home"]
        self.model_away = data["model_away"]
        
        # Sobrescribir device y n_jobs para evitar caídas por concurrencia CUDA en hilos de Streamlit
        if self.model_home is not None:
            self.model_home.set_params(device="cpu", n_jobs=1)
        if self.model_away is not None:
            self.model_away.set_params(device="cpu", n_jobs=1)

        self.is_fitted = data["is_fitted"]
        self.cv_scores_home = data.get("cv_scores_home")
        self.cv_scores_away = data.get("cv_scores_away")
        print(f"Modelos XGBoost cargados desde {load_path}")
