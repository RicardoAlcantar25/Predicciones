"""
bayesian_model.py — Modelo Bayesiano jerárquico Dixon-Coles con MCMC.

Estima parámetros de ataque y defensa por equipo usando PyMC + NUTS sampler.
Los goles esperados (xG) se calculan a partir de los parámetros posteriores.
"""

import numpy as np
import pymc as pm
import arviz as az
import pickle
import os
from src.utils import compute_poisson_matrix

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
BAYESIAN_MODEL_FILE = os.path.join(MODEL_DIR, "bayesian_trace.pkl")
BAYESIAN_PARAMS_FILE = os.path.join(MODEL_DIR, "bayesian_params.pkl")


class BayesianGoalModel:
    """
    Modelo Bayesiano jerárquico para predecir goles en partidos de fútbol.

    Modelo:
        Goles_home ~ Poisson(λ_home)
        Goles_away ~ Poisson(λ_away)

        log(λ_home) = intercept + home_adv*(1-neutral) + attack[home] - defense[away]
        log(λ_away) = intercept + attack[away] - defense[home]

        attack[i] ~ Normal(0, σ_att)
        defense[i] ~ Normal(0, σ_def)
        home_adv ~ Normal(0.3, 0.2)
        intercept ~ Normal(0.3, 0.5)
    """

    def __init__(self):
        self.trace = None
        self.team_index = None
        self.index_to_team = None
        self.attack_means = None
        self.defense_means = None
        self.home_advantage = None
        self.intercept = None
        self.is_fitted = False

    def fit(self, data: dict, samples: int = 1500, tune: int = 1000,
            chains: int = 2, target_accept: float = 0.9, method: str = "mcmc") -> az.InferenceData:
        """
        Entrena el modelo bayesiano con MCMC o MAP.

        Args:
            data: dict del prepare_bayesian_data()
            samples: número de muestras por cadena
            tune: número de pasos de tuning
            chains: número de cadenas MCMC
            target_accept: tasa de aceptación objetivo para NUTS
            method: "mcmc" (lento) o "map" (instantáneo)
        """
        self.team_index = data["team_index"]
        self.index_to_team = data["index_to_team"]
        n_teams = data["n_teams"]

        print(f"Entrenando modelo Bayesiano con {n_teams} equipos, "
              f"{data['n_matches']} partidos ({method.upper()})...")

        with pm.Model() as model:
            # Hiperparámetros
            sigma_att = pm.HalfNormal("sigma_att", sigma=1.0)
            sigma_def = pm.HalfNormal("sigma_def", sigma=1.0)

            # Intercepto (tasa base de goles)
            intercept = pm.Normal("intercept", mu=0.3, sigma=0.5)

            # Ventaja de local
            home_adv = pm.Normal("home_adv", mu=0.25, sigma=0.2)

            # Parámetros de ataque y defensa por equipo
            attack = pm.Normal("attack", mu=0, sigma=sigma_att, shape=n_teams)
            defense = pm.Normal("defense", mu=0, sigma=sigma_def, shape=n_teams)

            # Lambda para goles de local y visitante
            home_neutral_factor = home_adv * (1 - data["is_neutral"])

            log_lambda_home = (
                intercept
                + home_neutral_factor
                + attack[data["home_team_idx"]]
                - defense[data["away_team_idx"]]
            )
            log_lambda_away = (
                intercept
                + attack[data["away_team_idx"]]
                - defense[data["home_team_idx"]]
            )

            lambda_home = pm.math.exp(log_lambda_home)
            lambda_away = pm.math.exp(log_lambda_away)

            # Likelihood
            home_goals = pm.Poisson(
                "home_goals",
                mu=lambda_home,
                observed=data["home_goals"]
            )
            away_goals = pm.Poisson(
                "away_goals",
                mu=lambda_away,
                observed=data["away_goals"]
            )

            if method == "map":
                print("   -> Optimizando estimacion por Maximo A Posteriori (MAP)...")
                map_estimate = pm.find_MAP()
                
                self.attack_means = map_estimate["attack"]
                self.defense_means = map_estimate["defense"]
                self.home_advantage = float(map_estimate["home_adv"])
                self.intercept = float(map_estimate["intercept"])
                self.trace = None
                self.is_fitted = True
                print("Modelo Bayesiano (MAP) optimizado exitosamente")
                return None
            else:
                print(f"   -> {chains} cadenas x {samples} muestras + {tune} tuning")
                self.trace = pm.sample(
                    draws=samples,
                    tune=tune,
                    chains=chains,
                    target_accept=target_accept,
                    return_inferencedata=True,
                    progressbar=True,
                    cores=1,  # Más estable en Windows
                )

        # Extraer parámetros medios posteriores
        self._extract_parameters()
        self.is_fitted = True

        print("Modelo Bayesiano entrenado exitosamente con MCMC")
        return self.trace

    def _extract_parameters(self):
        """Extrae los parámetros medios del posterior."""
        posterior = self.trace.posterior

        self.attack_means = posterior["attack"].mean(dim=["chain", "draw"]).values
        self.defense_means = posterior["defense"].mean(dim=["chain", "draw"]).values
        self.home_advantage = float(posterior["home_adv"].mean())
        self.intercept = float(posterior["intercept"].mean())

    def predict(self, home_team: str, away_team: str,
                neutral: bool = True) -> dict:
        """
        Predice goles esperados para un partido.

        Returns:
            dict con lambda_home, lambda_away, y la matriz de Poisson.
        """
        if not self.is_fitted:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a fit() primero.")

        home_idx = self.team_index.get(home_team)
        away_idx = self.team_index.get(away_team)

        if home_idx is None:
            raise ValueError(f"Equipo '{home_team}' no encontrado en los datos.")
        if away_idx is None:
            raise ValueError(f"Equipo '{away_team}' no encontrado en los datos.")

        # Calcular lambda
        home_neutral = self.home_advantage * (0 if neutral else 1)

        log_lambda_home = (
            self.intercept + home_neutral
            + self.attack_means[home_idx]
            - self.defense_means[away_idx]
        )
        log_lambda_away = (
            self.intercept
            + self.attack_means[away_idx]
            - self.defense_means[home_idx]
        )

        lambda_home = np.exp(log_lambda_home)
        lambda_away = np.exp(log_lambda_away)

        # Matriz de marcadores
        matrix = compute_poisson_matrix(lambda_home, lambda_away)

        return {
            "xg_home": lambda_home,
            "xg_away": lambda_away,
            "matrix": matrix,
        }

    def get_team_rankings(self) -> list:
        """
        Devuelve el ranking de equipos por fuerza combinada (ataque - defensa).
        Mayor = mejor equipo.
        """
        if not self.is_fitted:
            return []

        rankings = []
        for team, idx in self.team_index.items():
            rankings.append({
                "team": team,
                "attack": float(self.attack_means[idx]),
                "defense": float(self.defense_means[idx]),
                "overall": float(self.attack_means[idx] - self.defense_means[idx]),
            })

        rankings.sort(key=lambda x: x["overall"], reverse=True)
        return rankings

    def get_diagnostics(self) -> dict:
        """Devuelve diagnósticos del MCMC."""
        if self.trace is None:
            return {}

        summary = az.summary(self.trace, var_names=["intercept", "home_adv", "sigma_att", "sigma_def"])
        rhat_ok = (summary["r_hat"] < 1.05).all()

        return {
            "summary": summary,
            "rhat_ok": rhat_ok,
        }

    def save(self, path: str = None):
        """Guarda el modelo entrenado."""
        os.makedirs(MODEL_DIR, exist_ok=True)

        params = {
            "attack_means": self.attack_means,
            "defense_means": self.defense_means,
            "home_advantage": self.home_advantage,
            "intercept": self.intercept,
            "team_index": self.team_index,
            "index_to_team": self.index_to_team,
            "is_fitted": self.is_fitted,
        }

        save_path = path or BAYESIAN_PARAMS_FILE
        with open(save_path, "wb") as f:
            pickle.dump(params, f)
        print(f"Modelo Bayesiano guardado en {save_path}")

    def load(self, path: str = None):
        """Carga un modelo previamente entrenado."""
        load_path = path or BAYESIAN_PARAMS_FILE
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No se encontró el modelo en {load_path}")

        with open(load_path, "rb") as f:
            params = pickle.load(f)

        self.attack_means = params["attack_means"]
        self.defense_means = params["defense_means"]
        self.home_advantage = params["home_advantage"]
        self.intercept = params["intercept"]
        self.team_index = params["team_index"]
        self.index_to_team = params["index_to_team"]
        self.is_fitted = params["is_fitted"]
        print(f"Modelo Bayesiano cargado desde {load_path}")
