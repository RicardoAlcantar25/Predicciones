# 🏆 Predictor del Mundial FIFA 2026 — Modelos Matemáticos

Este es un sistema avanzado de predicción para el Mundial de Fútbol 2026, desarrollado en Python y visualizado a través de una interfaz interactiva de Streamlit. El sistema utiliza datos de todos los partidos internacionales jugados desde 1872 para estimar la probabilidad de resultados utilizando dos enfoques matemáticos complementarios:

1. **Modelo Bayesiano Dixon-Coles (MCMC con PyMC):** Estima parámetros de fuerza de ataque y defensa por equipo para predecir los goles esperados (xG) en enfrentamientos directos.
2. **Modelo XGBoost Poisson:** Entrena árboles de decisión optimizados para distribución Poisson, utilizando características dinámicas de rendimiento reciente y Head-to-Head.

---

## 🚀 Requisitos de Instalación

El sistema requiere **Python 3.10 o superior**. Sigue estos pasos para instalarlo y correrlo en tu máquina:

### 1. Preparar el Entorno Virtual

Crea y activa un entorno virtual de Python:

```bash
# Crear entorno virtual
python -m venv .venv

# Activar en Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activar en Windows (CMD)
.venv\Scripts\activate.bat

# Activar en Linux / macOS
source .venv/bin/activate
```

### 2. Instalar Dependencias

Instala todas las librerías necesarias con el gestor de paquetes `pip`:

```bash
pip install -r requirements.txt
```

### 3. Ejecutar la Aplicación

Para lanzar la interfaz interactiva en tu navegador web, ejecuta:

```bash
streamlit run app.py
```

*Nota: La primera vez que abras la aplicación web, verás una advertencia indicando que los modelos no han sido entrenados. Simplemente presiona el botón **🚀 Iniciar Entrenamiento de Modelos** en la pestaña de Inicio. Los modelos se entrenarán, evaluarán y guardarán localmente en la carpeta `models/` para que los siguientes arranques y predicciones sean instantáneos.*

---

## 📊 Estructura del Proyecto

* **`app.py`:** Interfaz web interactiva en Streamlit.
* **`requirements.txt`:** Lista de dependencias del proyecto.
* **`src/`:**
  * **`data_loader.py`:** Descarga los datos históricos y aplica pesos por recencia temporal e importancia de torneo.
  * **`feature_engineering.py`:** Genera estadísticas recientes (goles, rachas) y Head-to-Head para XGBoost.
  * **`bayesian_model.py`:** Definición, entrenamiento MCMC y predicción del modelo Bayesiano jerárquico.
  * **`xgboost_model.py`:** Configuración, entrenamiento y predicción de los regresores XGBoost Poisson.
  * **`simulation.py`:** Motor de simulación del Mundial 2026 (fase de grupos y cuadro eliminatorio).
  * **`utils.py`:** Funciones de ayuda (distribución de Poisson, definición de grupos del Mundial 2026, cálculo de probabilidades).
* **`models/`:** Carpeta autogenerada donde se guardan los modelos entrenados.
* **`data/`:** Carpeta autogenerada donde se almacena el dataset de resultados internacionales.

---

## 🛠️ Detalles Técnicos de los Modelos

### Modelo Bayesiano Dixon-Coles
El modelo define que los goles anotados por el equipo de casa ($H$) y de visita ($A$) siguen distribuciones Poisson independientes:
$$H_i \sim \text{Poisson}(\lambda_{H, i})$$
$$A_i \sim \text{Poisson}(\lambda_{A, i})$$

Donde los ratios logarítmicos se modelan así:
$$\log(\lambda_{H, i}) = \mu + \text{local} \cdot (1 - \text{neutral}) + \text{ataque}_{\text{local}} - \text{defensa}_{\text{visitante}}$$
$$\log(\lambda_{A, i}) = \mu + \text{ataque}_{\text{visitante}} - \text{defensa}_{\text{local}}$$

Los parámetros se estiman mediante MCMC con PyMC, aplicando priors jerárquicas normales centradas en 0.

### Modelo XGBoost Poisson
El modelo utiliza regresores con la función de pérdida `count:poisson`. Recibe un vector de características por partido que incluye:
* Promedio de goles a favor y en contra de cada equipo (ponderado exponencialmente por fecha).
* Eficiencia de victoria reciente (win rate).
* Historial de enfrentamientos directos entre ambos equipos.
* Racha actual de partidos sin perder/ganar.
* Contexto de cancha neutral.
