#regression.py

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def construir_pipelines_regresion(preprocesador) -> dict:

    """
    Esta función construye los pipelines de regresión: ridge, elastic net, random forest y gradient boosting.
    Args:
        preprocesador: el ColumnTransformer preprocesador.
    Returns:
        pipelines (dict): diccionario con nombre del modelo y pipeline.
    """

    pipelines = {

        #regularización de ridge
        "Ridge": Pipeline([
            ("preprocesador", preprocesador),
            ("modelo", Ridge(alpha=1.0))
        ]),

        #regularización de EN
        "Elastic Net": Pipeline([
            ("preprocesador", preprocesador),
            ("modelo", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=42))
        ]),

        #métodos basados en árboles de RF
        "Random Forest": Pipeline([
            ("preprocesador", preprocesador),
            ("modelo", RandomForestRegressor(n_estimators=200, random_state=42))
        ]),

        #métodos basados en árboles de GB
        "Gradient Boosting": Pipeline([
            ("preprocesador", preprocesador),
            ("modelo", GradientBoostingRegressor(n_estimators=200, random_state=42))
        ])
    }

    return pipelines


def construir_baseline_regresion() -> DummyRegressor:

    """
    Esta función construye el baseline que predice la media.
    Args:
        None.
    Returns:
        baseline (DummyRegressor): el regresor baseline.
    """

    baseline = DummyRegressor(strategy="mean")

    return baseline


def evaluar_regresion(y_real, y_pred) -> dict:

    """
    Esta función evalúa las predicciones de regresión.
    Args:
        y_real: los valores reales.
        y_pred: los valores predichos.
    Returns:
        metricas (dict): diccionario con RMSE, MAE y R2.
    """

    metricas = {
        "rmse": np.sqrt(mean_squared_error(y_real, y_pred)),
        "mae": mean_absolute_error(y_real, y_pred),
        "r2": r2_score(y_real, y_pred)
    }

    return metricas
