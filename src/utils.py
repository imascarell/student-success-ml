#utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def grafico_importancia(nombres_features: list, importancias, titulo: str, top_n: int = 15) -> pd.DataFrame:

    """
    Esta función genera un gráfico de barras horizontales con las variables más importantes.
    Args:
        nombres_features (list): lista de nombres de features.
        importancias: array de valores de importancia.
        titulo (str): título del gráfico.
        top_n (int): número de features a mostrar.
    Returns:
        imp_df (pd.DataFrame): dataframe con todas las features ordenadas por importancia.
    """

    imp_df = pd.DataFrame({
        "variable": nombres_features,
        "importancia": importancias
    })

    imp_df = imp_df.sort_values("importancia", ascending=False)

    top = imp_df.head(top_n)

    plt.barh(top["variable"], top["importancia"])
    plt.xlabel("Importancia")
    plt.title(titulo)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    return imp_df


def extraer_importancias(modelo) -> tuple:

    """
    Esta función extrae las importancias de un modelo ya entrenado.
    Funciona con modelos de árboles (feature_importances_) y lineales (coef_).
    Args:
        modelo: el modelo entrenado.
    Returns:
        importancias: array de valores de importancia (o None si no disponible).
        tipo_importancia (str): descripción del tipo de importancia.
    """

    if hasattr(modelo, "feature_importances_"):
        importancias = modelo.feature_importances_
        tipo_importancia = "Importancia (impureza)"

    elif hasattr(modelo, "coef_"):
        #para modelos lineales: valor absoluto del coeficiente
        #si es multiclase, promedio entre clases
        if len(modelo.coef_.shape) > 1:
            importancias = np.mean(np.abs(modelo.coef_), axis=0)
        else:
            importancias = np.abs(modelo.coef_)
        tipo_importancia = "Importancia (|coef|)"

    else:
        importancias = None
        tipo_importancia = "No disponible"

    return importancias, tipo_importancia
