#preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def separar_tipos(df: pd.DataFrame, lista_features: list) -> tuple:

    """
    Esta función separa una lista de features en categóricas y numéricas según el tipo de dato en el dataset.
    Args:
        df (pd.DataFrame): el dataset.
        lista_features (list): lista de nombres de features.
    Returns:
        categoricas (list): nombres de features categóricas.
        numericas (list): nombres de features numéricas.
    """

    #valores iniciales
    categoricas = []
    numericas = []

    for col in lista_features:

        #es realmente numérica
        if pd.api.types.is_numeric_dtype(df[col]):
            numericas.append(col)
        else:
            categoricas.append(col)

    return categoricas, numericas


def construir_preprocesador(numericas: list, categoricas: list) -> ColumnTransformer:

    """
    Esta función construye un ColumnTransformer con StandardScaler para numéricas y OneHotEncoder para categóricas.
    Args:
        numericas (list): nombres de features numéricas.
        categoricas (list): nombres de features categóricas.
    Returns:
        preprocesador (ColumnTransformer): el preprocesador construido.
    """

    #preprocesador
    preprocesador = ColumnTransformer(
        transformers=[

            #var numéricas -> estandarizar
            ("num", StandardScaler(), numericas),

            #var categóricas -> codificar
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas)
        ]
    )

    return preprocesador


def obtener_nombres_features(preprocesador, numericas: list, categoricas: list) -> list:

    """
    Esta función devuelve la lista de nombres de features tras el preprocesado (numéricas originales + nombres generados por OneHotEncoder).
    Args:
        preprocesador: el ColumnTransformer ya ajustado.
        numericas (list): nombres originales de features numéricas.
        categoricas (list): nombres originales de features categóricas.
    Returns:
        nombres (list): todos los nombres de features tras el preprocesado.
    """

    nombres_num = list(numericas)
    nombres_cat = preprocesador.transformers_[1][1].get_feature_names_out(categoricas).tolist()

    #nombres totales
    nombres = nombres_num + nombres_cat

    return nombres
