#data_loading.py

import pandas as pd


def cargar_dataset(ruta: str) -> pd.DataFrame:

    """
    Esta función carga el dataset desde la ruta indicada.
    Args:
        ruta (str): la ruta al archivo CSV.
    Returns:
        df (pd.DataFrame): el dataset cargado.
    """

    df = pd.read_csv(ruta, sep=";")

    return df


def obtener_bloques() -> dict:

    """
    Esta función devuelve un diccionario con los bloques de variables del proyecto.
    Args:
        None.
    Returns:
        bloques (dict): los bloques de variables.
    """

    bloques = {

        #variables demográficas
        "demograficas": [
            "estado_civil",
            "nacionalidad",
            "desplazado",
            "necesidades_educativas_especiales",
            "genero",
            "edad_al_matricularse",
            "internacional"
        ],

        #variables de solicitud
        "acceso_previo": [
            "modo_solicitud",
            "orden_solicitud",
            "curso",
            "asistencia_diurna_vespertina",
            "cualificacion_previa",
            "nota_cualificacion_previa",
            "cualificacion_madre",
            "cualificacion_padre",
            "ocupacion_madre",
            "ocupacion_padre",
            "nota_admision"
        ],

        #variables socioadministrativas
        "socio_admin": [
            "deudor",
            "matricula_al_dia",
            "becado"
        ],

        #variables macroeconómicas
        "macroeconomicas": [
            "tasa_desempleo",
            "tasa_inflacion",
            "pib"
        ],

        #variables rendimiento primer semestre
        "rendimiento_1sem": [
            "asignaturas_1sem_convalidadas",
            "asignaturas_1sem_matriculadas",
            "asignaturas_1sem_evaluadas",
            "asignaturas_1sem_aprobadas",
            "nota_media_1sem",
            "asignaturas_1sem_sin_evaluacion"
        ],

        #variables rendimiento segundo semestre
        "rendimiento_2sem": [
            "asignaturas_2sem_convalidadas",
            "asignaturas_2sem_matriculadas",
            "asignaturas_2sem_evaluadas",
            "asignaturas_2sem_aprobadas",
            "asignaturas_2sem_sin_evaluacion"
        ]
    }

    return bloques


def obtener_features_clasificacion() -> tuple:

    """
    Esta función devuelve las listas de features para clasificación (base y extendida).
    Args:
        None.
    Returns:
        features_base (list): features sin primer semestre.
        features_extendida (list): features con primer semestre.
    """

    #obtener bloques variables
    bloques = obtener_bloques()

    #features de información inicial
    features_base = (
        bloques["demograficas"]
        + bloques["acceso_previo"]
        + bloques["socio_admin"]
        + bloques["macroeconomicas"]
    )

    #features info inicial y reciente
    features_extendida = features_base + bloques["rendimiento_1sem"]

    return features_base, features_extendida


def obtener_features_regresion() -> tuple:

    """
    Esta función devuelve las listas de features para regresión (sin y con 1er semestre).
    También devuelve la lista de variables excluidas por leakage.
    Args:
        None.
    Returns:
        features_sin_1sem (list): features sin primer semestre.
        features_con_1sem (list): features con primer semestre.
        excluidas_leakage (list): variables excluidas por leakage.
    """

    #obtener bloques variables
    bloques = obtener_bloques()

    #features de información inicial
    features_sin_1sem = (
        bloques["demograficas"]
        + bloques["acceso_previo"]
        + bloques["socio_admin"]
        + bloques["macroeconomicas"]
    )

    #features info inicial y reciente
    features_con_1sem = features_sin_1sem + bloques["rendimiento_1sem"]

    #features del segundo semestre (excluir por leakage)
    excluidas_leakage = bloques["rendimiento_2sem"]

    #la variable objetivo de clasificación también se excluye como predictor
    excluida_objetivo = "objetivo"

    return features_sin_1sem, features_con_1sem, excluidas_leakage
