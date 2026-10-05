#unsupervised.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def obtener_vars_no_supervisado() -> tuple:

    """
    Esta función devuelve las listas de variables numéricas y categóricas
    seleccionadas para la tarea no supervisada.
    Returns:
        numericas (list): nombres de variables numéricas.
        categoricas (list): nombres de variables categóricas.
    """

    #variables numéricas
    numericas = [
        "edad_al_matricularse",
        "nota_admision",
        "nota_cualificacion_previa",
        "orden_solicitud",
        "asignaturas_1sem_matriculadas",
        "asignaturas_1sem_evaluadas",
        "asignaturas_1sem_aprobadas",
        "nota_media_1sem",
        "asignaturas_1sem_sin_evaluacion",
        "asignaturas_1sem_convalidadas",
        "tasa_desempleo",
        "tasa_inflacion",
        "pib"
    ]

    #variables categóricas
    categoricas = [
        "genero",
        "desplazado",
        "internacional",
        "deudor",
        "matricula_al_dia",
        "becado"
    ]

    return numericas, categoricas


def evaluar_rango_k(X_escalado, k_min: int, k_max: int) -> tuple:

    """
    Esta función evalúa K-Means para un rango de valores de k.
    Args:
        X_escalado: la matriz de datos escalados.
        k_min (int): número mínimo de clusters.
        k_max (int): número máximo de clusters.
    Returns:
        inercias (list): valores de inercia para cada k.
        siluetas (list): coeficientes de silueta para cada k.
    """

    #valores iniciales
    inercias = []
    siluetas = []

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X_escalado)
        inercias.append(km.inertia_)
        siluetas.append(silhouette_score(X_escalado, km.labels_))

    return inercias, siluetas


def analisis_estabilidad(X_escalado, k: int, n_repeticiones: int = 10) -> list:

    """
    Esta función repite K-Means en submuestras del 80% para evaluar estabilidad.
    Args:
        X_escalado: la matriz de datos escalados.
        k (int): número de clusters.
        n_repeticiones (int): número de repeticiones.
    Returns:
        siluetas (list): coeficientes de silueta obtenidos en cada submuestra.
    """

    #valor inicial
    siluetas = []

    for semilla in range(n_repeticiones):
        np.random.seed(semilla)
        n_muestra = int(0.8 * len(X_escalado))
        indices = np.random.choice(len(X_escalado), size=n_muestra, replace=False)
        X_sub = X_escalado[indices]

        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        etiquetas = km.fit_predict(X_sub)
        sil = silhouette_score(X_sub, etiquetas)
        siluetas.append(sil)

    return siluetas
