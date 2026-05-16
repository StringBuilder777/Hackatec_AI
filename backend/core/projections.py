"""Funciones puras de ML — sin llamadas a API externa.

Candidatas a migrar a Rust en Fase 3 (cómputo puro).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression


def linear_forecast(valores: list[float], periodos: int = 6) -> dict[str, Any]:
    """Proyección lineal de una serie numérica.

    Devuelve histórico, proyección, pendiente, tendencia, R² y un chart_config
    listo para Chart.js (línea con dos datasets: histórico azul, forecast verde).
    """
    if not valores or len(valores) < 2:
        return {
            "historico": list(valores or []),
            "proyeccion": [],
            "pendiente": 0.0,
            "tendencia": "sin_datos",
            "confianza_r2": 0.0,
            "chart_config": _empty_chart(),
        }

    arr = np.asarray(valores, dtype=float)
    n = len(arr)
    X = np.arange(n).reshape(-1, 1)
    y = arr

    model = LinearRegression()
    model.fit(X, y)
    r2 = float(model.score(X, y))
    pendiente = float(model.coef_[0])

    fut_X = np.arange(n, n + periodos).reshape(-1, 1)
    proyeccion = model.predict(fut_X).tolist()

    tendencia = "creciente" if pendiente > 0 else "decreciente" if pendiente < 0 else "plana"

    # Chart.js: dos datasets — histórico relleno + forecast continuación
    labels = [f"t{i}" for i in range(n + periodos)]
    historico_padded = list(arr) + [None] * periodos
    forecast_padded = [None] * (n - 1) + [float(arr[-1])] + list(proyeccion)

    chart_config = {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [
                {
                    "label": "Histórico",
                    "data": historico_padded,
                    "borderColor": "#3b82f6",
                    "backgroundColor": "rgba(59,130,246,0.15)",
                    "tension": 0.3,
                    "spanGaps": False,
                },
                {
                    "label": "Proyección",
                    "data": forecast_padded,
                    "borderColor": "#10b981",
                    "backgroundColor": "rgba(16,185,129,0.10)",
                    "borderDash": [6, 4],
                    "tension": 0.3,
                    "spanGaps": False,
                },
            ],
        },
    }

    return {
        "historico": list(arr),
        "proyeccion": proyeccion,
        "pendiente": pendiente,
        "tendencia": tendencia,
        "confianza_r2": r2,
        "chart_config": chart_config,
    }


def detect_anomalies(
    valores: list[float],
    umbral_sigma: float = 2.0,
) -> dict[str, Any]:
    """Detección de outliers usando Z-score.

    Devuelve media, std, lista de anomalías (con índice y z-score) y total.
    """
    if not valores:
        return {
            "media": 0.0,
            "std": 0.0,
            "anomalias": [],
            "total": 0,
            "umbral_sigma": umbral_sigma,
        }

    arr = np.asarray(valores, dtype=float)
    media = float(np.mean(arr))
    std = float(np.std(arr))

    anomalias: list[dict] = []
    if std > 0:
        z_scores = (arr - media) / std
        for i, z in enumerate(z_scores):
            if abs(z) > umbral_sigma:
                anomalias.append({
                    "indice": int(i),
                    "valor": float(arr[i]),
                    "z_score": float(z),
                })

    return {
        "media": media,
        "std": std,
        "anomalias": anomalias,
        "total": len(anomalias),
        "umbral_sigma": umbral_sigma,
    }


def _empty_chart() -> dict[str, Any]:
    return {
        "type": "line",
        "data": {"labels": [], "datasets": []},
    }
