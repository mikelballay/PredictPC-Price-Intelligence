# src/models/gbm_model.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.analysis.features import create_features


@dataclass
class GBMConfig:
    """
    Configuración para el modelo de Gradient Boosting sobre features de la serie.

    - log_transform: trabajar en escala log(precio+1) para estabilizar la varianza.
    - outlier_clip: recortar precios extremos por cuantiles (para no sobre-ajustar).
    - n_estimators, learning_rate, etc.: hiperparámetros típicos de GBM.
    """
    log_transform: bool = True
    outlier_clip: Optional[Tuple[float, float]] = (0.01, 0.99)

    n_estimators: int = 400
    learning_rate: float = 0.03
    max_depth: int = 3
    min_samples_leaf: int = 3
    random_state: int = 42

    # Nivel de confianza aproximado para los intervalos (z≈1.64 ≈ 90%)
    interval_alpha: float = 0.10


_FEATURE_COLS: Sequence[str] = (
    "lag_1",
    "lag_7",
    "roll_mean_7",
    "roll_std_7",
    "roll_mean_30",
    "roll_std_30",
)


def _prepare_ml_dataframe(
    df: pd.DataFrame,
    config: GBMConfig,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    A partir de un DataFrame con columnas ['timestamp', 'product_id', 'price'],
    genera las features de ML y devuelve (df_feat, X, y).
    """
    if df.empty:
        raise ValueError("Empty dataframe passed to _prepare_ml_dataframe")

    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp")
    data = data[data["price"] > 0]

    if data.empty:
        raise ValueError("No positive prices available for ML model.")

    # Recorte de outliers ligero
    if config.outlier_clip is not None:
        lo_q, hi_q = config.outlier_clip
        lo, hi = data["price"].quantile([lo_q, hi_q]).values
        data["price"] = data["price"].clip(lower=lo, upper=hi)

    # Crear features diarias
    feat = create_features(data[["timestamp", "product_id", "price"]])
    feat = feat.dropna(subset=_FEATURE_COLS)

    if feat.empty:
        raise ValueError("Not enough data after feature construction for GBM model.")

    X = feat.loc[:, _FEATURE_COLS].to_numpy(dtype=float)

    if config.log_transform:
        y = np.log1p(feat["price"].astype(float).to_numpy())
    else:
        y = feat["price"].astype(float).to_numpy()

    return feat, X, y


def train_gbm(
    df: pd.DataFrame,
    config: Optional[GBMConfig] = None,
) -> Dict:
    """
    Entrena un GradientBoostingRegressor sobre la serie de precios de un producto.

    Devuelve un diccionario con:
      - model: el modelo de sklearn entrenado
      - config: la configuración usada
      - sigma: desviación típica de los residuos (para intervalos)
    """
    cfg = config or GBMConfig()
    feat, X, y = _prepare_ml_dataframe(df, cfg)

    model = GradientBoostingRegressor(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        random_state=cfg.random_state,
    )
    model.fit(X, y)

    y_pred = model.predict(X)
    resid = y - y_pred
    sigma = float(np.std(resid))

    return {
        "model": model,
        "config": cfg,
        "sigma": sigma,
    }


def forecast_product_gbm(
    df: pd.DataFrame,
    horizon: int,
    config: Optional[GBMConfig] = None,
) -> pd.DataFrame:
    """
    Pronostica 'horizon' días vista a partir de df (serie de un producto).

    Devuelve un DataFrame con columnas:
      - timestamp (fechas futuras, diarias)
      - price (None)
      - forecast
      - lower
      - upper
    Compatible con el resto del dashboard.
    """
    if df.empty or horizon <= 0:
        return pd.DataFrame(columns=["timestamp", "price", "forecast", "lower", "upper"])

    cfg = config or GBMConfig()

    # Entrenar modelo en los datos históricos
    model_info = train_gbm(df, cfg)
    model = model_info["model"]
    sigma = model_info["sigma"]

    # Historial que iremos extendiendo con nuestras predicciones
    hist = df[["timestamp", "product_id", "price"]].copy()
    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    hist = hist.sort_values("timestamp")
    product_id = hist["product_id"].iloc[0] if "product_id" in hist.columns else "unknown"

    records = []
    last_date = hist["timestamp"].max().normalize().date()

    # z aproximado para un 90% de confianza
    z = 1.64

    for _ in range(horizon):
        next_date = last_date + pd.Timedelta(days=1)
        ts_next = pd.Timestamp(next_date)

        # Creamos una copia con un punto futuro vacío...
        tmp = pd.concat(
            [
                hist,
                pd.DataFrame(
                    {
                        "timestamp": [ts_next],
                        "product_id": [product_id],
                        "price": [np.nan],
                    }
                ),
            ],
            ignore_index=True,
        )
        tmp["timestamp"] = pd.to_datetime(tmp["timestamp"])

        # ...y volvemos a construir las features para obtener las del día siguiente
        tmp_feat = create_features(tmp[["timestamp", "product_id", "price"]])
        mask = tmp_feat["timestamp"].dt.date == next_date
        row = tmp_feat.loc[mask]

        if row.empty:
            break

        X_next = row.loc[:, _FEATURE_COLS].to_numpy(dtype=float)
        y_fc = float(model.predict(X_next)[0])

        # Intervalos en el espacio del modelo
        lower_y = y_fc - z * sigma
        upper_y = y_fc + z * sigma

        if cfg.log_transform:
            fc = np.expm1(y_fc)
            lower = max(0.0, np.expm1(lower_y))
            upper = max(0.0, np.expm1(upper_y))
        else:
            fc = y_fc
            lower = lower_y
            upper = upper_y

        records.append(
            {
                "timestamp": ts_next,
                "price": None,
                "forecast": float(fc),
                "lower": float(lower),
                "upper": float(upper),
            }
        )

        # Actualizamos el histórico con la predicción para poder seguir iterando
        hist = pd.concat(
            [
                hist,
                pd.DataFrame(
                    {
                        "timestamp": [ts_next],
                        "product_id": [product_id],
                        "price": [fc],
                    }
                ),
            ],
            ignore_index=True,
        )
        last_date = next_date

    if not records:
        return pd.DataFrame(columns=["timestamp", "price", "forecast", "lower", "upper"])

    return pd.DataFrame(records)
