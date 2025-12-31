from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    from prophet import Prophet
except ImportError:  # pragma: no cover
    Prophet = None

from src.analysis.features import resample_daily

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["ProphetConfig", "forecast_product_prophet"]


@dataclass
class ProphetConfig:
    """Configurable settings for Prophet price forecasting."""
    growth: str = "linear"  # 'linear' or 'logistic'
    changepoint_prior_scale: float = 0.1   # moderado para no forzar tendencia plana ni excesiva
    changepoint_range: float = 0.9
    seasonality_mode: str = "additive"  # reduce efecto multiplicativo
    yearly_seasonality: str | bool = False
    weekly_seasonality: str | bool = True  # permitimos patrón semanal pero con poca fuerza
    daily_seasonality: str | bool = False
    seasonality_prior_scale: float = 2.0   # debilita la repetición periódica
    holidays_prior_scale: float = 10.0
    interval_width: float = 0.8

    # Preprocesado
    impute_method: str = "ffill"  # 'ffill', 'bfill', 'linear'
    outlier_clip: Optional[Tuple[float, float]] = (0.01, 0.99)
    log_transform: bool = True  # model on log1p(price) and invert
    floor: Optional[float] = None  # for logistic growth
    cap: Optional[float] = None    # for logistic growth


def _prepare_prophet_df(
    df: pd.DataFrame,
    config: ProphetConfig,
) -> pd.DataFrame:
    """Preparar dataframe para Prophet (ds/y) con limpieza básica.

    - Resampleo diario (resample_daily)
    - Imputación de huecos
    - Clip de outliers por cuantiles
    - (Opcional) log-transform de precios
    """
    if df.empty:
        return pd.DataFrame(columns=["ds", "y"])

    daily = resample_daily(df)[["timestamp", "price"]].copy()
    daily["timestamp"] = pd.to_datetime(daily["timestamp"])
    daily = daily.sort_values("timestamp")

    # Imputación simple
    if config.impute_method == "ffill":
        daily["price"] = daily["price"].ffill().bfill()
    elif config.impute_method == "bfill":
        daily["price"] = daily["price"].bfill().ffill()
    elif config.impute_method == "linear":
        daily["price"] = daily["price"].interpolate(method="linear", limit_direction="both")
    else:
        raise ValueError(f"Unknown impute_method {config.impute_method}")

    # Clip de outliers
    if config.outlier_clip is not None:
        lo_q, hi_q = config.outlier_clip
        lo, hi = daily["price"].quantile([lo_q, hi_q]).values
        daily["price"] = daily["price"].clip(lower=lo, upper=hi)

    # Avoid non-positive values if using log transform
    if config.log_transform:
        # Aseguramos valores > 0
        eps = 1e-6
        daily["price"] = daily["price"].clip(lower=eps)
        daily["y"] = np.log1p(daily["price"])
    else:
        daily["y"] = daily["price"].astype(float)

    daily = daily.rename(columns={"timestamp": "ds"})
    daily = daily[["ds", "y"]]

    return daily


def _train_prophet_model(df_prophet: pd.DataFrame, config: ProphetConfig):
    """Fit a Prophet model with reasonable defaults for price series."""
    if Prophet is None:
        raise ImportError("prophet is not installed. Install it to use this model.")

    if df_prophet.empty:
        raise ValueError("Cannot train Prophet with empty dataframe.")

    # Build Prophet model
    model = Prophet(
        growth=config.growth,
        changepoint_prior_scale=config.changepoint_prior_scale,
        changepoint_range=config.changepoint_range,
        seasonality_mode=config.seasonality_mode,
        yearly_seasonality=config.yearly_seasonality,
        weekly_seasonality=config.weekly_seasonality,
        daily_seasonality=config.daily_seasonality,
        seasonality_prior_scale=config.seasonality_prior_scale,
        holidays_prior_scale=config.holidays_prior_scale,
        interval_width=config.interval_width,
    )

    # Logistic growth requires 'floor' and 'cap'
    df_train = df_prophet.copy()
    if config.growth == "logistic":
        floor = config.floor if config.floor is not None else df_train["y"].min() * 0.9
        cap = config.cap if config.cap is not None else df_train["y"].max() * 1.1
        df_train["floor"] = floor
        df_train["cap"] = cap

    logger.info("Fitting Prophet model with %d observations", len(df_train))
    model.fit(df_train)

    return model


def _forecast_prophet(model, horizon: int, config: ProphetConfig, last_ds: pd.Timestamp) -> pd.DataFrame:
    """Forecast future values with a trained Prophet model."""
    # include_history=True para que Prophet construya internamente correcto
    future = model.make_future_dataframe(periods=horizon, freq="D", include_history=True)

    if config.growth == "logistic":
        # Necesitamos poner floor/cap también en el futuro
        # -> usamos los valores usados en train
        if "floor" in model.history.columns and "cap" in model.history.columns:
            floor = model.history["floor"].iloc[0]
            cap = model.history["cap"].iloc[0]
        else:
            floor = config.floor if config.floor is not None else 0.0
            cap = config.cap if config.cap is not None else 1.0
        future["floor"] = floor
        future["cap"] = cap

    forecast = model.predict(future)

    # Nos quedamos solo con los últimos `horizon` días futuros
    forecast = forecast[forecast["ds"] > last_ds].head(horizon)

    return forecast


def forecast_product_prophet(
    df: pd.DataFrame,
    horizon: int,
    config: Optional[ProphetConfig] = None,
) -> pd.DataFrame:
    """Wrapper para producir un forecast estándar con Prophet.

    Devuelve columnas:
    - timestamp
    - price (None para futuro, para compatibilidad con otras funciones)
    - forecast
    - lower
    - upper
    """
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "price", "forecast", "lower", "upper"])

    config = config or ProphetConfig()

    # Preparar datos para Prophet
    df_prophet = _prepare_prophet_df(df, config)
    if df_prophet.empty:
        logger.warning("Prepared Prophet dataframe is empty; returning empty forecast.")
        return pd.DataFrame(columns=["timestamp", "price", "forecast", "lower", "upper"])

    last_ds = df_prophet["ds"].max()

    # Entrenar modelo
    model = _train_prophet_model(df_prophet, config)

    # Forecast
    forecast_raw = _forecast_prophet(model, horizon, config, last_ds)

    # Extraer columnas básicas
    merged = forecast_raw[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
        columns={"ds": "timestamp", "yhat": "forecast", "yhat_lower": "lower", "yhat_upper": "upper"}
    )

    # Si usamos log_transform, volvemos a escala de precio
    if config.log_transform:
        merged["forecast"] = np.expm1(merged["forecast"])
        merged["lower"] = np.expm1(merged["lower"])
        merged["upper"] = np.expm1(merged["upper"])

    merged["price"] = None
    merged = merged[["timestamp", "price", "forecast", "lower", "upper"]]

    return merged
