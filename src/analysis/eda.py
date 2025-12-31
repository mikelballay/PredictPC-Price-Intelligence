from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


def plot_price_time_series(df: pd.DataFrame, product_id: str):
    """Return a plotly line chart for a single product."""
    subset = df[df["product_id"] == product_id]
    fig = px.line(subset, x="timestamp", y="price", title=f"Price over time for {product_id}")
    fig.update_layout(xaxis_title="Date", yaxis_title="Price")
    return fig


def compute_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily percentage returns."""
    out = df.copy()
    out["return"] = out.groupby("product_id")["price"].pct_change()
    return out


def compute_moving_averages(df: pd.DataFrame, windows: Iterable[int] = (7, 30)) -> pd.DataFrame:
    """Compute moving averages for given windows."""
    out = df.copy()
    for w in windows:
        out[f"ma_{w}"] = out.groupby("product_id")["price"].transform(lambda s: s.rolling(window=w, min_periods=1).mean())
    return out


def summarize_price_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return statistics per product."""
    stats = (
        df.groupby("product_id")["price"]
        .agg(["min", "max", "mean", "std", "count"])
        .rename(columns={"count": "n_obs"})
        .reset_index()
    )
    return stats


def detect_outliers(df: pd.DataFrame, zscore_threshold: float = 4.0) -> pd.DataFrame:
    """
    Detect basic outliers: negative prices or values far from median.

    Returns rows flagged as outliers.
    """
    outliers: List[pd.Series] = []
    for pid, group in df.groupby("product_id"):
        prices = group["price"]
        median = prices.median()
        mad = (prices - median).abs().median() or 1e-9
        zscores = (prices - median).abs() / mad
        flags = (prices < 0) | (zscores > zscore_threshold)
        outliers.append(group[flags])
    return pd.concat(outliers) if outliers else pd.DataFrame(columns=df.columns)
