from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_time_series(df: pd.DataFrame, x: str, y: str, title: str = "", hue: Optional[str] = None) -> plt.Figure:
    """Generic time series plot with optional hue for multiple products."""
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(y)
    fig.autofmt_xdate()
    return fig


def plot_price_distribution(df: pd.DataFrame, column: str = "price", title: str = "") -> plt.Figure:
    """Histogram of price values."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[column].dropna(), kde=True, ax=ax)
    ax.set_title(title or f"Distribution of {column}")
    return fig


def plot_multiple_products(df: pd.DataFrame, product_ids: List[str], x: str = "timestamp", y: str = "price") -> plt.Figure:
    """Compare multiple product price series."""
    filtered = df[df["product_id"].isin(product_ids)]
    return plot_time_series(filtered, x=x, y=y, hue="product_id", title="Price comparison")
