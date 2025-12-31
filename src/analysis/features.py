import pandas as pd


def resample_daily(series_df: pd.DataFrame) -> pd.DataFrame:
    """Resample to daily frequency with forward fill."""
    df = series_df.copy()
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    df = df.set_index("timestamp")
    daily = df.resample("D").ffill()
    daily["product_id"] = daily["product_id"].iloc[0]
    return daily.reset_index()


def create_features(series_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a price series DataFrame with columns timestamp, product_id, price,
    create lag features and rolling stats.
    """
    df = resample_daily(series_df)
    df["lag_1"] = df["price"].shift(1)
    df["lag_7"] = df["price"].shift(7)
    df["roll_mean_7"] = df["price"].rolling(window=7, min_periods=1).mean()
    df["roll_std_7"] = df["price"].rolling(window=7, min_periods=1).std()
    df["roll_mean_30"] = df["price"].rolling(window=30, min_periods=1).mean()
    df["roll_std_30"] = df["price"].rolling(window=30, min_periods=1).std()
    return df
