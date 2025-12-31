import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on PYTHONPATH when running via `streamlit run`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.analysis.eda import summarize_price_stats, compute_moving_averages
from src.analysis.features import resample_daily
from src.config import get_config
from src.data.database import get_engine, load_all_prices
from src.models.prophet_model import forecast_product_prophet, ProphetConfig
from src.models.gbm_model import GBMConfig, forecast_product_gbm
from src.utils.logging_utils import get_logger


logger = get_logger(__name__)
cfg = get_config()
product_labels = getattr(cfg, "product_labels", {})
engine = get_engine(cfg.database_path)

# -------------------------------------------------------------------------
# GLOBAL PAGE STYLE
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="PredictPC · Price Intelligence",
    page_icon="💻",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Manrope:wght@500;600;700&display=swap');

    :root {
        --bg: #020617;
        --bg-alt: #020617;
        --panel: rgba(15,23,42,0.96);
        --card: rgba(15,23,42,0.96);
        --accent: #5ad8ff;           /* azul neon principal */
        --accent-soft: rgba(90,216,255,0.18);
        --accent-2: #a855f7;
        --accent-2-soft: rgba(168,85,247,0.18);
        --text: #f9fafb;             /* texto casi blanco para contraste */
        --muted: #cbd5e1;
        --border: rgba(148,163,184,0.4);
        --danger: #fb7185;
        --success: #4ade80;
        --warning: #facc15;
    }

    /* Background with subtle gradients */
    .stApp {
        background:
            radial-gradient(circle at 10% 20%, rgba(90,216,255,0.18), transparent 35%),
            radial-gradient(circle at 80% 0%, rgba(168,85,247,0.16), transparent 40%),
            radial-gradient(circle at 0% 100%, rgba(34,197,94,0.12), transparent 40%),
            var(--bg);
        color: var(--text);
        font-family: 'Manrope', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1220px;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', system-ui, sans-serif !important;
        letter-spacing: -0.02em;
        color: var(--text);
    }

    h1 { font-weight: 700 !important; font-size: 2.1rem !important; }
    h2 { font-weight: 600 !important; }
    h3 { font-weight: 600 !important; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #020617;
        border-right: 1px solid rgba(148,163,184,0.45);
    }

    /* Texto general sidebar en azul neon con buen contraste */
    section[data-testid="stSidebar"] * {
        color: #e5f6ff !important;
        font-weight: 500;
    }

    /* Labels de inputs en sidebar */
    section[data-testid="stSidebar"] label {
        color: #5ad8ff !important;
        font-size: 0.92rem !important;
    }

    /* Selectbox del sidebar */
    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #e5f6ff !important;
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: rgba(15,23,42,0.95) !important;
        border-radius: 10px !important;
    }

    /* Sliders en sidebar */
    section[data-testid="stSidebar"] .stSlider label {
        color: #5ad8ff !important;
    }
    section[data-testid="stSidebar"] .stSlider span {
        color: #e5f6ff !important;
    }

    /* Checkbox texto */
    section[data-testid="stSidebar"] .stCheckbox label {
        color: #e5f6ff !important;
    }

    /* Texto pequeño / tips sidebar */
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] p {
        color: #bfe9ff !important;
        font-size: 0.8rem !important;
    }

    /* Header sidebar */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f9fafb !important;
    }

    /* Cards */
    .card {
        background: linear-gradient(145deg, rgba(15,23,42,0.98), rgba(15,23,42,0.96));
        border-radius: 18px;
        padding: 1rem 1.25rem;
        border: 1px solid var(--border);
        box-shadow: 0 18px 45px rgba(15,23,42,0.9);
    }

    .card-soft {
        background: radial-gradient(circle at top left, rgba(90,216,255,0.18), transparent 55%),
                    radial-gradient(circle at bottom right, rgba(168,85,247,0.16), transparent 55%),
                    rgba(15,23,42,0.97);
        border-radius: 18px;
        padding: 1.1rem 1.3rem;
        border: 1px solid rgba(148,163,184,0.55);
        box-shadow: 0 18px 40px rgba(15,23,42,0.95);
    }

    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.18rem 0.6rem;
        border-radius: 999px;
        background: rgba(15,23,42,0.9);
        border: 1px solid rgba(148,163,184,0.6);
        font-size: 0.75rem;
        color: #e5f6ff;
        text-transform: uppercase;
        letter-spacing: 0.09em;
    }

    .pill-dot {
        width: 7px; height: 7px;
        border-radius: 999px;
        background: var(--accent);
        box-shadow: 0 0 14px rgba(90,216,255,0.95);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(120deg, var(--accent), var(--accent-2));
        color: #020617;
        border: none;
        border-radius: 999px;
        padding: 0.55rem 1.1rem;
        font-weight: 600;
        font-size: 0.9rem;
        box-shadow: 0 12px 28px rgba(90,216,255,0.45);
    }
    .stButton>button:hover {
        filter: brightness(1.04);
        box-shadow: 0 16px 40px rgba(90,216,255,0.65);
    }

    /* Tabs */
    .stTabs [role="tablist"] {
        border-radius: 999px;
        padding: 0.15rem;
        background: rgba(15,23,42,0.96);
        border: 1px solid rgba(148,163,184,0.5);
    }
    .stTabs [role="tablist"] button {
        border-radius: 999px;
        padding: 0.45rem 0.9rem;
        font-size: 0.9rem;
    }
    .stTabs [role="tablist"] button[data-baseweb="tab"] {
        background: transparent;
        color: #cbd5e1;
    }
    .stTabs [role="tablist"] button[data-baseweb="tab"][aria-selected="true"] {
        background: radial-gradient(circle at top left, rgba(90,216,255,0.28), rgba(168,85,247,0.58));
        color: #f9fafb;
        box-shadow: 0 12px 30px rgba(15,23,42,0.95);
        border: 1px solid rgba(148,163,184,0.7);
    }

    /* Metrics */
    .metric-card {
        border-radius: 16px;
        padding: 0.7rem 0.9rem;
        border: 1px solid rgba(148,163,184,0.6);
        background: radial-gradient(circle at top left, rgba(90,216,255,0.26), rgba(15,23,42,0.98));
        color: #e5f6ff;
    }
    div[data-testid="stMetricValue"] {
        color: var(--accent);
        font-weight: 600;
    }
    div[data-testid="stMetricDelta"] {
        color: var(--accent-2);
    }

    /* Tables / dataframes */
    .stDataFrame, .stTable { color: var(--text); }
    .stDataFrame thead tr th { color: var(--accent) !important; font-weight: 600 !important; }
    .stTable thead tr th { color: var(--accent) !important; font-weight: 600 !important; }
    .stTable tbody tr td { color: var(--text) !important; }

    /* Inputs fuera del sidebar (labels) */
    label {
        color: var(--text) !important;
    }

    /* Small text */
    .muted {
        color: var(--muted);
        font-size: 0.8rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------------
# LOGIC FUNCTIONS
# -------------------------------------------------------------------------


def compute_buy_now_indicator(
    forecast_df: pd.DataFrame,
    last_price: float,
    up_threshold: float = 0.02,
    drop_alert_pct: float = 0.05,
) -> dict:
    """
    Trend/volatility-based indicator.

    - slope_pct: pendiente entre primer y último forecast (en %)
    - prob_up: proporción de puntos de forecast por encima del último precio observado
    - signal: semáforo en base a pendiente y prob_up
    - drop_alert: si hay caída esperada > drop_alert_pct
    """
    if forecast_df.empty or forecast_df["forecast"].dropna().empty:
        return {
            "signal": "WAIT",
            "color": "yellow",
            "reason": "No forecast available.",
            "prob_up": 0.0,
            "drop_alert": False,
        }

    fc = forecast_df["forecast"].dropna()
    slope_pct = (fc.iloc[-1] - fc.iloc[0]) / max(fc.iloc[0], 1e-9)
    prob_up = (fc > last_price).mean()
    vol_pct = fc.pct_change().std() or 0.0

    signal = "WAIT"
    color = "yellow"
    reason = f"Slope {slope_pct:.1%}, prob up {prob_up:.0%}, vol {vol_pct:.1%}"

    if slope_pct > up_threshold and prob_up >= 0.6:
        signal = "BUY_NOW"
        color = "green"
        reason = f"Uptrend (+{slope_pct:.1%}) and prob up {prob_up:.0%}"
    elif slope_pct < -up_threshold:
        signal = "WAIT"
        color = "red"
        reason = f"Downtrend ({slope_pct:.1%}); better to wait"

    drop_alert = False
    if last_price and ((fc.min() - last_price) / max(last_price, 1e-9)) <= -drop_alert_pct:
        drop_alert = True

    return {"signal": signal, "color": color, "reason": reason, "prob_up": prob_up, "drop_alert": drop_alert}


def load_data() -> pd.DataFrame:
    try:
        df = load_all_prices(engine)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[df["timestamp"] >= pd.Timestamp("2000-01-01")]  # guardrail against corrupt early dates
        df = df[df["price"] > 0]  # drop zero/negative placeholders
        return df.sort_values("timestamp")
    except Exception as exc:  # pragma: no cover - runtime safety
        logger.error("Failed to load data: %s", exc)
        return pd.DataFrame(columns=["product_id", "timestamp", "price", "source"])


def _prepare_training_df(df: pd.DataFrame, lookback_days: int) -> tuple[pd.DataFrame, dict]:
    """
    Clean and trim the series before modeling.

    - filter price > 0 and timestamp >= 2015-01-01
    - limit to recent window (lookback_days)
    - compute daily coverage (solo informativa, no cambiamos frecuencia)
    """
    info = {"freq": "D", "coverage": None}
    if df.empty:
        return df, info

    df_clean = df.copy()
    df_clean = df_clean[df_clean["price"] > 0]
    df_clean = df_clean[df_clean["timestamp"] >= pd.Timestamp("2015-01-01")]
    df_clean = df_clean.sort_values("timestamp")
    df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])
    if lookback_days and lookback_days > 0:
        cutoff = df_clean["timestamp"].max() - pd.Timedelta(days=lookback_days)
        df_clean = df_clean[df_clean["timestamp"] >= cutoff]

    daily_counts = df_clean.set_index("timestamp")["price"].resample("D").count()
    coverage = float((daily_counts > 0).mean() if not daily_counts.empty else 0.0)
    info["coverage"] = coverage

    # Ensure product_id column is present for downstream models
    if "product_id" not in df_clean.columns:
        pid = df["product_id"].iloc[0] if "product_id" in df.columns and not df.empty else "unknown"
        df_clean["product_id"] = pid

    return df_clean.reset_index(drop=True), info


def build_comparison_df(df_all: pd.DataFrame, products: list[str], normalize: bool = True, window_days: int = 360) -> pd.DataFrame:
    """Assemble multi-product dataframe for comparison chart."""
    if df_all.empty or not products:
        return pd.DataFrame()
    subset = df_all[df_all["product_id"].isin(products)].copy()
    subset["timestamp"] = pd.to_datetime(subset["timestamp"])

    subset_price = subset[subset["price"] > 0]
    if subset_price.empty:
        subset_price = subset  # fallback: allow <=0 if nothing else
    subset = subset_price

    if subset.empty:
        return pd.DataFrame()
    max_ts = subset["timestamp"].max()
    cutoff = max_ts - pd.Timedelta(days=window_days)
    subset_in_window = subset[subset["timestamp"] >= cutoff]
    if subset_in_window.empty:
        subset_in_window = subset  # fallback: use all available if recent window is empty
    subset = subset_in_window

    frames = []
    for pid, group in subset.groupby("product_id"):
        daily = resample_daily(group)[["timestamp", "price"]].copy()
        daily = daily.dropna(subset=["price"])
        daily["product_id"] = pid
        if daily.empty:
            continue
        if normalize:
            base = daily["price"].iloc[0] if pd.notnull(daily["price"].iloc[0]) else daily["price"].dropna().iloc[0]
            base = base if base != 0 else 1e-9
            daily["value"] = (daily["price"] / base) * 100.0
        else:
            daily["value"] = daily["price"]
        frames.append(daily[["timestamp", "product_id", "value"]])
    return pd.concat(frames).sort_values("timestamp") if frames else pd.DataFrame()


def monthly_return_heatmap_data(df_all: pd.DataFrame, months_back: int = 12) -> pd.DataFrame:
    """Compute monthly percentage change per product for heatmap."""
    if df_all.empty:
        return pd.DataFrame()
    df = df_all[df_all["price"] > 0].copy()
    if df.empty:
        return pd.DataFrame()
    max_ts = df["timestamp"].max()
    cutoff = max_ts - pd.DateOffset(months=months_back + 1)
    df = df[df["timestamp"] >= cutoff]
    if df.empty:
        return pd.DataFrame()

    monthly = (
        df.set_index("timestamp")
        .groupby("product_id")["price"]
        .resample("M")
        .last()
        .groupby(level=0)
        .pct_change()
        .reset_index()
        .rename(columns={"price": "return"})
    )
    monthly["month"] = monthly["timestamp"].dt.strftime("%Y-%m")
    pivot = monthly.pivot(index="product_id", columns="month", values="return") * 100.0
    return pivot.sort_index(axis=1)


def rolling_volatility_data(df_all: pd.DataFrame, products: list[str], window_days: int = 30, lookback_days: int = 360) -> pd.DataFrame:
    """Compute rolling std of daily returns for selected products."""
    if df_all.empty or not products:
        return pd.DataFrame()
    subset = df_all[df_all["product_id"].isin(products)].copy()
    subset = subset[subset["price"] > 0]
    if subset.empty:
        return pd.DataFrame()
    max_ts = subset["timestamp"].max()
    cutoff = max_ts - pd.Timedelta(days=lookback_days)
    subset = subset[subset["timestamp"] >= cutoff]
    frames = []
    for pid, group in subset.groupby("product_id"):
        daily = resample_daily(group)[["timestamp", "price"]].copy()
        daily = daily.sort_values("timestamp")
        daily["ret"] = daily["price"].pct_change()
        daily["vol"] = daily["ret"].rolling(window=window_days, min_periods=5).std() * 100.0
        daily["product_id"] = pid
        frames.append(daily[["timestamp", "product_id", "vol"]])
    return pd.concat(frames).dropna(subset=["vol"]).sort_values("timestamp") if frames else pd.DataFrame()


def comparison_stats(df_all: pd.DataFrame, products: list[str], window_days: int = 360, normalize: bool = True) -> pd.DataFrame:
    """Return summary stats per product over the selected window."""
    if df_all.empty or not products:
        return pd.DataFrame()
    subset = df_all[df_all["product_id"].isin(products)].copy()
    subset = subset[subset["price"] > 0]
    if subset.empty:
        return pd.DataFrame()
    max_ts = subset["timestamp"].max()
    cutoff = max_ts - pd.Timedelta(days=window_days)
    subset = subset[subset["timestamp"] >= cutoff]
    if subset.empty:
        return pd.DataFrame()

    rows = []
    for pid, group in subset.groupby("product_id"):
        daily = resample_daily(group)[["timestamp", "price"]].dropna().sort_values("timestamp")
        if daily.empty:
            continue
        start = daily["price"].iloc[0]
        end = daily["price"].iloc[-1]
        ret_pct = (end - start) / max(start, 1e-9) * 100
        max_price = daily["price"].max()
        min_price = daily["price"].min()
        drawdown = (min_price - max_price) / max(max_price, 1e-9) * 100
        vol = daily["price"].pct_change().std() * 100
        latest = daily["price"].iloc[-1]
        base = start if normalize else 1.0
        rows.append(
            {
                "product_id": pid,
                "price_change_%": round(ret_pct, 2),
                "volatility_%": round(vol if pd.notnull(vol) else 0.0, 2),
                "drawdown_%": round(drawdown, 2),
                "last_price": round(latest, 2),
                "base_price": round(start, 2),
            }
        )
    return pd.DataFrame(rows)


def performance_table(df_all: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """Compute return %, volatility %, and last price per product over window."""
    if df_all.empty:
        return pd.DataFrame()
    max_ts = df_all["timestamp"].max()
    cutoff = max_ts - pd.Timedelta(days=window_days)
    df = df_all[df_all["timestamp"] >= cutoff].copy()
    df = df[df["price"] > 0]
    rows = []
    for pid, group in df.groupby("product_id"):
        daily = resample_daily(group)[["timestamp", "price"]].dropna().sort_values("timestamp")
        if daily.empty:
            continue
        start = daily["price"].iloc[0]
        end = daily["price"].iloc[-1]
        ret_pct = (end - start) / max(start, 1e-9) * 100
        vol = daily["price"].pct_change().std() * 100
        rows.append(
            {
                "product_id": pid,
                "price_change_%": round(ret_pct, 2),
                "volatility_%": round(vol if pd.notnull(vol) else 0.0, 2),
                "last_price": round(end, 2),
            }
        )
    return pd.DataFrame(rows)


def run_model(
    df: pd.DataFrame,
    model_name: str,
    horizon: int,
    use_log: bool = True,
    lookback_days: int = 365,
) -> tuple[pd.DataFrame, dict]:
    last_price = df["price"].iloc[-1] if not df.empty else None

    df_train, info = _prepare_training_df(df, lookback_days)

    if model_name == "Prophet":
        try:
            cfg_prophet = ProphetConfig(
                changepoint_prior_scale=0.35,
                changepoint_range=0.9,
                seasonality_mode="additive",
                weekly_seasonality=True,
                yearly_seasonality=False,
                seasonality_prior_scale=1.0,
                log_transform=True,
            )
            fc = forecast_product_prophet(df_train, horizon, config=cfg_prophet)
        except Exception as exc:
            st.error(f"Prophet failed: {exc}")
            fc = pd.DataFrame(columns=["timestamp", "price", "forecast", "lower", "upper"])

    elif model_name == "GradientBoosting":
        try:
            gbm_cfg = GBMConfig(
                log_transform=use_log,
                outlier_clip=(0.01, 0.99),
            )
            min_points = 20
            if len(df_train) < min_points:
                st.warning(
                    f"Too few clean points ({len(df_train)}) to train GBM."
                )
                fc = pd.DataFrame(columns=["timestamp", "price", "forecast", "lower", "upper"])
            else:
                fc = forecast_product_gbm(df_train, horizon, config=gbm_cfg)
        except Exception as exc:
            st.error(f"Gradient Boosting failed: {exc}")
            fc = pd.DataFrame(columns=["timestamp", "price", "forecast", "lower", "upper"])

    else:
        st.warning("Model not implemented; showing empty forecast.")
        fc = pd.DataFrame(columns=["timestamp", "price", "forecast", "lower", "upper"])

    # Bias-correct so the forecast starts at the last observed price
    if last_price is not None and not fc.empty and "forecast" in fc:
        delta = last_price - fc["forecast"].iloc[0]
        fc["forecast"] = fc["forecast"] + delta
        if "lower" in fc:
            fc["lower"] = fc["lower"] + delta
        if "upper" in fc:
            fc["upper"] = fc["upper"] + delta

    return fc, info


# -------------------------------------------------------------------------
# MAIN DASHBOARD
# -------------------------------------------------------------------------


def main() -> None:
    # ---------- HEADER ----------
    catalog = cfg.product_catalog
    df_all = load_data()

    st.markdown(
        """
        <div class="card-soft" style="margin-bottom: 1.2rem;">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1.2rem;">
                <div>
                    <div class="pill">
                        <span class="pill-dot"></span>
                        <span>PROPHET · GRADIENT BOOSTING</span>
                    </div>
                    <h1 style="margin-top:0.5rem;margin-bottom:0.25rem;">PredictPC · Price Intelligence</h1>
                    <p class="muted" style="max-width:540px;">
                        Interactive dashboard to analyze price history and generate forecasts
                        with Prophet and Gradient Boosting models on hardware products.
                    </p>
                </div>
                <div style="display:flex;gap:0.75rem;align-items:flex-end;">
                    <div class="metric-card">
                        <div style="font-size:0.8rem;color:var(--muted);">Products</div>
                        <div style="font-size:1.25rem;font-weight:600;">
                            {n_products}
                        </div>
                    </div>
                    <div class="metric-card">
                        <div style="font-size:0.8rem;color:var(--muted);">Price records</div>
                        <div style="font-size:1.25rem;font-weight:600;">
                            {n_rows}
                        </div>
                    </div>
                    <div class="metric-card">
                        <div style="font-size:0.8rem;color:var(--muted);">Available period</div>
                        <div style="font-size:1.25rem;font-weight:600;">
                            {period}
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """.format(
            n_products=df_all["product_id"].nunique() if not df_all.empty else 0,
            n_rows=len(df_all),
            period=f"{df_all['timestamp'].min().date()} → {df_all['timestamp'].max().date()}"
            if not df_all.empty
            else "–",
        ),
        unsafe_allow_html=True,
    )

    # ---------- SIDEBAR ----------
    st.sidebar.header("Control Panel ⚙️")

    category = st.sidebar.selectbox("Category", list(catalog.keys()))
    product_options = [
        (product_labels.get(asin, asin), asin) for asin in catalog[category]
    ]
    selected_option = st.sidebar.selectbox(
        "Product",
        product_options,
        format_func=lambda opt: opt[0],
    )
    product_id = selected_option[1]

    model_name = st.sidebar.selectbox(
        "Model",
        ["Prophet", "GradientBoosting"],
    )

    horizon = st.sidebar.slider(
        "Forecast horizon (days)",
        min_value=30,
        max_value=90,
        value=max(cfg.default_forecast_horizon, 30),
        step=5,
    )
    lookback_days = st.sidebar.slider(
        "Training window (days)",
        min_value=90,
        max_value=720,
        value=365,
        step=15,
    )
    use_log = True

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "💡 Tip: shorter windows capture recent moves; longer windows smooth the series but can flatten forecasts."
    )

    # ---------- DATA SELECTION ----------
    product_df = df_all[df_all["product_id"] == product_id]

    # ---------- TABS LAYOUT ----------
    tab_overview, tab_forecast, tab_compare, tab_insights = st.tabs(
        ["📊 Overview", "🔮 Forecast", "🛰️ Comparison", "💡 Insights"]
    )

    # ---------------- OVERVIEW TAB ----------------
    with tab_overview:
        col_info, col_stats = st.columns([1.4, 1])

        with col_info:
            st.markdown("#### Product info")
            product_name = product_labels.get(product_id, product_id)
            st.markdown(f"**{product_name}** · ASIN: `{product_id}`")
            if product_df.empty:
                st.info("No historical data for this product.")
            else:
                stats = summarize_price_stats(product_df)
                st.table(stats)

        with col_stats:
            if not product_df.empty:
                last_price = product_df["price"].iloc[-1]
                first_price = product_df["price"].iloc[0]
                change_pct = (last_price - first_price) / max(first_price, 1e-9) * 100

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Last price", f"{last_price:,.2f}")
                with c2:
                    st.metric("Total change", f"{change_pct:+.1f}%")

        st.markdown("#### Price history (recent)")
        with st.spinner("Drawing history chart..."):
            history_window_days = 720
            if not product_df.empty:
                hist_cutoff = product_df["timestamp"].max() - pd.Timedelta(days=history_window_days)
                hist_df = product_df[product_df["timestamp"] >= hist_cutoff]
            else:
                hist_df = product_df

            enriched = compute_moving_averages(hist_df, windows=[7])
            if not enriched.empty:
                fig_hist = go.Figure()
                fig_hist.add_trace(
                    go.Scatter(
                        x=enriched["timestamp"],
                        y=enriched["price"],
                        mode="lines",
                        name="Price",
                        line=dict(color="#5ad8ff", width=2),
                    )
                )
                fig_hist.add_trace(
                    go.Scatter(
                        x=enriched["timestamp"],
                        y=enriched["ma_7"],
                        mode="lines",
                        name="MA 7d",
                        line=dict(color="#a855f7", width=2, dash="dash"),
                    )
                )
                fig_hist.update_layout(
                    xaxis_title="Fecha",
                    yaxis_title="Precio",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.3)",
                    font=dict(color="#e5e7eb"),
                    legend=dict(orientation="h", y=-0.2, font=dict(color="#5ad8ff")),
                )
                st.plotly_chart(fig_hist, width="stretch")
            else:
                st.info("No historical data for this product.")

    # ---------------- FORECAST TAB ----------------
    with tab_forecast:
        st.markdown("#### Forecast (last 30 days + horizon)")

        if len(product_df) < cfg.minimum_history_days:
            st.warning(
                f"Not enough history ({len(product_df)} days) to forecast. "
                f"Need at least {cfg.minimum_history_days} days."
            )
        else:
            forecast_df, info = run_model(
                product_df,
                model_name,
                horizon,
                use_log=use_log,
                lookback_days=lookback_days,
            )

            if forecast_df.empty:
                st.warning("No forecast available.")
            else:
                recent_start = product_df["timestamp"].max() - pd.Timedelta(days=90)
                recent_hist = resample_daily(product_df)
                recent_hist = recent_hist[recent_hist["timestamp"] >= recent_start]

                fig_fc = go.Figure()
                fig_fc.add_trace(
                    go.Scatter(
                        x=recent_hist["timestamp"],
                        y=recent_hist["price"],
                        mode="lines",
                        name="History",
                        line=dict(color="#5ad8ff", width=2),
                    )
                )
                fig_fc.add_trace(
                    go.Scatter(
                        x=forecast_df["timestamp"],
                        y=forecast_df["forecast"],
                        mode="lines+markers",
                        name="Forecast",
                        line=dict(color="#a855f7", width=2),
                        marker=dict(size=6),
                    )
                )

                if {"lower", "upper"}.issubset(forecast_df.columns):
                    fig_fc.add_trace(
                        go.Scatter(
                            x=pd.concat(
                                [forecast_df["timestamp"], forecast_df["timestamp"][::-1]]
                            ),
                            y=pd.concat(
                                [forecast_df["upper"], forecast_df["lower"][::-1]]
                            ),
                            fill="toself",
                            fillcolor="rgba(90,216,255,0.2)",
                            line=dict(color="rgba(255,255,255,0)"),
                            hoverinfo="skip",
                            name="CI",
                        )
                    )

                # Connect last observed point to first forecast for visual continuity
                if not recent_hist.empty and not forecast_df.empty:
                    fig_fc.add_trace(
                        go.Scatter(
                            x=[
                                recent_hist["timestamp"].iloc[-1],
                                forecast_df["timestamp"].iloc[0],
                            ],
                            y=[
                                recent_hist["price"].iloc[-1],
                                forecast_df["forecast"].iloc[0],
                            ],
                            mode="lines",
                            name="Bridge",
                            line=dict(color="#7dd3fc", width=1, dash="dot"),
                            showlegend=False,
                        )
                    )

                x_min = recent_start
                x_max = forecast_df["timestamp"].max() + pd.Timedelta(days=1)
                fig_fc.update_layout(
                    xaxis_title="Fecha",
                    yaxis_title="Precio",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.3)",
                    font=dict(color="#e5e7eb"),
                    legend=dict(orientation="h", y=-0.2, font=dict(color="#5ad8ff")),
                )
                fig_fc.update_xaxes(
                    dtick=86400000 * 5,
                    tickformat="%Y-%m-%d",
                    range=[x_min, x_max],
                )
                st.plotly_chart(fig_fc, width="stretch")

                st.markdown("##### Forecast table")
                st.dataframe(
                    forecast_df[["timestamp", "forecast", "lower", "upper"]].tail(
                        horizon
                    ),
                    width="stretch",
                )

                # Quick indicators
                last_price = product_df["price"].iloc[-1]
                last_fc = forecast_df["forecast"].iloc[-1]
                delta_pct = (last_fc - last_price) / max(last_price, 1e-9) * 100

                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Last observed price", f"{last_price:,.2f}")
                with col_m2:
                    st.metric("Forecast at horizon end", f"{last_fc:,.2f}")
                with col_m3:
                    st.metric("Δ vs last price", f"{delta_pct:+.1f}%")

                indicator = compute_buy_now_indicator(forecast_df, last_price)
                st.markdown("##### Buy-now indicator")

                emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(
                    indicator["color"], "🟡"
                )
                st.info(f"{emoji} {indicator['signal']}: {indicator['reason']}")
                if indicator["drop_alert"]:
                    st.warning("⚠️ Potential sharp drop expected. Consider waiting.")

                st.caption(
                    f"Series trained with a window of {lookback_days} days, frequency {info.get('freq','D')}, "
                    f"daily coverage ~{(info.get('coverage') or 0)*100:.0f}%."
                )

    # ---------------- COMPARISON TAB ----------------
    with tab_compare:
        st.markdown("#### Multi-product comparison")
        compare_ids = sorted(df_all["product_id"].unique()) if not df_all.empty else []
        compare_options = [(product_labels.get(asin, asin), asin) for asin in compare_ids]
        default_ids = [product_id] if product_id in compare_ids else compare_ids[:3]
        default_selection = [opt for opt in compare_options if opt[1] in default_ids]
        selected_options = st.multiselect(
            "Products to compare",
            options=compare_options,
            default=default_selection,
            format_func=lambda opt: opt[0],
        )
        selected_products = [asin for _, asin in selected_options]
        normalize_idx = st.checkbox("Normalize to index 100", value=True, key="normalize_compare")
        window_compare = st.slider("History window (days)", min_value=90, max_value=720, value=360, step=30)

        comp_df = build_comparison_df(df_all, selected_products, normalize=normalize_idx, window_days=window_compare)
        # Fallback: si no hay datos en la ventana, intenta todo el histórico
        if comp_df.empty and not df_all.empty:
            comp_df = build_comparison_df(df_all, selected_products, normalize=normalize_idx, window_days=10_000)

        if comp_df.empty:
            subset = df_all[df_all["product_id"].isin(selected_products)]
            counts = (
                subset.groupby("product_id")["price"]
                .agg(n="count", n_pos=lambda s: (s > 0).sum())
                .reset_index()
            )
            st.info("No data for the selected products in the chosen window.")
            st.dataframe(counts, width="stretch")
        else:
            fig_cmp = go.Figure()
            y_label = "Index (base=100)" if normalize_idx else "Price"
            # baseline line
            if normalize_idx:
                fig_cmp.add_trace(
                    go.Scatter(
                        x=comp_df["timestamp"],
                        y=[100] * len(comp_df),
                        mode="lines",
                        name="Baseline 100",
                        line=dict(color="#334155", width=1, dash="dot"),
                        showlegend=True,
                    )
                )
            for pid, series in comp_df.groupby("product_id"):
                fig_cmp.add_trace(
                    go.Scatter(
                        x=series["timestamp"],
                        y=series["value"],
                        mode="lines",
                        name=pid,
                        line=dict(width=2),
                    )
                )
            fig_cmp.update_layout(
                xaxis_title="Fecha",
                yaxis_title=y_label,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.3)",
                font=dict(color="#e5e7eb"),
                legend=dict(orientation="h", y=-0.25, font=dict(color="#5ad8ff")),
            )
            st.plotly_chart(fig_cmp, width="stretch")
            stats_df = comparison_stats(df_all, selected_products, window_days=window_compare, normalize=normalize_idx)
            if not stats_df.empty:
                st.markdown("##### Comparison stats (price changes over window)")
                display_stats = stats_df.rename(
                    columns={
                        "price_change_%": "Price change (%)",
                        "volatility_%": "Volatility (%)",
                        "drawdown_%": "Drawdown (%)",
                        "last_price": "Last price",
                        "base_price": "Start price",
                    }
                )
                st.dataframe(display_stats, width="stretch")

        st.markdown("#### Rolling 30d volatility (return std)")
        vol_df = rolling_volatility_data(
            df_all[df_all["product_id"].isin(selected_products)] if selected_products else df_all,
            products=selected_products if selected_products else list(df_all["product_id"].unique()),
            window_days=30,
            lookback_days=window_compare,
        )
        if vol_df.empty:
            st.info("Not enough data to compute volatility.")
        else:
            fig_vol = go.Figure()
            for pid, series in vol_df.groupby("product_id"):
                fig_vol.add_trace(
                    go.Scatter(
                        x=series["timestamp"],
                        y=series["vol"],
                        mode="lines",
                        name=f"{pid} vol",
                        line=dict(width=2, dash="dot"),
                    )
                )
            fig_vol.update_layout(
                xaxis_title="Date",
                yaxis_title="Volatility 30d (%)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.3)",
                font=dict(color="#e5e7eb"),
                legend=dict(orientation="h", y=-0.2, font=dict(color="#5ad8ff")),
            )
            st.plotly_chart(fig_vol, width="stretch")
            if not stats_df.empty:
                st.caption(f"Returns/vol/drawdown computed over last {window_compare} days.")

    # ---------------- INSIGHTS TAB ----------------
    with tab_insights:
        st.markdown("#### Highlights")
        window_insights = st.slider("Insights window (days)", min_value=30, max_value=360, value=90, step=15)
        perf = performance_table(df_all, window_days=window_insights)
        if perf.empty:
            st.info("No data available for insights.")
        else:
            # Top movers by absolute price change
            top_changes = (
                perf.sort_values("price_change_%", key=lambda s: s.abs(), ascending=False)
                .head(6)
            )
            gainers = perf.sort_values("price_change_%", ascending=False).head(3)
            losers = perf.sort_values("price_change_%").head(3)
            volatiles = perf.sort_values("volatility_%", ascending=False).head(3)

            fig_changes = go.Figure()
            fig_changes.add_trace(
                go.Bar(
                    x=top_changes["product_id"],
                    y=top_changes["price_change_%"],
                    name="Price change (%)",
                    marker_color=[
                        "#5ad8ff" if v >= 0 else "#fb7185"
                        for v in top_changes["price_change_%"]
                    ],
                )
            )
            fig_changes.update_layout(
                title="Top movers by absolute price change (last window)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.3)",
                font=dict(color="#e5e7eb"),
                legend=dict(orientation="h", y=-0.2, font=dict(color="#5ad8ff")),
                yaxis_title="Price change (%)",
            )
            st.plotly_chart(fig_changes, width="stretch")

            col_g, col_l, col_v = st.columns(3)
            with col_g:
                st.markdown("**Biggest price increases**")
                st.dataframe(gainers, width="stretch")
            with col_l:
                st.markdown("**Biggest price drops**")
                st.dataframe(losers, width="stretch")
            with col_v:
                st.markdown("**Most volatile prices**")
                st.dataframe(volatiles, width="stretch")

            st.markdown("##### Full summary")
            st.dataframe(
                perf.rename(
                    columns={
                        "price_change_%": "Price change (%)",
                        "volatility_%": "Volatility (%)",
                        "last_price": "Last price",
                    }
                ),
                width="stretch",
            )

        st.markdown("##### Model notes")
        st.markdown(
            """
            - **Prophet**: additive seasonality, optional weekly seasonality, log-transform for stability.<br>
            - **Gradient Boosting**: tree-based regressor on lag/rolling features with log/clip options.<br>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
