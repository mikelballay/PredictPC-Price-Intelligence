from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table, create_engine, select
from sqlalchemy.engine import Engine

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

metadata = MetaData()

prices_table = Table(
    "prices",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("product_id", String, index=True, nullable=False),
    Column("timestamp", DateTime, index=True, nullable=False),
    Column("price", Float, nullable=False),
    Column("source", String, default="keepa"),
)


def get_engine(db_path: str) -> Engine:
    """Create and return a SQLAlchemy engine for SQLite."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", future=True)


def init_db(engine: Engine) -> None:
    """Create tables if they do not exist."""
    metadata.create_all(engine)
    logger.info("Database initialized at %s", engine.url)


def save_price_history(df: pd.DataFrame, engine: Engine, table_name: str = "prices") -> None:
    """Persist price history to SQLite."""
    if df.empty:
        logger.warning("No data to save for table %s", table_name)
        return
    df.to_sql(table_name, con=engine, if_exists="append", index=False)
    logger.info("Saved %s rows to %s", len(df), table_name)


def load_price_history(engine: Engine, product_id: str, table_name: str = "prices") -> pd.DataFrame:
    """Load price history for a single product."""
    stmt = select(prices_table).where(prices_table.c.product_id == product_id).order_by(prices_table.c.timestamp)
    with engine.connect() as conn:
        result = conn.execute(stmt)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df


def load_all_prices(engine: Engine, table_name: str = "prices") -> pd.DataFrame:
    """Load all price records."""
    with engine.connect() as conn:
        df = pd.read_sql_table(table_name, conn)
    return df
