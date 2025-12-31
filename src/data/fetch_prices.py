from typing import List

from src.config import get_config
from src.data.database import get_engine, init_db, save_price_history
from src.data.keepa_client import KeepaClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def fetch_and_store_products_price_history(product_ids: List[str]) -> None:
    """Fetch price history for each ASIN and store in the database."""
    config = get_config()
    engine = get_engine(config.database_path)
    init_db(engine)
    client = KeepaClient(api_key=config.keepa_api_key)

    for asin in product_ids:
        df = client.get_product_price_history(asin)
        save_price_history(df, engine)
        logger.info("ASIN %s -> %s records saved", asin, len(df))


def _flatten_products(catalog: dict) -> List[str]:
    seen = []
    for values in catalog.values():
        seen.extend(values)
    return seen


if __name__ == "__main__":
    cfg = get_config()
    products = _flatten_products(cfg.product_catalog)
    fetch_and_store_products_price_history(products)
