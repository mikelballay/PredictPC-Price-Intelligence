import time
from typing import Dict, Optional

import pandas as pd
import requests

from src.utils.logging_utils import get_logger
from src.utils.time_utils import keepa_timestamp_to_datetime


class KeepaClient:
    """
    Lightweight Keepa API client.
    Fetches homogeneous NEW price time series suitable for forecasting.
    """

    BASE_URL = "https://api.keepa.com/product"

    def __init__(self, api_key: str, max_retries: int = 3, backoff_seconds: int = 5) -> None:
        self.api_key = api_key
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Internal request with retry + rate limit handling
    # ------------------------------------------------------------------
    def _request(self, asin: str) -> Optional[Dict]:
        params = {
            "key": self.api_key,
            "domain": 3,   # Amazon.de (estable para histórico)
            "asin": asin,
            "history": 1,
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(self.BASE_URL, params=params, timeout=30)

                if response.status_code == 429:
                    self.logger.warning(
                        "Keepa rate limited. Sleeping %s seconds (attempt %s/%s).",
                        self.backoff_seconds, attempt, self.max_retries
                    )
                    time.sleep(self.backoff_seconds)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.RequestException as exc:
                self.logger.error(
                    "Keepa request failed for ASIN %s (attempt %s/%s): %s",
                    asin, attempt, self.max_retries, exc
                )
                time.sleep(self.backoff_seconds)

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_product_price_history(self, asin: str) -> pd.DataFrame:
        """
        Fetch NEW price history from Keepa and return a tidy DataFrame.

        Columns:
        - product_id
        - timestamp (UTC)
        - price (float, EUR)
        - source
        """

        empty_df = pd.DataFrame(
            columns=["product_id", "timestamp", "price", "source"]
        )

        if not self.api_key:
            self.logger.error("KEEPA_API_KEY not set.")
            return empty_df

        payload = self._request(asin)
        if not payload or "products" not in payload or not payload["products"]:
            self.logger.warning("No product data returned for ASIN %s.", asin)
            return empty_df

        if payload.get("error"):
            self.logger.error("Keepa API error for ASIN %s: %s", asin, payload["error"])
            return empty_df

        tokens_left = payload.get("tokensLeft")
        if tokens_left is not None:
            self.logger.info("Keepa tokens left: %s", tokens_left)

        product = payload["products"][0]
        csv_history = product.get("csv", [])

        # --------------------------------------------------------------
        # Use ONLY NEW price (index 1)
        # --------------------------------------------------------------
        if len(csv_history) <= 1 or not csv_history[1]:
            self.logger.warning("No NEW price history for ASIN %s.", asin)
            return empty_df

        price_history = csv_history[1]  # NEW price (min new offer)

        timestamps = price_history[0::2]
        prices_cents = price_history[1::2]

        rows = []
        for ts, price in zip(timestamps, prices_cents):
            if price is None or price <= 0:
                continue

            rows.append(
                {
                    "product_id": asin,
                    "timestamp": keepa_timestamp_to_datetime(ts),
                    "price": price / 100.0,
                    "source": "keepa_new_de",
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            self.logger.warning("Parsed empty NEW price history for ASIN %s.", asin)
            return empty_df

        # --------------------------------------------------------------
        # Cleaning & basic sanity checks
        # --------------------------------------------------------------
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Forward-fill very small gaps only
        df["price"] = df["price"].ffill(limit=3)

        # Remove corrupted timestamps (paranoia check)
        df = df[df["timestamp"] >= pd.Timestamp("2000-01-01", tz=df["timestamp"].dt.tz)]

        if df.empty:
            self.logger.warning("Price history empty after cleaning for ASIN %s.", asin)
            return empty_df

        return df
