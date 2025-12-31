import os
from dataclasses import dataclass, field
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

@dataclass
class PredictPCConfig:
    """Central configuration for PredictPC."""

    keepa_api_key: str = os.getenv("KEEPA_API_KEY", "")
    database_path: str = os.getenv("DATABASE_PATH", "data/predictpc.db")
    default_forecast_horizon: int = int(os.getenv("FORECAST_HORIZON", 7))
    minimum_history_days: int = int(os.getenv("MIN_HISTORY_DAYS", 60))
    product_catalog: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "gpu": [
                "B0C4F7KX1B",  # MSI GeForce RTX 4060 Ti Ventus 2X Black 8G OC
                "B096Y2TYKV",  # Gigabyte GeForce RTX 3060 GAMING OC V2
                "B0CBL3HQ68",  # MSI Gaming GeForce GT 1030
                "B0CVCG2VPK",  # ASUS Dual NVIDIA GeForce RTX 3050
                "B08WPRMVWB",  # MSI Gaming GeForce RTX 3060

            ],
            "cpu": [
                "B09VCJ171S",  # AMD Ryzen 5 5500 (Box)
                "B09VCHR1VH",  # AMD Ryzen 5 5600 (Box)
                "B0CGJ41C9W",  # Intel core i7-14700K
                "B09FX4D72T",  # Intel Core i5-12600K
                "B09FXDLX95",  # Intel Core i9-12900K
                "B09VCJ171S",  # AMD Ryzen 5 5500
                "B0BTZB7F88",  # AMD Ryzen 7 7800X
                "B0BBJDS62N",  # AMD Ryzen 5 7600X
                "B09VCHQHZ6",  # AMD Ryzen 7 5700X
            ],
        }
    )
    product_labels: Dict[str, str] = field(
        default_factory=lambda: {
            # GPU
            "B0C4F7KX1B": "MSI GeForce RTX 4060 Ti Ventus 2X Black 8G OC",
            "B096Y2TYKV": "Gigabyte GeForce RTX 3060 GAMING OC V2",
            "B0CBL3HQ68": "MSI Gaming GeForce GT 1030",
            "B0CVCG2VPK": "ASUS Dual NVIDIA GeForce RTX 3050",
            "B08WPRMVWB": "MSI Gaming GeForce RTX 3060",
            # CPU
            "B09VCJ171S": "AMD Ryzen 5 5500",
            "B09VCHR1VH": "AMD Ryzen 5 5600 (Box)",
            "B0CGJ41C9W": "Intel Core i7-14700K",
            "B09FX4D72T": "Intel Core i5-12600K",
            "B09FXDLX95": "Intel Core i9-12900K",
            "B0BTZB7F88": "AMD Ryzen 7 7800X",
            "B0BBJDS62N": "AMD Ryzen 5 7600X",
            "B09VCHQHZ6": "AMD Ryzen 7 5700X",
        }
    )

def get_config() -> PredictPCConfig:
    """Return a loaded configuration instance."""
    return PredictPCConfig()

