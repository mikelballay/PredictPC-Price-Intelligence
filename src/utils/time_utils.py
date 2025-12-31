from datetime import datetime, timedelta, timezone
from typing import Optional


def keepa_timestamp_to_datetime(keepa_ts: int) -> datetime:
    """
    Convert Keepa's minute-based time to UTC datetime.

    Keepa timestamps are minutes since January 1, 2011 (Keepa epoch).
    """
    keepa_epoch = datetime(2011, 1, 1, tzinfo=timezone.utc)
    return keepa_epoch + timedelta(minutes=keepa_ts)


def ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensure a datetime is timezone-aware in UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
