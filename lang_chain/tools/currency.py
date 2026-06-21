"""Currency exchange rate tool."""

from __future__ import annotations

import json
from urllib.error import URLError
from urllib.request import urlopen

from langchain_core.tools import tool

EXCHANGE_API_URL = "https://open.er-api.com/v6/latest"
REQUEST_TIMEOUT_SEC = 10


@tool
def get_currency_rate(
    from_currency: str,
    to_currency: str,
    amount: float = 1.0,
) -> str:
    """Get exchange rate between ISO 4217 codes (USD, EUR, RUB, etc.)."""
    base = from_currency.strip().upper()
    target = to_currency.strip().upper()
    if not base or not target:
        return "Both from_currency and to_currency are required."

    if base == target:
        return (
            f"1 {base} = 1 {target}. "
            f"{amount} {base} = {amount} {target}."
        )

    try:
        rate, updated = _fetch_rate(base, target)
    except (URLError, ValueError, KeyError) as error:
        return f"Currency lookup failed: {error}"

    converted = amount * rate
    return (
        f"1 {base} = {rate:.6f} {target} (updated: {updated}). "
        f"{amount} {base} = {converted:.4f} {target}."
    )


def _fetch_rate(base: str, target: str) -> tuple[float, str]:
    url = f"{EXCHANGE_API_URL}/{base}"
    with urlopen(url, timeout=REQUEST_TIMEOUT_SEC) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if payload.get("result") != "success":
        raise ValueError("Exchange API returned an error.")

    rates = payload.get("rates", {})
    if target not in rates:
        raise ValueError(f"Unknown currency code: {target}")

    updated = str(payload.get("time_last_update_utc", "unknown"))
    return float(rates[target]), updated
