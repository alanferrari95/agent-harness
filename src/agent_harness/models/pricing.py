"""Model IDs and prices, in one place.

Inventories age in weeks (the paper's methodological lesson), so this table carries the
date it was checked and where. Prices are USD per million tokens, standard context,
first-party API.
"""

from __future__ import annotations

from dataclasses import dataclass

VERIFIED_ON = "2026-10-22"
SOURCES = {
    "anthropic": "Claude API model table (claude-api reference, cached 2026-10-06)",
    "gemini": "https://ai.google.dev/gemini-api/docs/models and /pricing",
}


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float | None = None  # defaults to 0.1x input
    cache_write_per_mtok: float | None = None  # defaults to 1.25x input (5-minute TTL)
    note: str = ""


PRICES: dict[str, Price] = {
    # Anthropic
    "claude-opus-5-5": Price(4.00, 20.00, cache_read_per_mtok=0.20),
    "claude-sonnet-5-5": Price(2.00, 10.00, cache_read_per_mtok=0.20),
    "claude-haiku-5-5": Price(0.10, 0.50, note="prompts up to 100K tokens; $0.50/$2.50 beyond"),
    # Google Gemini
    "gemini-3.8-flash": Price(0.75, 3.75, note="rises to $1.50/$7.50 on 2027-01-01"),
    "gemini-3.5-flash-lite": Price(0.30, 2.50),
    "gemini-4-argon": Price(2.00, 10.00, note="rises to $4.00/$20.00 on 2026-12-01"),
}

def cost_usd(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    price = PRICES.get(model_id)
    if price is None:
        raise KeyError(
            f"No price for model '{model_id}'. Add it to agent_harness/models/pricing.py "
            "so cost budgets can be enforced."
        )
    read = price.cache_read_per_mtok
    write = price.cache_write_per_mtok
    read = price.input_per_mtok * 0.1 if read is None else read
    write = price.input_per_mtok * 1.25 if write is None else write
    total = (
        input_tokens * price.input_per_mtok
        + output_tokens * price.output_per_mtok
        + cache_read_tokens * read
        + cache_write_tokens * write
    )
    return total / 1_000_000
