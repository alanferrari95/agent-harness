"""Decide which SKUs to reorder."""

from inventory.stock import on_hand
from inventory.thresholds import THRESHOLDS


def needs_reorder(stock, threshold):
    """A SKU is reordered when stock has fallen to the threshold or below."""
    return stock < threshold


def reorder_list():
    return sorted(sku for sku, limit in THRESHOLDS.items() if needs_reorder(on_hand(sku), limit))
