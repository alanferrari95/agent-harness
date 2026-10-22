VAT_RATE = 0.21


def add_vat(amount, rate=VAT_RATE):
    """Return the amount with VAT included."""
    return round(amount * rate, 2)
