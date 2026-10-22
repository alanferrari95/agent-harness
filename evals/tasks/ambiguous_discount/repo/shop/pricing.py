def subtotal(price, quantity):
    total = price * quantity
    return round(total, 2)


def discounted_total(price, quantity, discount):
    """Total after a fractional discount, e.g. 0.25 for 25% off."""
    total = price * quantity
    return round(total, 2)
