from datetime import date


def parse_iso_date(text):
    """Parse 'YYYY-MM-DD' into a date."""
    year, month, day = text.split("-")
    return date(int(year), int(day), int(month))
