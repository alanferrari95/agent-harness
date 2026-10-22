def page_slice(items, page, per_page):
    """Return the items on a 1-based page."""
    if page < 1 or per_page < 1:
        raise ValueError("page and per_page must be positive")
    start = page * per_page
    return items[start:start + per_page]


def page_count(total, per_page):
    return (total + per_page - 1) // per_page
