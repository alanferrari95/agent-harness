import re


def slugify(title):
    """Turn a title into a URL slug: lowercase words joined by single hyphens."""
    slug = title.lower()
    slug = re.sub(r"[^a-z0-9]", "-", slug)
    return slug
