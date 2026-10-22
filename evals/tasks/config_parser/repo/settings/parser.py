def parse_config(text):
    """Parse 'key = value' lines. Blank lines and lines starting with # are ignored."""
    result = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        result[key] = value
    return result
