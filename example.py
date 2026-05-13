def fetch_status():
    """
    Naive implementation that does a network call.
    This is intentionally fragile for the demo (network disabled in sandbox).
    """
    import requests

    resp = requests.get("https://example.com")
    return resp.status_code


def add(a, b):
    # INTENTIONAL BUG for demo: this incorrectly subtracts instead of adding
    return a - b
