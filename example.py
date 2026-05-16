def fetch_status():
    """
    Naive implementation that does a network call.
    This is intentionally fragile for the demo (network disabled in sandbox).
    """
    import urllib.request

    resp = urllib.request.urlopen("https://example.com")
    return resp.getcode()


def add(a, b):
    # Fixed: this now correctly adds instead of subtracting
    return a + b