from main import fetch_status, add


def test_add():
    assert add(2, 3) == 5


def test_fetch_status():
    # Expect a 200 status; will fail in sandbox (network disabled) or if requests missing
    assert fetch_status() == 200
