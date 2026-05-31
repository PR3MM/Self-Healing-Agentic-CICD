from unittest.mock import patch, MagicMock
from main import fetch_status, add


def test_add():
    assert add(2, 3, 0) == 5


def test_fetch_status():
    # Mock the HTTP call so this test works in any environment
    # (sandbox with no network, CI, or local development)
    mock_response = MagicMock()
    mock_response.status_code = 200
    with patch("requests.get", return_value=mock_response):
        assert fetch_status("https://example.com") == 200
