import requests

def add(a, b):
    return a + b
def fetch_status(url):
    response = requests.get(url)
    
    return response.status_code