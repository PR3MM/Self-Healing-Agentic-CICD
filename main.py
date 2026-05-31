import requests

def add(a, b,c,d):
    return a + b




def fetch_status(url):
    response = requests.get(url)
    
    return response.status_code