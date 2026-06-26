import requests
from bs4 import BeautifulSoup

url = "https://www.transfermarkt.com/argentinien_frankreich/index/spielbericht/3976352"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive"
}

try:
    resp = requests.get(url, headers=headers, timeout=10)
    print(f"Status Code: {resp.status_code}")
    print(f"Content length: {len(resp.text)}")
    soup = BeautifulSoup(resp.text, 'html.parser')
    title = soup.find('title')
    print(f"Page Title: {title.text if title else 'No title'}")
    
    # Check if we can find starting players
    players = soup.select(".lineup-coords .spieler-name, .starting-lineup .spieler-name")
    print(f"Found players via standard selectors: {len(players)}")
    if players:
        for idx, p in enumerate(players[:5]):
            print(f"  {idx+1}: {p.text.strip()}")
            
except Exception as e:
    print(f"Error: {e}")
