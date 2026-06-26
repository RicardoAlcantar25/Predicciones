import requests
from bs4 import BeautifulSoup

url = "https://www.transfermarkt.es/live/index"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

try:
    resp = requests.get(url, headers=headers, timeout=10)
    print(f"Live games page status: {resp.status_code}")
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    # Search for links containing ticker/begegnung/live/
    live_links = soup.select("a[href*='ticker/begegnung/live/']")
    print(f"Found live ticker links: {len(live_links)}")
    for link in live_links[:5]:
        print(f"  Href: {link.get('href')} | Text: {link.text.strip()}")
        
except Exception as e:
    print(f"Error fetching live matches: {e}")
