import requests
from bs4 import BeautifulSoup

url = "https://www.transfermarkt.es/ticker/begegnung/live/4897684"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive"
}

try:
    resp = requests.get(url, headers=headers, timeout=10)
    print(f"Status Code: {resp.status_code}")
    print(f"Content Length: {len(resp.text)}")
    
    with open("scratch/real_ticker_page.html", "w", encoding="utf-8") as f:
        f.write(resp.text)
        
    soup = BeautifulSoup(resp.text, 'html.parser')
    title = soup.find('title')
    print(f"Title: {title.text.strip() if title else 'No Title'}")
    
    # Let's search for referee
    print("\n--- Searching for referee text/links ---")
    for link in soup.select("a[href*='schiedsrichter']"):
        print(f"Referee link: {link.get('href')} | Text: {link.text.strip()}")
    for el in soup.find_all(text=True):
        if any(kw in el.lower() for kw in ["arbitro", "schiedsrichter", "referee"]):
            print(f"Referee text: {el.strip()}")
            
    # Let's search for player links
    print("\n--- Searching for player links ---")
    player_links = soup.select("a[href*='/profil/spieler/']")
    print(f"Total player links: {len(player_links)}")
    for link in player_links[:10]:
        parent = link.parent
        grandparent = parent.parent if parent else None
        print(f"Player: {link.text.strip()} | Parent: {parent.name} ({parent.get('class')}) | Grandparent: {grandparent.name if grandparent else None} ({grandparent.get('class') if grandparent else None})")
        
except Exception as e:
    print(f"Error: {e}")
