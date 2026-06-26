import requests
import re
from bs4 import BeautifulSoup

url = "https://www.transfermarkt.es/ticker/begegnung/live/4776649"
match_id = None
id_match = re.search(r'/([0-9]+)(?:$|/|\?)', url)
if id_match:
    match_id = id_match.group(1)

target_url = f"https://www.transfermarkt.es/spielbericht/index/spielbericht/{match_id}"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.google.com/",
    "Upgrade-Insecure-Requests": "1"
}

print(f"Target URL: {target_url}")
try:
    resp = requests.get(target_url, headers=headers, timeout=10)
    print(f"Status Code: {resp.status_code}")
    print(f"Content Length: {len(resp.text)}")
    if resp.status_code == 200:
        soup = BeautifulSoup(resp.text, 'html.parser')
        ref_links = soup.select("a[href*='/schiedsrichter/']")
        print(f"Ref links found: {len(ref_links)}")
        if ref_links:
            print(f"Referee name: {ref_links[0].text.strip()}")
        
        # Check lineups containers
        containers = soup.find_all("div", class_="aufstellung-vereinsseite")
        print(f"Alineaciones containers: {len(containers)}")
        for i, div in enumerate(containers):
            player_links = div.select("a[href*='/profil/spieler/']")
            print(f"  Container {i} players: {len(player_links)}")
    else:
        print(resp.text[:500])
except Exception as e:
    print(f"Exception: {e}")
