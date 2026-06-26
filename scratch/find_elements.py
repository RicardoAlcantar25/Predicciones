import requests
from bs4 import BeautifulSoup
import re

url = "https://www.transfermarkt.com/argentinien_frankreich/index/spielbericht/3976352"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive"
}

resp = requests.get(url, headers=headers)
soup = BeautifulSoup(resp.text, 'html.parser')

print("=== Search for player tooltip links ===")
player_links = soup.select("a[href*='/profil/spieler/']")
print(f"Total player links: {len(player_links)}")

# Show unique classes of player links
classes = set()
for link in player_links:
    cls = link.get('class')
    if cls:
        classes.update(cls)
print(f"Player link classes: {classes}")

# Find table classes
tables = soup.find_all('table')
print(f"Total tables: {len(tables)}")
for i, table in enumerate(tables):
    print(f"Table {i+1} class: {table.get('class')} | id: {table.get('id')}")

# Let's inspect some of the player links and their parents
print("\n=== Sample player links and parents ===")
for link in player_links[:10]:
    parent = link.parent
    grandparent = parent.parent if parent else None
    print(f"Player: {link.text.strip()} | Link: {link.get('href')}")
    print(f"  Parent tag: {parent.name} | class: {parent.get('class')}")
    if grandparent:
        print(f"  Grandparent tag: {grandparent.name} | class: {grandparent.get('class')}")
