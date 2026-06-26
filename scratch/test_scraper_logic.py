import requests
from bs4 import BeautifulSoup
import re
import unicodedata

url = "https://www.transfermarkt.com/argentinien_frankreich/index/spielbericht/3976352"

# Real browser headers
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.google.com/",
    "Upgrade-Insecure-Requests": "1"
}

resp = requests.get(url, headers=headers)
soup = BeautifulSoup(resp.text, 'html.parser')

def clean_scraped_player_name(name: str) -> str:
    # Remove captain tag, e.g. " (C)" or " (c)" or "(C)"
    name = re.sub(r'\s*\([cC]\)\s*', ' ', name)
    # Remove numbers (e.g. shirt number "10 ")
    name = re.sub(r'^\d+\s+', '', name)
    name = re.sub(r'\s+\d+\s*$', '', name)
    # Normalize spaces
    name = name.strip()
    name = re.sub(r'\s+', ' ', name)
    return name

def is_substitute(el):
    parent = el.parent
    while parent:
        if parent.name in ['table', 'div'] and parent.get('class'):
            classes = parent.get('class')
            if isinstance(classes, str):
                classes = [classes]
            if any(c in ['ersatzbank', 'bench', 'substitutes', 'substitute'] for c in classes):
                return True
        parent = parent.parent
    return False

# 1. Extract starting players
# We'll try specific selectors first, then fall back to filtering player links
starters = []

# Selector 1: formation number name
elements = soup.select(".formation-number-name a")
if elements:
    print(f"Selector '.formation-number-name a' found {len(elements)} elements.")
    for el in elements:
        name = clean_scraped_player_name(el.text.strip())
        href = el.get('href')
        if name and (name, href) not in starters:
            starters.append((name, href))

# Selector 2: standard lineup name selectors
if len(starters) != 22:
    elements = soup.select(".lineup-coords .spieler-name, .starting-lineup .spieler-name")
    if elements:
        print(f"Selector '.lineup-coords .spieler-name' found {len(elements)} elements.")
        for el in elements:
            name = clean_scraped_player_name(el.text.strip())
            if name and name not in [s[0] for s in starters]:
                starters.append((name, None))

# Selector 3: general player links filtering
if len(starters) != 22:
    all_player_links = soup.select("a[href*='/profil/spieler/']")
    print(f"General player links filtering from {len(all_player_links)} links...")
    for el in all_player_links:
        if not is_substitute(el):
            name = clean_scraped_player_name(el.text.strip())
            href = el.get('href')
            if name and (name, href) not in starters:
                # Make sure we don't accidentally grab empty text links
                starters.append((name, href))

print(f"Total unique starters found: {len(starters)}")

# Split into home and away
if len(starters) >= 22:
    # First 11 are home, next 11 are away
    home_lineup = [s[0] for s in starters[:11]]
    away_lineup = [s[0] for s in starters[11:22]]
else:
    # Fallback to even split if we got less than 22 but some count
    half = len(starters) // 2
    home_lineup = [s[0] for s in starters[:half]]
    away_lineup = [s[0] for s in starters[half:2*half]]

print(f"Home lineup ({len(home_lineup)}): {home_lineup}")
print(f"Away lineup ({len(away_lineup)}): {away_lineup}")

# 2. Extract referee
referee_name = ""
ref_links = soup.select("a[href*='/schiedsrichter/']")
if ref_links:
    referee_name = ref_links[0].text.strip()
    print(f"Found referee via link: {referee_name}")

if not referee_name:
    # Search in text elements
    for element in soup.find_all(text=True):
        text = element.strip()
        if any(kw in text.lower() for kw in ["referee:", "arbitro:", "arbitre:", "schiedsrichter:"]):
            referee_name = text.split(":")[-1].strip()
            print(f"Found referee via text: {referee_name}")
            break
