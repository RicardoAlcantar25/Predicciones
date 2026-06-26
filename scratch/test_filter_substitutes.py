import requests
from bs4 import BeautifulSoup

url = "https://www.transfermarkt.com/argentinien_frankreich/index/spielbericht/3976352"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

resp = requests.get(url, headers=headers)
soup = BeautifulSoup(resp.text, 'html.parser')

def is_substitute(el):
    parent = el.parent
    while parent:
        if parent.name in ['table', 'div'] and parent.get('class'):
            classes = parent.get('class')
            # Normalize to list of strings
            if isinstance(classes, str):
                classes = [classes]
            if any(c in ['ersatzbank', 'bench', 'substitutes', 'substitute'] for c in classes):
                return True
        parent = parent.parent
    return False

# Get all player links
all_players = soup.select("a[href*='/profil/spieler/']")
print(f"Total player links: {len(all_players)}")

starters = []
for p in all_players:
    if not is_substitute(p):
        name = p.text.strip()
        href = p.get('href')
        # Avoid duplicates (sometimes player names appear in the tactical graphic and as links elsewhere)
        if name and (name, href) not in starters:
            starters.append((name, href))

print(f"Filtered starters found: {len(starters)}")
for i, (name, href) in enumerate(starters):
    print(f"  {i+1}: {name} ({href})")
