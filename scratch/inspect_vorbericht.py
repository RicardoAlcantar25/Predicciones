import requests
from bs4 import BeautifulSoup

url = "https://www.transfermarkt.es/suiza-canada/vorbericht/spielbericht/4776649"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
}

resp = requests.get(url, headers=headers)
soup = BeautifulSoup(resp.text, 'html.parser')

print(f"Title: {soup.title.text.strip() if soup.title else 'None'}")
print("--- HEADINGS ---")
for h in soup.find_all(['h1', 'h2', 'h3']):
    print(f"{h.name}: {h.text.strip()}")

print("--- ALL PLAYERS ---")
for a in soup.find_all('a'):
    href = a.get('href', '')
    if 'spieler' in href:
        print(f"Player: {a.text.strip()} -> {href}")
