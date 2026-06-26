import requests
from bs4 import BeautifulSoup

url = "https://www.transfermarkt.es/spielbericht/index/spielbericht/4776650"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
}

resp = requests.get(url, headers=headers)
soup = BeautifulSoup(resp.text, 'html.parser')

print(f"Title: {soup.title.text.strip().encode('ascii', 'backslashreplace').decode('ascii')}")
print("--- HEADINGS ---")
for h in soup.find_all(['h1', 'h2', 'h3']):
    print(f"{h.name}: {h.text.strip().encode('ascii', 'backslashreplace').decode('ascii')}")

print("--- ALL LINKS CONTAINING SPIELER ---")
for a in soup.find_all('a'):
    href = a.get('href', '')
    if 'profil/spieler' in href:
        txt = a.text.strip().encode('ascii', 'backslashreplace').decode('ascii')
        hr = href.encode('ascii', 'backslashreplace').decode('ascii')
        print(f"Player link: {txt} -> {hr}")
