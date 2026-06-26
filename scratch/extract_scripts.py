import requests
from bs4 import BeautifulSoup

url = "https://www.transfermarkt.es/ticker/begegnung/live/4776650"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
}

resp = requests.get(url, headers=headers)
soup = BeautifulSoup(resp.text, 'html.parser')

with open("scratch/inline_scripts.txt", "w", encoding="utf-8") as f:
    for i, s in enumerate(soup.find_all('script')):
        if not s.get('src'):
            f.write(f"\n\n=== SCRIPT {i} ({s.get('type', 'no-type')}) ===\n")
            f.write(s.text.strip())

print("Inline scripts written to scratch/inline_scripts.txt")
