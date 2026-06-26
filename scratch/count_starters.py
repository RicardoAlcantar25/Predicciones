import requests
from bs4 import BeautifulSoup

url = "https://www.transfermarkt.com/argentinien_frankreich/index/spielbericht/3976352"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

resp = requests.get(url, headers=headers)
soup = BeautifulSoup(resp.text, 'html.parser')

starters = soup.select(".formation-number-name a")
print(f"Total starters found: {len(starters)}")
for i, p in enumerate(starters):
    print(f"  {i+1}: {p.text.strip()} ({p.get('href')})")
