import requests
import re

url = "https://www.transfermarkt.es/spielbericht/aufstellung/spielbericht/4776650"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
}

resp = requests.get(url, headers=headers)
html = resp.text

# Find where 'Formación inicial' appears and print 500 characters around it
for m in re.finditer(r'Formaci\xf3n inicial|Formacion inicial', html):
    start = max(0, m.start() - 100)
    end = min(len(html), m.end() + 600)
    print(f"--- MATCH AT {m.start()} ---")
    print(html[start:end].encode('ascii', 'backslashreplace').decode('ascii'))
