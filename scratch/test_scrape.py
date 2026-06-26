import sys
import requests
from bs4 import BeautifulSoup
import re

# Set stdout to use utf-8
sys.stdout.reconfigure(encoding='utf-8')

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}

urls = {
    "index": "https://www.transfermarkt.es/spielbericht/index/spielbericht/4776650",
    "aufstellung": "https://www.transfermarkt.es/spielbericht/aufstellung/spielbericht/4776650",
    "live": "https://www.transfermarkt.es/ticker/begegnung/live/4776650"
}

for name, url in urls.items():
    print(f"--- Fetching {name}: {url} ---")
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        print(f"Status: {resp.status_code}, Length: {len(resp.text)}")
        text = resp.text
        for term in ["Dzeko", "Demirović", "Demirovic", "Jesús Valenzuela", "Valenzuela"]:
            count = text.lower().count(term.lower())
            print(f"  Term '{term}': {count} occurrences")
            if count > 0:
                # find where it occurs
                matches = [m.start() for m in re.finditer(term.lower(), text.lower())]
                for m in matches[:2]:
                    snippet = text[max(0, m-50):min(len(text), m+50)]
                    print(f"    Snippet: {repr(snippet)}")
    except Exception as e:
        print(f"Error fetching {name}: {e}")
