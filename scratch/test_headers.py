import requests
from bs4 import BeautifulSoup
import time

url = "https://www.transfermarkt.es/ticker/begegnung/live/477633"

headers_options = [
    # Option 1: Basic user agent
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    },
    # Option 2: Complete browser emulation
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://www.google.com/",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site",
        "Upgrade-Insecure-Requests": "1"
    },
    # Option 3: Another browser (Firefox)
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "es-ES,es;q=0.8,en-US;q=0.5,en;q=0.3",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://www.transfermarkt.es/",
        "Upgrade-Insecure-Requests": "1"
    }
]

for idx, headers in enumerate(headers_options):
    try:
        print(f"--- Testing Option {idx+1} ---")
        resp = requests.get(url, headers=headers, timeout=10)
        print(f"Status Code: {resp.status_code}")
        print(f"Content Length: {len(resp.text)}")
        soup = BeautifulSoup(resp.text, 'html.parser')
        title = soup.find('title')
        print(f"Title: {title.text if title else 'No Title'}")
        
        # Check if error or main content
        if "error" not in (title.text if title else "").lower() and resp.status_code == 200:
            print("Success! Got actual content.")
            break
    except Exception as e:
        print(f"Error: {e}")
    time.sleep(2)
