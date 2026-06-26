import requests

urls = [
    "https://www.transfermarkt.es/ticker/begegnung/live/477633",
    "https://www.transfermarkt.com/ticker/begegnung/live/477633",
    "https://www.transfermarkt.de/ticker/begegnung/live/477633",
    "https://www.transfermarkt.co.uk/ticker/begegnung/live/477633"
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive"
}

for url in urls:
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        print(f"URL: {url} -> Status: {resp.status_code}, Length: {len(resp.text)}")
        if resp.status_code == 200 and "error" not in url.lower():
            # Save a snippet
            with open(f"scratch/ticker_{url.split('.')[-1][:2]}.html", "w", encoding="utf-8") as f:
                f.write(resp.text)
    except Exception as e:
        print(f"Error for {url}: {e}")
