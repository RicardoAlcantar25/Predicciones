import requests

url = "https://www.transfermarkt.es/spielbericht/index/spielbericht/4776631"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.google.com/",
    "Upgrade-Insecure-Requests": "1"
}

try:
    print(f"Haciendo request a Spielbericht: {url}")
    resp = requests.get(url, headers=headers, timeout=10)
    print(f"Status Code: {resp.status_code}")
    print(f"Content-Length: {len(resp.text)}")
    with open("scratch/spielbericht_4776631.html", "w", encoding="utf-8") as f:
        f.write(resp.text)
    print("Guardado en scratch/spielbericht_4776631.html")
except Exception as e:
    print(f"Error: {e}")
