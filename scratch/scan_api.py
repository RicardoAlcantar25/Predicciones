import requests

match_id = "4776650"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
}

urls = [
    f"https://www.transfermarkt.es/ticker/begegnung/live-data/{match_id}",
    f"https://www.transfermarkt.es/ticker/live-data/{match_id}",
    f"https://www.transfermarkt.es/ticker/begegnung/data/{match_id}",
    f"https://www.transfermarkt.es/ticker/data/{match_id}",
    f"https://www.transfermarkt.es/ticker/begegnung/live/{match_id}/data",
    f"https://www.transfermarkt.es/ticker/begegnung/live/{match_id}/json",
    f"https://www.transfermarkt.es/spielbericht/live-data/{match_id}",
]

for url in urls:
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        print(f"URL: {url} -> Status: {resp.status_code}, Length: {len(resp.text)}")
        if resp.status_code == 200 and ('Dzeko' in resp.text or 'Basic' in resp.text or 'dzeko' in resp.text.lower()):
            print("  ==> FOUND PLAYER DATA! <==")
            print(resp.text[:500])
    except Exception as e:
        print(f"URL: {url} -> Error: {e}")
