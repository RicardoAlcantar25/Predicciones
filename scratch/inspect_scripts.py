import requests
from bs4 import BeautifulSoup

url = "https://www.transfermarkt.es/ticker/begegnung/live/4776650"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
}

resp = requests.get(url, headers=headers)
soup = BeautifulSoup(resp.text, 'html.parser')

print(f"Total script tags: {len(soup.find_all('script'))}")
for i, s in enumerate(soup.find_all('script')):
    src = s.get('src', '')
    type_ = s.get('type', '')
    text = s.text.strip()
    print(f"Script {i}: src={repr(src)}, type={repr(type_)}, len_text={len(text)}")
    if text and ('Dzeko' in text or 'Basic' in text or 'D\u017eeko' in text or '34397' in text):
        print(f"  ==> SCRIPT {i} HAS PLAYER DATA! <==")
        print(text[:1000])
