from bs4 import BeautifulSoup

with open("scratch/real_ticker_page.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, 'html.parser')

print("=== Checking first 50 links ===")
links = soup.find_all('a')
print(f"Total links: {len(links)}")
for idx, link in enumerate(links[:50]):
    href = link.get('href')
    text = link.text.strip()
    if href or text:
        print(f"Link {idx+1}: href={href} | text={text}")
