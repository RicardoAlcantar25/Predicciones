from bs4 import BeautifulSoup

with open("scratch/real_ticker_page.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, 'html.parser')

links = soup.find_all('a')
print(f"Total links: {len(links)}")
for idx, link in enumerate(links[50:106]):
    href = link.get('href')
    text = link.text.strip()
    print(f"Link {idx+51}: href={href} | text={text}")
