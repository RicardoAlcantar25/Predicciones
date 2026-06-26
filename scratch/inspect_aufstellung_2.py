import requests
from bs4 import BeautifulSoup

url = "https://www.transfermarkt.es/spielbericht/aufstellung/spielbericht/4776650"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
}

resp = requests.get(url, headers=headers)
soup = BeautifulSoup(resp.text, 'html.parser')

print(f"Title: {soup.title.text.strip().encode('ascii', 'backslashreplace').decode('ascii')}")

# Print first 20 links to see their structure
print("--- FIRST 30 LINKS ---")
links = soup.find_all('a')
for a in links[:30]:
    href = a.get('href', '')
    print(f"Link: text={repr(a.text.strip())} -> href={repr(href)}")

print("--- TABLES ---")
for i, table in enumerate(soup.find_all('table')):
    print(f"Table {i} rows: {len(table.find_all('tr'))}")
    # Print first row text
    first_row = table.find('tr')
    if first_row:
        print(f"  First row: {repr(first_row.text.strip().encode('ascii', 'backslashreplace').decode('ascii'))}")
