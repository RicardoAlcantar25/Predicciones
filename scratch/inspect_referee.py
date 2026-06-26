from bs4 import BeautifulSoup

with open("scratch/spielbericht_4776631.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

print("--- INSPECCIONANDO ARBITRO ---")
ref_links = soup.select("a[href*='/schiedsrichter/']")
for r in ref_links:
    print("Link:", r)
    # Ver los padres del link
    parent = r.parent
    print("  Parent:", parent.name, parent.get('class'), parent.text.strip())
    if parent.parent:
        print("  Grandparent:", parent.parent.name, parent.parent.get('class'))
