from bs4 import BeautifulSoup

with open("scratch/spielbericht_4776631.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

containers = []
for div in soup.find_all("div", class_="aufstellung-vereinsseite"):
    classes = div.get("class", [])
    if "aufstellung-ersatzbank-box" not in classes:
        containers.append(div)

print("Contenedores encontrados:", len(containers))
for idx, c in enumerate(containers):
    links = c.select("a[href*='/profil/spieler/']")
    print(f"[{idx}] class={c.get('class')} | links={len(links)} | texto_inicio='{c.text.strip()[:100]}'")
