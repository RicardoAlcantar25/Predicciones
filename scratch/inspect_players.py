from bs4 import BeautifulSoup
import re

with open("scratch/4776631.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

print("--- ANALIZANDO ESTRUCTURA DE ALINEACIONES ---")

# 1. Busquemos si hay contenedores con clases que tengan "aufstellung", "lineup", "viewport", "formation"
containers = soup.find_all(class_=re.compile(r'(?i)aufstellung|lineup|viewport|formation'))
for c in containers:
    c_class = c.get('class')
    c_id = c.get('id')
    # Ver cuántos links de jugadores tiene dentro
    spieler_links = c.select("a[href*='/profil/spieler/']")
    if len(spieler_links) > 0:
        print(f"Contenedor: tag={c.name}, id={c_id}, class={c_class}, links={len(spieler_links)}")

# 2. Imprimir todos los links de jugadores y sus textos para ver dónde están
all_spieler = soup.select("a[href*='/profil/spieler/']")
print(f"\nTotal links de jugadores en toda la página: {len(all_spieler)}")
for i, s in enumerate(all_spieler[:60]):
    print(f"[{i}] Text: '{s.text.strip()}', Href: '{s.get('href')}'")
