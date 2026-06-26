from bs4 import BeautifulSoup
import re

with open("scratch/spielbericht_4776631.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

print("--- CLASES CON RE/AUFSTELLUNG/SPIELER/LINEUP ---")
for el in soup.find_all(class_=re.compile(r'(?i)aufstellung|lineup|spieler|team|box|columns')):
    classes = el.get('class')
    tag = el.name
    # Buscar si tiene links de jugadores
    links = el.select("a[href*='/profil/spieler/']")
    if len(links) > 0:
        print(f"Tag: {tag}, classes={classes}, links={len(links)}, texto_inicio='{el.text.strip()[:60]}'")
