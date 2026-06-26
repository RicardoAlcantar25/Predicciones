with open("scratch/4776631.html", "r", encoding="utf-8") as f:
    html = f.read()

import re
from bs4 import BeautifulSoup
soup = BeautifulSoup(html, 'html.parser')

print("--- BUSCANDO SCRIPTS CON DATOS ---")
for i, s in enumerate(soup.find_all("script")):
    text = s.text.strip()
    if len(text) > 500:
        print(f"Script [{i}] - longitud={len(text)} - inicio={text[:150]}")
        # buscar palabras clave
        for kw in ["benitez", "benítez", "gakpo", "alemania", "deutschland", "spieler"]:
            if kw in text.lower():
                print(f"  -> Contiene '{kw}'!")
