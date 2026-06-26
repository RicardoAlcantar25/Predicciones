with open("scratch/4776631.html", "r", encoding="utf-8") as f:
    html = f.read()

import re
print("Links con spieler:", len(re.findall(r'/spieler/', html)))
print("Links con profil:", len(re.findall(r'/profil/', html)))
print("Tablas en html:", len(re.findall(r'<table', html)))
print("Divs en html:", len(re.findall(r'<div', html)))

# Imprimir las secciones que parezcan interesantes (e.g. que tengan alineaciones o aufstellung)
for m in re.finditer(r'(?i)aufstellung|alineaci|line-up|lineup', html):
    start = max(0, m.start() - 100)
    end = min(len(html), m.end() + 100)
    print("MATCH ALINEACION:", html[start:end])
