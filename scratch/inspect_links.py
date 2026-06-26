with open("scratch/4776631.html", "r", encoding="utf-8") as f:
    html = f.read()

import re
# Busquemos todos los links href de la página
links = re.findall(r'href="([^"]+)"', html)
for l in links:
    if "ticker" in l or "begegnung" in l or "aufstellung" in l or "lineup" in l:
        print("Link relevante:", l)
