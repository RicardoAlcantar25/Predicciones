with open("scratch/4776631.html", "r", encoding="utf-8") as f:
    html = f.read()

print("Longitud total:", len(html))
print("Alemania en html:", "Alemania" in html)
print("Alemania (case-insensitive):", "alemania" in html.lower())
print("Costa de Marfil en html:", "Costa de Marfil" in html)
print("Juan en html:", "Juan" in html)
print("Benitez en html:", "Benitez" in html)
print("Benítez en html:", "Benítez" in html)
print("árbitro en html:", "árbitro" in html.lower())
print("arbitro en html:", "arbitro" in html.lower())
print("schiedsrichter en html:", "schiedsrichter" in html.lower())

# Busquemos etiquetas que tengan schiedsrichter en href o class
import re
for m in re.finditer(r'(?i)schiedsrichter', html):
    start = max(0, m.start() - 100)
    end = min(len(html), m.end() + 100)
    print("MATCH SCHIEDSRICHTER:", html[start:end])
