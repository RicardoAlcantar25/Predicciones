with open("scratch/spielbericht_4776631.html", "r", encoding="utf-8") as f:
    html = f.read()

print("Longitud:", len(html))
print("Benítez en html:", "Benítez" in html)
print("Benitez en html:", "Benitez" in html)
print("Neuer en html:", "Neuer" in html)
print("Guessand en html:", "Guessand" in html)
print("Kessié en html:", "Kessié" in html)
print("Kessie en html:", "Kessie" in html)

# Extraer el árbitro
import re
from bs4 import BeautifulSoup
soup = BeautifulSoup(html, "html.parser")

ref_links = soup.select("a[href*='/schiedsrichter/']")
print("Enlaces a árbitro:", [r.text.strip() for r in ref_links])

# Buscar los jugadores de alineaciones
players = soup.select("a[href*='/profil/spieler/']")
print("Total enlaces de jugadores:", len(players))
print("Primeros 20 jugadores:", [p.text.strip() for p in players[:20]])
