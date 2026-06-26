from bs4 import BeautifulSoup

with open("scratch/spielbericht_4776631.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

print("--- EXTRACCION DE ALINEACIONES DE SPIELBERICHT ---")

# Encontrar todos los divs con clase 'aufstellung-vereinsseite' que no tengan 'aufstellung-ersatzbank-box'
lineup_containers = []
for div in soup.find_all("div", class_="aufstellung-vereinsseite"):
    # Verificar si es de suplentes
    classes = div.get("class", [])
    if "aufstellung-ersatzbank-box" not in classes:
        lineup_containers.append(div)

print(f"Contenedores de alineación titular encontrados: {len(lineup_containers)}")

if len(lineup_containers) >= 2:
    home_div = lineup_containers[0]
    away_div = lineup_containers[1]
    
    home_players = [a.text.strip() for a in home_div.select("a[href*='/profil/spieler/']")]
    away_players = [a.text.strip() for a in away_div.select("a[href*='/profil/spieler/']")]
    
    print("\nLocal Titulares:", len(home_players), home_players)
    print("Visitante Titulares:", len(away_players), away_players)
else:
    print("No se encontraron suficientes contenedores de titulares.")
