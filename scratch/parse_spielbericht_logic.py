from bs4 import BeautifulSoup
import re

with open("scratch/spielbericht_4776631.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

print("--- DETECTANDO ARBITRO EN SPIELBERICHT ---")
# En Spielbericht el árbitro suele estar dentro de un enlace con /schiedsrichter/
ref_link = soup.select_one("a[href*='/schiedsrichter/']")
if ref_link:
    print("Árbitro detectado:", ref_link.text.strip())
else:
    print("Árbitro no detectado por selector directo.")

print("\n--- DETECTANDO JUGADORES TITULARES EN SPIELBERICHT ---")
# En un Spielbericht, las alineaciones de titulares están en dos cajas (Home y Away)
# Normalmente tienen clases como '.aufstellung-box' o tablas dentro de las alineaciones.
# Busquemos los divs con la clase 'aufstellung-box'
boxes = soup.select(".aufstellung-box")
print(f"Cantidad de cajas de alineación (.aufstellung-box): {len(boxes)}")

if len(boxes) >= 2:
    for i, box in enumerate(boxes[:2]):
        team_name = "Local" if i == 0 else "Visitante"
        print(f"\n--- {team_name} ---")
        # En la caja, los titulares están en la tabla principal antes de los suplentes (Ersatzbank)
        # Los suplentes suelen estar en una sección de 'Ersatzbank' o en una tabla específica.
        # Vamos a ver si hay enlaces de jugadores en esta caja.
        players = []
        # Busquemos todas las filas (tr) de la tabla de la alineación titular
        # En Transfermarkt, los titulares están listados en una tabla donde cada fila tiene una clase específica
        # o los suplentes se listan en una tabla que tiene la clase 'ersatzbank' o similar.
        # Busquemos todos los a[href*='/profil/spieler/'] que NO estén dentro de un contenedor o fila de suplentes
        # Los suplentes están dentro de una tabla o tr de suplentes.
        # Veamos si la estructura tiene una tabla de titulares y otra de suplentes.
        # A veces, hay un tr de cabecera 'Suplentes' o 'Ersatzbank'.
        # Imprimamos todos los links de jugadores dentro del box y veamos si podemos discriminar.
        all_links = box.select("a[href*='/profil/spieler/']")
        print("Total jugadores en la caja:", len(all_links))
        
        # Veamos si hay un tr con clase 'ersatzbank' o texto 'Suplentes' / 'Ersatzbank'
        # e imprimamos los textos de las filas
        rows = box.find_all("tr")
        for r_idx, row in enumerate(rows):
            text = row.text.strip()
            if "ersatz" in text.lower() or "suplentes" in text.lower() or "substitution" in text.lower():
                print(f"Fila suplentes detectada en el índice {r_idx}: '{text[:50]}'")
                
        # Por lo general, los primeros 11 jugadores en la caja de alineación de cada equipo son los TITULARES.
        # Vamos a comprobar si es así.
        for p in all_links[:11]:
            players.append(p.text.strip())
        print("Primeros 11 jugadores en la caja:", players)
else:
    # Si no hay cajas de alineación, veamos cómo encontrar los jugadores titulares en el HTML
    print("No se encontraron cajas .aufstellung-box")
