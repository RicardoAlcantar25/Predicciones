from bs4 import BeautifulSoup

with open("scratch/spielbericht_4776631.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

box = soup.select_one(".aufstellung-box")
if box:
    print("Clase de la caja:", box.get('class'))
    # Busquemos todas las tablas dentro de la caja
    tables = box.select("table")
    print(f"Tablas dentro de .aufstellung-box: {len(tables)}")
    for j, t in enumerate(tables):
        print(f"Tabla [{j}] - class={t.get('class')} - rows={len(t.select('tr'))}")
        # Veamos los primeros links de jugadores
        spieler = t.select("a[href*='/profil/spieler/']")
        print(f"  -> Jugadores en esta tabla: {len(spieler)}")
        print(f"  -> Primeros 5 jugadores: {[s.text.strip() for s in spieler[:5]]}")
else:
    print("No se encontró aufstellung-box")
