"""
test_scraper.py — Test reproducible para validar la lógica del módulo de web scraping.
"""

import sys
import os

# Add project directory to python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.scraper import scrape_match_details

def main():
    print("=== INICIANDO VALIDACIÓN DEL SCRAPER ===")
    
    # URL de prueba del live ticker (Alemania vs Costa de Marfil)
    url_test = "https://www.transfermarkt.es/ticker/begegnung/live/4776631"
    
    # 1. Probar con Alemania vs Costa de Marfil
    print("\n[1/3] Probando con Alemania vs Costa de Marfil (Live Ticker)...")
    data = scrape_match_details(url_test, "Germany", "Ivory Coast")
    
    print("\n--- Resultados Extraídos ---")
    print(f"Árbitro Central: {data['referee']} (Promedio Tarjetas: {data['referee_cards_avg']} amarillas/juego)")
    print(f"Origen de los datos: {data['source'].upper()}")
    
    print(f"\nAlineación Titular Germany ({len(data['home_lineup'])} jugadores):")
    print(", ".join(data['home_lineup']))
    print(f"Valor Real en Cancha (Starting 11): {data['starting_val_home']:.2f} Mde")
    
    print(f"\nAlineación Titular Ivory Coast ({len(data['away_lineup'])} jugadores):")
    print(", ".join(data['away_lineup']))
    print(f"Valor Real en Cancha (Starting 11): {data['starting_val_away']:.2f} Mde")
    
    # Validaciones obligatorias
    has_havertz = any("Havertz" in p or "Musiala" in p for p in data['home_lineup'])
    has_kessie = any("Kessi" in p or "Fofana" in p for p in data['away_lineup'])
    print(f"\n--- Comprobación de Jugadores Clave ---")
    print(f"Havertz/Musiala detectado en la alineación local (Germany): {has_havertz}")
    print(f"Kessié/Fofana detectado en la alineación visitante (Ivory Coast): {has_kessie}")
    
    if not has_havertz or not has_kessie:
        raise ValueError("FALLO DE VALIDACIÓN: No se pudieron extraer 'Havertz' y 'Kessié' de las alineaciones reales!")
    
    # 2. Probar con equipos fuera de la base de datos mock (ej: Senegal vs Australia)
    print("\n[2/3] Probando con Senegal vs Australia (Deberían generarse fallbacks numéricos)...")
    # Usamos una URL no válida para forzar fallback mock
    data_fallback = scrape_match_details("https://www.transfermarkt.es/ticker/begegnung/live/999999", "Senegal", "Australia")
    print(f"Valor Real en Cancha Senegal: {data_fallback['starting_val_home']:.2f} Mde")
    print(f"Valor Real en Cancha Australia: {data_fallback['starting_val_away']:.2f} Mde")
    print(f"Árbitro: {data_fallback['referee']} ({data_fallback['referee_cards_avg']})")
    
    print("\n[3/3] Validación completada con éxito.")

if __name__ == "__main__":
    main()

