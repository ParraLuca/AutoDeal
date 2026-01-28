"""Test the fallback extraction on existing data"""
import sys
sys.path.insert(0, '.')

# Import the functions from the pipeline
from deuxiememain_pipeline import extract_from_text_fallback

# Test with actual data from the scraped file
title = "Volvo s60 2,0d. D3"
description = "Volvo s60 D3 2,0 diesel 5 cilinder Euro 5 05/2013 260000 km Tesrit mogelijk Motor 10/10 Koppeling 10/10 Versnellingsbak 10/10 Carrosserie 8/10 De wagen start en rijdt super goed Zo meenemen 4950€ oky Keurig voor verkoop + 250€ +32 489255417 Zaventem/Machelen"

result = extract_from_text_fallback(title, description)

print("Fallback extraction test:")
print(f"Title: {title}")
print(f"Description: {description[:100]}...")
print(f"\nExtracted data:")
for key, value in result.items():
    print(f"  {key}: {value}")

print("\n" + "="*50)
print("Testing on more examples:")
print("="*50)

examples = [
    ("Volvo S60 D4 Momentum", "kenmerken/opties : - rechtstreeks van eerste eigenaar 2011 150000km diesel"),
    ("Volvo s60 automaat stage1,bj 2008", "Volvo s60,bj 2008,235000km,2.4 tdi,automaat"),
    ("Volvo S60 2.0 B3 R-Design Navi Leder", "Voor meer informatie 2021 98691 km Benzine Automaat"),
]

for title, desc in examples:
    result = extract_from_text_fallback(title, desc)
    print(f"\nTitle: {title}")
    print(f"  -> Make: {result['make']}, Model: {result['model']}, Year: {result['year']}, KM: {result['mileage_km']}, Fuel: {result['fuel']}")
