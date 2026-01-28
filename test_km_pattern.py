import re

def to_int(s):
    if not s: return None
    try:
        s = str(s).replace(' ', '').replace('.', '').replace(',', '')
        s = re.sub(r'[^\d]', '', s)
        return int(s) if s else None
    except:
        return None

text = "volvo s60 d3 2,0 diesel 5 cilinder euro 5 05/2013 260000 km tesrit mogelijk"

km_patterns = [
    r'(\d[\d\s.,]*)\s*km',  # "260000 km" or "260.000 km"
    r'kilometerstand\s*[:\s]*(\d[\d\s.,]*)',
    r'kilométrage\s*[:\s]*(\d[\d\s.,]*)',
]

for pattern in km_patterns:
    m = re.search(pattern, text, re.I)
    if m:
        print(f"Pattern matched: {pattern}")
        print(f"Match group: '{m.group(1)}'")
        km_str = m.group(1).replace('.', '').replace(',', '').replace(' ', '')
        print(f"After cleanup: '{km_str}'")
        km_val = to_int(km_str)
        print(f"to_int result: {km_val}")
        if km_val and 1000 <= km_val <= 1000000:
            print(f"FOUND: {km_val}")
            break
        else:
            print(f"Failed range check: {km_val}")
