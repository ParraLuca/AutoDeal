import requests
from bs4 import BeautifulSoup
import re

def check_market_size():
    # URL for Volvo S60 (search query)
    url = "https://www.2ememain.be/q/volvo+s60/"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept-Language": "fr-BE,fr;q=0.9,en-US;q=0.8,en;q=0.7"
    }
    
    print(f"Fetching {url}...")
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return

    soup = BeautifulSoup(r.text, "lxml")
    
    # Try to find the total result count
    # Usually in a span or div with specific class or text
    # e.g., "1.234 resultaten" or "1 234 résultats"
    
    total_text = ""
    # Look for common patterns
    for tag in soup.find_all(["span", "div"]):
        txt = tag.get_text(strip=True)
        if re.search(r"\d+\s+r[ée]sultats?", txt, re.I) or re.search(r"\d+\s+resultaten", txt, re.I):
            total_text = txt
            break
            
    try:
        print(f"Found total text: '{total_text}'")
    except UnicodeEncodeError:
        print(f"Found total text: '{total_text.encode('ascii', 'ignore').decode('ascii')}'")
    
    # Also count ads on first page
    ads = soup.find_all("a", href=re.compile(r"/v/autos/"))
    print(f"Number of ad links on first page: {len(ads)}")
    
    # Extract number from total_text
    match = re.search(r"(\d+)", total_text.replace(" ", "").replace(".", ""))
    if match:
        print(f"Estimated Market Size: {match.group(1)}")
    else:
        print("Could not parse total count.")

if __name__ == "__main__":
    check_market_size()
