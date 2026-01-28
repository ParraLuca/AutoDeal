"""Debug scraper to see what's being extracted"""
from playwright.sync_api import sync_playwright
import re
import json

def clean(s):
    if not s: return ""
    s = re.sub(r'\s+', ' ', str(s)).strip()
    return s

def wait_dom_stable(page, ms=350):
    page.wait_for_timeout(ms)

def get_visible_tab_ids(page):
    tabs = []
    loc = page.get_by_role("tab")
    try:
        count = loc.count()
    except Exception:
        count = 0
    for i in range(count):
        try:
            el = loc.nth(i)
            name = clean(el.inner_text()) or clean(el.get_attribute("aria-label") or "")
            controls = el.get_attribute("aria-controls") or ""
            if name and controls:
                tabs.append((name, controls))
        except Exception:
            continue
    return tabs

def extract_active_tab_content(page):
    pairs, values = [], []
    label_nodes = page.locator("div.CarAttributesTabs-itemLabel")
    try:
        label_count = label_nodes.count()
    except Exception:
        label_count = 0
        
    print(f"DEBUG: Found {label_count} label nodes")
        
    for i in range(label_count):
        try:
            label = clean(label_nodes.nth(i).inner_text())
            value_node = label_nodes.nth(i).locator(
                "xpath=following-sibling::div[contains(@class,'CarAttributesTabs-itemValue')]"
            ).first
            value = clean(value_node.inner_text()) if value_node else ""
            if label and value:
                pairs.append({"label": label, "value": value})
                print(f"  Found: {label} = {value}")
        except Exception as e:
            print(f"  Error extracting pair {i}: {e}")
            continue

    return {"pairs": pairs, "values": values}

url = "https://www.2ememain.be/v/autos/volvo/m2360164083-volvo-s60-2-0d-d3"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)  # Not headless so we can see
    ctx = browser.new_context(
        viewport={"width": 1440, "height": 900},
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        locale="fr-FR",
    )
    page = ctx.new_page()
    page.set_default_timeout(8000)
    
    print(f"Navigating to {url}")
    page.goto(url, wait_until="domcontentloaded")
    wait_dom_stable(page, 1000)
    
    # Try to accept cookies
    try:
        cookie_btn = page.locator("button:has-text('Accepter'), button:has-text('Tout accepter')")
        if cookie_btn.count() > 0:
            cookie_btn.first.click(timeout=2000)
            wait_dom_stable(page, 500)
            print("Accepted cookies")
    except:
        print("No cookie banner or couldn't click")
    
    # Check for tabs
    tabs = get_visible_tab_ids(page)
    print(f"\nFound {len(tabs)} tabs: {tabs}")
    
    # Try clicking each tab
    for tab_name, _ in tabs:
        print(f"\nClicking tab: {tab_name}")
        try:
            page.get_by_role("tab", name=re.compile(rf"^{re.escape(tab_name)}$", re.I)).first.click(timeout=1500)
            wait_dom_stable(page, 500)
            content = extract_active_tab_content(page)
            print(f"Extracted {len(content['pairs'])} pairs from this tab")
        except Exception as e:
            print(f"Error clicking tab: {e}")
    
    # Also try the main extraction without clicking
    print("\n\nTrying extraction without clicking tabs:")
    content = extract_active_tab_content(page)
    print(f"Found {len(content['pairs'])} pairs total")
    for p in content['pairs']:
        print(f"  {p['label']}: {p['value']}")
    
    print("\nPress Enter to close browser...")
    input()
    
    ctx.close()
    browser.close()
