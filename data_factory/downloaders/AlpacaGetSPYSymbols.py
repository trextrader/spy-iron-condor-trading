import requests
import time
import json
import sys

API_KEY = "YOUR_KEY"
API_SECRET = "YOUR_SECRET"

BASE = "https://data.alpaca.markets/v1beta1/options/snapshots/SPY"

HEADERS = {
    "PKWCQL536DJKE7EJCP5OEETWE2": API_KEY,
    "8PK6xfXx13Hqna2ryjHyQCMAf6D1zj6kNGE96CjnzmKM": API_SECRET,
    "accept": "application/json"
}

# ============================================================
# Utility: Detect HTML (Alpaca returns HTML on 403/401/invalid)
# ============================================================
def looks_like_html(text):
    if not text:
        return False
    t = text.strip().lower()
    return t.startswith("<!doctype html") or t.startswith("<html")

# ============================================================
# Rate‑limit safe GET with:
# - OPRA→indicative fallback
# - HTML detection
# - JSON decode protection
# - exponential backoff
# - verbose console logging
# ============================================================
def safe_get(url, params=None, max_retries=5):
    for attempt in range(max_retries):
        print(f"\n→ GET {url}")
        print(f"  params = {params}")

        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=10)
        except Exception as e:
            print(f"❌ Network error: {e}")
            time.sleep(2 ** attempt)
            continue

        print(f"  HTTP {r.status_code}")
        print(f"  Raw response preview:\n{r.text[:300]}\n")

        # Handle rate limit
        if r.status_code == 429:
            wait = 2 ** attempt
            print(f"⚠️ 429 Too Many Requests — waiting {wait}s...")
            time.sleep(wait)
            continue

        # Handle server errors
        if r.status_code >= 500:
            wait = 2 ** attempt
            print(f"⚠️ Server error {r.status_code} — retrying in {wait}s...")
            time.sleep(wait)
            continue

        # Handle HTML (403, 401, invalid OPRA feed)
        if looks_like_html(r.text):
            print("⚠️ HTML detected — likely OPRA forbidden or invalid endpoint.")
            return {"html_error": True, "status": r.status_code, "body": r.text}

        # Try JSON decode
        try:
            return r.json()
        except json.JSONDecodeError:
            print("❌ JSON decode failed — raw body above.")
            return {"json_error": True, "status": r.status_code, "body": r.text}

    print("❌ Max retries exceeded.")
    return None

# ============================================================
# Fetch SPY symbol universe with OPRA→indicative fallback
# ============================================================
def get_symbol_universe():
    print("\n=== Fetching full SPY symbol universe ===\n")

    # First try OPRA (will fail on free plan)
    params = {"limit": 1000, "feed": "opra"}
    print("Trying OPRA feed first...")

    data = safe_get(BASE, params=params)

    # If OPRA fails or returns HTML → fallback to indicative
    if data is None or "html_error" in data or "json_error" in data:
        print("\n⚠️ OPRA feed unavailable — switching to indicative feed.\n")
        params["feed"] = "indicative"
        data = safe_get(BASE, params=params)

    if data is None:
        print("❌ Could not retrieve symbol universe.")
        sys.exit(1)

    # Validate JSON structure
    if "snapshots" not in data:
        print("❌ Unexpected JSON structure:", data)
        sys.exit(1)

    symbols = list(data["snapshots"].keys())
    print(f"\n✔ Retrieved {len(symbols)} SPY option symbols.")
    return symbols

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    symbols = get_symbol_universe()

    print("\n=== SYMBOL LIST (first 20) ===")
    for s in symbols[:20]:
        print(" ", s)

    print(f"\nTotal symbols: {len(symbols)}")
    print("\nDone.")
