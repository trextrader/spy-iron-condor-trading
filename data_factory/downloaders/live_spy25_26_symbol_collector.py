import asyncio
from alpaca.data.live import OptionDataStream

# ----------------------------------------------------------
# Alpaca API Keys
# ----------------------------------------------------------
API_KEY = "PKWCQL536DJKE7EJCP5OEETWE2"
API_SECRET = "8PK6xfXx13Hqna2ryjHyQCMAf6D1zj6kNGE96CjnzmKM"

# ----------------------------------------------------------
# Storage for discovered symbols
# ----------------------------------------------------------
all_symbols = set()

# ----------------------------------------------------------
# Callback: runs every time Alpaca sends an option update
# ----------------------------------------------------------
async def on_option_update(data):
    sym = data.symbol

    # Only keep SPY 2025 and 2026 expirations
    if sym.startswith("SPY25") or sym.startswith("SPY26"):
        if sym not in all_symbols:
            all_symbols.add(sym)
            print(f"Discovered: {sym}  (total={len(all_symbols)})")

            # Save updated list to file
            with open("spy_symbols_2025_2026.txt", "w") as f:
                f.write("\n".join(sorted(all_symbols)))

# ----------------------------------------------------------
# Main async runner
# ----------------------------------------------------------
async def main():
    stream = OptionDataStream(API_KEY, API_SECRET)

    # Subscribe to ALL SPY option updates
    # Alpaca uses OCC root "SPY" for all SPY options
    stream.subscribe_option_quotes("SPY", on_option_update)
    stream.subscribe_option_trades("SPY", on_option_update)
    stream.subscribe_option_greeks("SPY", on_option_update)

    print("Listening for SPY25/SPY26 option symbols...")
    print("Press CTRL+C to stop.")

    await stream.run()

# ----------------------------------------------------------
# Entry point
# ----------------------------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")
        print(f"Saved {len(all_symbols)} symbols to spy_symbols_2025_2026.txt")
