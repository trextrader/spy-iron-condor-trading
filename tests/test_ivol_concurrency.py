import ivolatility as ivol
import concurrent.futures
import time
import pandas as pd

IVOL_API_KEY = "MFGkqVygN5NSgF2I"
ivol.setLoginParams(apiKey=IVOL_API_KEY)

def fetch_single(option_id):
    try:
        getOpts = ivol.setMethod('/equities/eod/single-stock-option-raw-iv')
        # Using a fixed date known to have data
        data = getOpts(optionId=option_id, from_='2025-06-30', to='2025-06-30')
        return data is not None
    except Exception as e:
        return False

def test_concurrency(workers=10):
    print(f"Testing with {workers} workers...")
    
    # Use IDs from previous knowledge (e.g. 1 to 50? No, IDs are large integers)
    # We need valid option IDs. I'll fetch a chain first.
    getOptsChain = ivol.setMethod('/equities/eod/option-series-on-date')
    chain = getOptsChain(symbol='SPY', date='2025-06-30', expFrom='2025-07-30', expTo='2025-08-30', callPut='C')
    
    ids = chain['optionId'].unique()[:50].tolist()
    print(f"Testing on {len(ids)} option IDs")
    
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(fetch_single, ids))
    
    duration = time.time() - start
    print(f"Duration: {duration:.2f}s")
    print(f"Rate: {len(ids)/duration:.2f} req/s")
    print(f"Successes: {sum(results)}")

if __name__ == "__main__":
    test_concurrency(1)
    test_concurrency(5)
    test_concurrency(10)
    test_concurrency(20)
