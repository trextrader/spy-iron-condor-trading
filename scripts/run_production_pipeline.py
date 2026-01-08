
import subprocess
import sys
import os
import time

def run_script(script_path):
    """Runs a python script and waits for it to complete."""
    print(f"\n{'='*60}")
    print(f"Starting: {script_path}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run([sys.executable, "-u", script_path], check=False)
    duration = time.time() - start_time
    
    if result.returncode != 0:
        print(f"Error: {script_path} failed with exit code {result.returncode}")
        # We might want to stop here?
        # For pipeline, yes, usually we stop.
        return False
    else:
        print(f"Success: {script_path} completed in {duration:.1f}s")
        return True

def main():
    print("=" * 70)
    print("SPY OPTION TRADER - PRODUCTION PIPELINE (PHASE 5)")
    print("=" * 70)
    print("This script will execute the full data generation workflow:")
    print("1. Download Top 100 Liquid Options (ATM) from IVolatility")
    print("2. Fetch Min-by-Min Bars from Alpaca for those options")
    print("3. Merge and Resample to final M1/M5/M15 datasets")
    print("4. Run Full Verification Backtest")
    print("\nEstimated Runtime: ~8-10 Hours (due to API rate limits)")
    print("You can leave this running overnight.")
    print("=" * 70)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Download IVolatility Data
    # Note: Requires IVolatility API Key
    ivol_script = os.path.join(base_dir, "data_factory", "download_ivolatility_options.py")
    if not run_script(ivol_script):
        print("Pipeline halted at Step 1.")
        return

    # 2. Download Alpaca Data
    # Note: Requires Alpaca API Keys
    alpaca_script = os.path.join(base_dir, "data_factory", "download_alpaca_matched.py")
    if not run_script(alpaca_script):
         print("Pipeline halted at Step 2.")
         return

    # 3. Merge and Resample
    merge_script = os.path.join(base_dir, "data_factory", "merge_intraday_with_greeks.py")
    if not run_script(merge_script):
        print("Pipeline halted at Step 3.")
        return

    # 4. Run Backtest
    backtest_script = os.path.join(base_dir, "scripts", "run_full_backtest.py")
    if not run_script(backtest_script):
         print("Pipeline halted at Step 4.")
         return

    print("\n" + "="*70)
    print("PIPELINE COMPLETE SUCCESS")
    print("="*70)

if __name__ == "__main__":
    main()
