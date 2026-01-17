
import sys
import os
sys.path.insert(0, os.getcwd())

print("Attempting to import strategies.options_strategy...")
try:
    from strategies.options_strategy import OptionsStrategy
    print("✅ Success: OptionsStrategy imported.")
except ImportError as e:
    print(f"❌ Failed: {e}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\nAttempting to import intelligence.condor_brain...")
try:
    import intelligence.condor_brain as cb
    print(f"✅ Success: CondorBrain imported. HAS_PANDAS_TA={getattr(cb, 'HAS_PANDAS_TA', 'Unknown')}")
except ImportError as e:
    print(f"❌ Failed: {e}")
