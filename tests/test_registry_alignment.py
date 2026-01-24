import subprocess
import sys
from pathlib import Path


def test_registry_alignment_script():
    root = Path('c:/SPYOptionTrader_test')
    script = root / 'scripts' / 'verify_registry_alignment.py'
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
