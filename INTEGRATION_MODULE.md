# QTMF as a Python Module (for Gaussian)

This repo can be installed as an **editable Python module** so other projects (e.g. `gaussian-system`) can import:

```python
from qtmf import benchmark_and_size, TradeIntent
```

## Editable install

From the Python environment used by the caller:

```bash
python -m pip install -e /path/to/SPYOptionTrader_test
```

This keeps local backtesting, paper trading, and Alpaca demo testing intact, while allowing external callers to import the newest changes instantly.

## Public facade

- `qtmf.facade.benchmark_and_size(...)` is the stable integration entrypoint.
- Internal code can evolve without breaking callers.
