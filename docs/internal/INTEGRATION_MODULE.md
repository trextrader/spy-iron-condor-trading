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


---

## Repository Sync Addendum (2026-01-24)

This document is part of the synchronized documentation set. The authoritative engineering spec and audit references are:

- `docs/INTEGRATION_PLAN_MASTER.md`
- `docs/INTERFACE_CATALOG.md`

Key alignment requirements:
1. Feature schema selection by **name** (V2.2) only; no CSV order dependence.
2. Dataset column order differs across years; schema validation must be strict.
3. Model config metadata (layers/heads/input_dim) must match deployed checkpoints.

If this document conflicts with the master spec, the master spec governs implementation.
