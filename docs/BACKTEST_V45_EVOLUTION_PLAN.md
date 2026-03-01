# CondorNet v4.5 Backtest Evolution Plan
## Multi-Strategy Portfolio-Aware Policy Engine
**Framework**: Institutional Quant Engineering — Volatility Strategy Research Platform
**Base file**: `kaggle/condor_brain_backtest_v45.py` (evolved from v2.2)
**Status**: In progress — Entry block complete ✓

---

## Ordered Execution Sequence (Part XIII Priority)

> Do NOT optimize model and strategy simultaneously.
> Order: Make backtester truth-accurate → Generate dataset → Train.

---

## Phase 1 — StrategyRegistry ✓ (STUB COMPLETE)

Entry block already contains single-strategy iron condor with registry-compatible structure.
Full registry to be implemented as standalone module.

### Tasks
- [x] Entry block rewritten — multi-position, capital-gated (`condor_brain_backtest_v45.py`)
- [x] OPEN event rows emitted to `closed_trades` for trajectory export
- [x] 12-counter gate audit + `_entry_audit_log()` to `bar_trace.jsonl`
- [x] `_deployed_margin()` + `_can_allocate()` capital helpers inline
- [ ] Create `core/strategy_registry.py` — `StrategyRegistry` class
      ```python
      class StrategyRegistry:
          def register(self, name, cls): ...
          def get(self, name): ...
          def all(self): ...   # → (name, cls) pairs
      ```
- [ ] Define base `OptionsStrategy` interface:
      ```python
      class OptionsStrategy:
          def build_structure(self, chain, params) -> dict | None
          def validate_structure(self, legs) -> bool
          def estimate_risk(self, legs, credit) -> dict   # margin, max_loss, pop
      ```
- [ ] Register `IronCondorStrategy` as first concrete implementation
- [ ] Wire registry into `run_backtest()` — replace hardcoded IC logic with `registry.get('iron_condor')`

---

## Phase 2 — StrategyParameterGrid

Cartesian product engine across all strategy parameters.

### Tasks
- [ ] Create `core/param_grid.py` — `StrategyParameterGrid` class
      ```python
      class StrategyParameterGrid:
          def __init__(self, config_dict): ...
          def generate(self): yield dict(zip(keys, combo))
      ```
- [ ] Define Iron Condor grid (first grid, validation target):
      ```
      DTE_range    ∈ {7, 14, 21, 30}
      delta_range  ∈ {0.10, 0.16, 0.20}
      width_range  ∈ {5, 10, 15}
      exit_rules   ∈ {'50pct_max', '21DTE', 'delta_breach'}
      → 4×3×3×3 = 108 combinations
      ```
- [ ] Add `--param-grid` CLI flag to backtester
- [ ] Validate grid generates correct combination count before running

---

## Phase 3 — Optimize Mode in Backtester

Systematic optimization loop with per-run metrics output.

### Tasks
- [ ] Add `--optimize` CLI flag to `condor_brain_backtest_v45.py`
- [ ] Add `--strategy` CLI flag (e.g. `--strategy iron_condor`)
- [ ] Add `--timeframe` CLI flag (`--timeframe m5`)
- [ ] Implement `OptimizationRun` record:
      ```python
      @dataclass
      class OptimizationRun:
          strategy_name: str
          param_hash:    str
          total_return:  float
          max_drawdown:  float
          sharpe:        float
          win_rate:      float
          avg_hold_time: float
          avg_credit:    float
          capital_efficiency: float
      ```
- [ ] Implement optimization loop:
      ```python
      for strategy in registry.all():
          for params in grid.generate():
              metrics = simulate(strategy, params)
              log_metrics()
              save_optimization_row()
      ```
- [ ] Output: `reports/optimization/<strategy>/<timeframe>.csv`
- [ ] Print every iteration to console: `strategy | params | Return=X | Sharpe=Y | WinRate=Z`

---

## Phase 4 — Trajectory Export Mode

State-action dataset recorder — the most important change for policy model training.

### Tasks
- [ ] Add `--export-trajectories` CLI flag
- [ ] Create `core/trajectory_logger.py` — `TrajectoryRecorder` class:
      ```python
      class TrajectoryRecorder:
          def log(self, state, action, reward, next_state, done): ...
          def to_dataframe(self): ...
          def save(self, path): ...
      ```
- [ ] Define `StateActionRecord` schema:
      ```
      timestamp | strategy_id | strategy_type | dte | width | delta_exposure
      gamma_exposure | theta | vega | unrealized_pnl | days_in_trade
      spot_features... | regime_features... | surface_features...
      action_label | reward | done_flag
      ```
- [ ] Record every bar while in trade (not just open/close events)
- [ ] Replace `trades.csv` with full state-action trajectory export when flag active
- [ ] Validate trajectory schema matches CondorNet v4.5 policy training input spec

---

## Phase 5 — Capital Constraint Engine (External Module)

Promote inline capital helpers to standalone `CapitalConstraintEngine`.

### Tasks
- [x] Inline `_deployed_margin()` + `_can_allocate()` in v4.5 entry block
- [x] Config constants: `STARTING_EQUITY=100k`, `LEVERAGE=2.0`, `MAX_OPTIONS_ALLOC_PCT=0.40`
- [ ] Create standalone `core/capital_engine.py` — `CapitalConstraintEngine` class:
      ```python
      class CapitalConstraintEngine:
          max_portfolio_alloc: float   # 0.40
          max_strategy_alloc:  float   # 0.15 per trade
          max_gamma_exposure:  float
          max_delta_exposure:  float
          margin_model:        str     # 'defined_risk' | 'naked' | 'spread'

          def can_allocate(self, equity_used, equity) -> bool
          def deployed_margin(self, open_trades) -> float
          def position_size(self, equity, strategy_params) -> int
      ```
- [ ] Wire external `CapitalConstraintEngine` into v4.5 (replace inline helpers)
- [ ] Add multi-strategy capital allocation: each strategy gets a sub-allocation slice
- [ ] Real-time adjustment pool (remaining capital after open positions):
      - After opening N positions, scan remaining capital
      - Pool filter Tier 1: strategies with win_prob ≥ 75%
      - Pool filter Tier 2: collateral-vs-reward ratio screen
      - Pool filter Tier 3: fast parameter sweep on surviving candidates
      - Select best-fit strategy for remaining capital (greedy)

### Capital Model Reference
```
Account: Options Level 3
Starting equity:  $100,000
Leverage:         2:1
Buying power:     $200,000
Max deploy:       40% × $200,000 = $80,000
Min collateral:   0.5% × buying_power = $1,000
Max collateral:   15% × buying_power  = $30,000
Win prob gate:    ≥ 75% to enter pool (Tier-1 filter)
Credit floor:     max($0.10, 5% × spread_width)
```

---

## Phase 6 — Hard Exit Taxonomy (Deterministic)

Non-learnable hard stops overlaid on model exit signal.

### Tasks
- [x] `HardExitRules` in `intelligence/exit_stack.py` (Phases 4-6 complete ✓)
- [x] `ExitDecisionStack` with protected hold zones wired in `backtest_engine.py`
- [ ] Port `ExitDecisionStack` into `condor_brain_backtest_v45.py`:
      ```python
      from intelligence.exit_stack import ExitDecisionStack, HardExitRules
      exit_stack = ExitDecisionStack(...)
      # In per-trade update loop:
      dec = exit_stack.evaluate(credit_received, current_cost, net_delta, ...)
      if dec.should_exit:
          close_trade(reason=dec.reason)
      ```
- [ ] Implement deterministic exits in v4.5 update loop:
      ```python
      if ps_pnl_pct <= -2.0:           force_exit('max_loss')
      if abs(ps_delta_exp) > 0.30:     force_exit('delta_violation')
      if dte_remaining < 3:            force_exit('expiry')
      ```
- [ ] Overlay model exit probability:
      ```python
      if p_exit > threshold and not in_protected_hold_zone:
          exit_trade('neural_exit')
      ```
- [ ] Standardize all CLOSE appends: `{'action': 'CLOSE', 'trade_id': ..., ...}`
      so OPEN/CLOSE pairs can be matched cleanly for trajectory export

---

## Phase 7 — Multi-Dataset Ingestion

M5 anchor + M1/M15 + options chain per bar.

### Tasks
- [ ] Define `MultiTFDataBundle`:
      ```python
      @dataclass
      class MultiTFDataBundle:
          m1_df:      pd.DataFrame   # 1-min bars
          m5_df:      pd.DataFrame   # 5-min bars (anchor clock)
          m15_df:     pd.DataFrame   # 15-min regime features
          options_df: pd.DataFrame   # options chain snapshots
          regime_df:  pd.DataFrame   # optional derived regime surface
      ```
- [ ] Define `UnifiedBarState`:
      ```python
      @dataclass
      class UnifiedBarState:
          timestamp:        pd.Timestamp
          spot_features:    np.ndarray    # from M1
          regime_features:  np.ndarray    # from M5/M15
          surface_features: np.ndarray    # from options surface
          chain_slice:      pd.DataFrame  # current chain snapshot
      ```
- [ ] Replace `spot_bars + ts_ranges` logic in v4.5 with `MultiTFDataBundle`
- [ ] M5 is the anchor clock — all other TFs align to M5 timestamps
- [ ] Load all 5 datasets at simulation start (pre-load, no per-bar disk reads)
- [ ] Validate alignment: M1 must have ≥ 5 bars per M5 bar; M15 resampled to M5

---

## Phase 8 — Iron Condor Grid Run (Validation)

First real optimization run — single strategy, confirm output CSV surfaces correct.

### Tasks
- [ ] Run Iron Condor grid (108 combinations) on 2025 data
- [ ] Confirm `reports/optimization/iron_condor/m5.csv` contains all 108 rows
- [ ] Verify metrics: `total_return`, `max_drawdown`, `sharpe`, `win_rate`, `avg_hold_time`
- [ ] Confirm OPEN/CLOSE event matching works (no orphaned OPEN rows)
- [ ] Confirm capital ceiling never breached in any run (assert via `_deployed_margin` check)
- [ ] Review entry gate summary — identify dominant blocking gate
- [ ] Inspect `bar_trace.jsonl` for audit trail completeness

---

## Phase 9 — Verify Output CSV Surfaces

Validate optimization surface correctness before expanding strategies.

### Tasks
- [ ] Sort `iron_condor/m5.csv` by Sharpe — confirm top row matches intuition
- [ ] Plot return surface: DTE × width, colored by Sharpe
- [ ] Confirm `reports/optimization/iron_condor/m5.csv` is reproducible (deterministic seed)
- [ ] Walk-forward check: train years 2022–2024, validate on 2025 only
      - Do NOT optimize on 2025 data — validation set only
      - Monthly walk-forward splits
- [ ] Gate summary across all 108 runs — which parameters cause most credit failures?

---

## Phase 10 — Expand to 5 Strategies

After Iron Condor grid is validated, add 4 more strategies.

### Tasks
- [ ] Implement `BullCallSpreadStrategy` (class + grid: DTE×delta×width×exit)
- [ ] Implement `BearPutSpreadStrategy`
- [ ] Implement `StrangleStrategy`
- [ ] Implement `ButterflyCallStrategy`
- [ ] Register all 5 in `StrategyRegistry`
- [ ] Define parameter grids for each (see Part III blueprint)
- [ ] Run combined 5-strategy optimization grid
- [ ] Verify capital allocation works correctly across mixed strategy types

---

## Phase 11 — Full 50+ Strategy Expansion

### Tasks
- [ ] Implement remaining strategies to reach 50+ total
- [ ] Define parameter grids for each strategy
- [ ] Estimated total combinations: ~5,400 (50 strategies × ~108 avg combos)
- [ ] GPU parallelization (Phase 12 below)

---

## Phase 12 — GPU Parallelization

### Tasks
- [ ] Benchmark single-strategy simulation time on T4
- [ ] Implement `torch.multiprocessing` or `Ray` worker pool:
      ```python
      def run_single_strategy(strategy_name, params):
          metrics = simulate(strategy_name, params)
          return metrics

      # Spawn workers per strategy
      with Pool(n_workers) as pool:
          results = pool.map(run_single_strategy, strategy_param_combos)
      ```
- [ ] One worker per strategy (not per combo — avoid memory fragmentation)
- [ ] Save results to strategy-specific CSV files in parallel

---

## Phase 13 — Validation Protocol

### Tasks
- [ ] Fix train/val split: train=2022–2024, validate=2025 only
- [ ] Walk-forward: monthly refit (retrain on rolling window, val on next month)
- [ ] Ensure optimization loop NEVER uses 2025 data for parameter selection
- [ ] Add `--val-year 2025` CLI flag that enforces holdout discipline
- [ ] Report: out-of-sample Sharpe vs in-sample Sharpe (generalization gap)

---

## CLOSE Event Standardization (Unblocks Phase 4)

> Required before trajectory export works correctly.

- [ ] Audit all close appends in v4.5 simulation loop
- [ ] Standardize every CLOSE to include:
      `{'action': 'CLOSE', 'trade_id': ..., 'bar_idx': i, 'dt': ts, 'reason': ..., 'pnl': ..., 'pnl_pct': ..., 'held_bars': ..., 'exit_details': ...}`
- [ ] Match OPEN/CLOSE pairs in trajectory export: `trade_id` is the join key
- [ ] Validate no orphaned OPEN rows in output

---

## Notes

- `condor_brain_backtest_v45.py` is the **active** backtester (v2.2 is frozen/archived)
- `core/backtest_engine.py` is deprecated — not used
- 2025 dataset only for now (2022–2024 not yet generated)
- Policy model training comes **after** trajectory dataset is validated
- `intelligence/exit_stack.py` (Phases 4–6) is tested and ready to port into v4.5
- Entry block v4.5 commit: `7761385`
- Phase 3 ETL validation script: `test_phase3_etl.py`
- Exit scaffold tests: `test_exit_stack.py` — all 42 PASS ✓
