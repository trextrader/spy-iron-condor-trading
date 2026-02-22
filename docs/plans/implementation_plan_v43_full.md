# CondorNet v4.3 — Full Data & Modeling Implementation Plan
<!-- schema_version: v4.3.0 | author: Dr. T. Jerry Mahabub, Ph.D. | date: 2026-02-22 -->

> [!IMPORTANT]
> **Authoritative spec**: This document supersedes `implementation_plan_v43.md` for all new v4.3 work.
> The canonical governing spec remains `docs/INTEGRATION_PLAN_MASTER.md`.
> If any section conflicts with that document, the master spec governs.

---

## Executive Summary

CondorNet v4.3 upgrades the system from a single-strategy (Iron Condor only) single-timeframe
(M5) model into a **full multi-strategy options AI** that:

1. Ingests all four synchronized TF datasets (M1/M5/M15/H1 — 66 features each, 264 total TF features)
2. Ingests the live SPY options chain (options_2025_v43.csv — 2.3 M rows, 20 columns)
3. Selects from a grammar of 9 strategy types (single option through custom multi-leg)
4. Predicts Probability of Profit (PoP), Expected Value (EV), and risk metrics per strategy
5. Outputs fuzzy-scaled position size driven by PoP + market regime
6. Is fully observable via the existing Tracks A–D telemetry/interpretability system

All datasets are confirmed complete with correct headers.
No GPU is available locally — all training and testing runs on Lightning AI.

---

## Dataset Inventory (Confirmed Ready)

| File | Rows | Size | Status |
|------|------|------|--------|
| `data/Datasetv4/v43/m1_dataset_v43_final.csv`  | ~91,700  | 75 MB  | ✅ Headers complete |
| `data/Datasetv4/v43/m5_dataset_v43_final.csv`  | ~18,500  | 14 MB  | ✅ Headers complete |
| `data/Datasetv4/v43/m15_dataset_v43_final.csv` | ~6,160   | 5.1 MB | ✅ Headers complete |
| `data/Datasetv4/v43/h1_dataset_v43_final.csv`  | ~1,660   | 1.4 MB | ✅ Headers complete |
| `data/Datasetv4/v43/options_2025_v43.csv`      | ~2,300,000 | 310 MB | ✅ Headers complete |

**All TF datasets share exactly 66 feature columns + 1 timestamp column = 67-column schema.**

---

## Feature Schema — V2.2 (66 Features per TF)

Feature selection is **by name only** — never by CSV column index.
Column order may differ between years/files. Always use `.loc[:, FEATURE_NAMES]`.

### Group 1 — Price / Trend (12 features)
| Feature | Type | Range (M1) | Notes |
|---------|------|-----------|-------|
| `open` | Continuous | [482, 690] | Raw price |
| `high` | Continuous | [483, 690] | Raw price |
| `low` | Continuous | [481, 690] | Raw price |
| `close` | Continuous | [482, 690] | Raw price |
| `AnchoredVWAP` | Continuous | [559, 616] | VWAP anchored to session open |
| `FRAMA` | Continuous | [483, 690] | Fractal Adaptive MA |
| `MA` | Continuous | [493, 689] | Moving average (configurable window) |
| `sma` | Continuous | [488, 689] | Simple MA |
| `bb_mu_dyn` | Continuous | [485, 689] | Bollinger midline |
| `bb_lower_dyn` | Continuous | [468, 689] | Bollinger lower band |
| `bb_upper_dyn` | Continuous | [490, 692] | Bollinger upper band |
| `target_spot` | Continuous | [485, 689] | **TARGET label** — next-period mid price; exclude from input features |

### Group 2 — Volatility / Risk (8 features)
| Feature | Type | Range (M1) | Notes |
|---------|------|-----------|-------|
| `ATR_recon` | Continuous | [0.06, 4.10] | Reconstructed ATR (absolute) |
| `atr_pct` | Bounded [0,1] | [0.00009, 0.008] | ATR / price — normalized vol regime |
| `bandwidth` | Continuous | [0.026, 16.4] | Bollinger bandwidth |
| `bb_sigma_dyn` | Continuous | [0.026, 16.4] | Bollinger std dev (= bandwidth here) |
| `bb_percentile` | Continuous | [0, 100] | Bollinger %B percentile |
| `bw_expansion_rate` | Continuous | [-0.95, 1.0] | Rate of bandwidth change |
| `adx_adaptive` | Continuous | [6, 92] | Adaptive ADX trend strength |
| `kappa_proxy` | Continuous | [-2.73, 2.72] | Kurtosis proxy |

### Group 3 — Momentum / Flow (12 features)
| Feature | Type | Range (M1) | Notes |
|---------|------|-----------|-------|
| `rsi_dyn` | Continuous | [0.5, 99.6] | Dynamic RSI; bounded [0,100] |
| `stoch_k_dyn` | Continuous | [0.2, 99.3] | Dynamic Stochastic |
| `McClellanOsc` | Continuous | [-0.30, 0.31] | McClellan breadth oscillator |
| `WeightedAlpha` | Continuous | [-8.2, 11.1] | Weighted return alpha (M1 scale) |
| `WilderAccSwingIndex` | Continuous | [-547, 561] | Wilder cumulative swing index |
| `OSC-Volume` | Continuous | [-1.5M, 1.25M] | Volume oscillator (large scale) |
| `AccDistWill` | Continuous | [-38.9, 380.8] | Accumulation/Distribution Williams |
| `AccDistWillMovAvg` | Continuous | [-35.3, 380.3] | Moving avg of AccDistWill |
| `pressure_up` | Bounded [0,1] | [0, 1] | Fraction of up-pressure bars |
| `pressure_down` | Bounded [0,1] | [0, 1] | Fraction of down-pressure bars |
| `vol_energy` | Continuous | [2e-6, 1.32] | Volume energy EWMA |
| `vol_ewma` | Bounded [0,1] | [6e-5, 0.006] | Volume EWMA normalized |

### Group 4 — Regime / Score Flags (11 features)
| Feature | Type | Range / Cardinality | Notes |
|---------|------|-----------|-------|
| `gap_risk_score` | Low-cardinality | {0.0, 0.4, 0.8} | Gap risk tier |
| `breakout_score` | Ternary | {-1, 0, 1} | Direction of breakout |
| `chaos_membership` | Bounded [0,1] | sparse ~99.6% NaN | Chaotic regime flag |
| `consolidation_score` | Bounded [0,1] | [0.02, 0.52] | Range-bound strength |
| `fuzzy_reversion_11` | Bounded [0,1] | [0.35, 0.63] | Fuzzy mean-reversion signal |
| `max_dd_60m` | Bounded [0,1] | [0.0002, 0.060] | Max drawdown over 60 min |
| `bb_percentile` | Continuous | [0, 100] | *(also in Group 2)* |
| `spread_ratio` | Bounded [0,1] | [3e-5, 0.032] | Bid-ask spread / price |
| `friction_ratio` | Low-cardinality | M1: const 0.5; M5+: [0.52, 1.0] | ⚠ TYPE_MISMATCH across TFs — handle in ETL |
| `position_size_mult` | Low-cardinality | {0.5, 0.75, 1.0} | **TARGET label** — exclude from input features |
| `log_return` | Continuous | [-0.035, 0.034] | Log return per bar |

### Group 5 — Trend Structure (6 features)
| Feature | Type | Notes |
|---------|------|-------|
| `TrendSeekerUp` | Continuous | Uptrend structural level; ~44% NaN |
| `TrendSeekerDown` | Continuous | Downtrend structural level; ~44% NaN |
| `TurtleChn-Up` | Continuous | Turtle channel upper |
| `TurtleChn-Low` | Continuous | Turtle channel lower |
| `psar_mark` | Continuous | Parabolic SAR price level |
| `psar_adaptive` | Continuous | SAR acceleration |

### Group 6 — PSAR / Regime (4 features)
| Feature | Type | Notes |
|---------|------|-------|
| `psar_trend` | Ternary | {-1, 1} — current SAR direction |
| `psar_reversion_mu` | Bounded [0,1] | SAR reversion probability |
| `ret_z` | Continuous | Z-score of log return |
| `trade_count` | Continuous | Trade count per bar |

### Group 7 — Pivot & Segment Features (13 features — SPARSE)
| Feature | Type | NaN% (M1) | Notes |
|---------|------|-----------|-------|
| `PivotHigh` | Continuous | 97.5% | Price at confirmed pivot high; NaN otherwise |
| `PivotLow` | Continuous | 97.6% | Price at confirmed pivot low; NaN otherwise |
| `Slope` | Continuous | 0.03% | Segment slope (price/bar); ⚠ RANGE_MISMATCH across TFs |
| `SlopeATR` | Continuous | 0.03% | ATR-normalized slope; ⚠ RANGE_MISMATCH |
| `PivotResidual` | Continuous | 0.02% | Last-bar deviation from linear fit |
| `PivotResidualATR` | Continuous | 0.02% | ATR-normalized residual |
| `PivotResidualZ` | Continuous | 0.02% | Z-score of residual |
| `PivotCurvatureProxy` | Continuous | 5.85% | 2nd derivative proxy; ⚠ SPARSITY_MISMATCH |
| `PivotCurvatureATR` | Continuous | 5.85% | ATR-normalized curvature; ⚠ SPARSITY_MISMATCH |
| `PivotSegmentLengthBars` | Continuous | 0.02% | Segment length in bars |
| `PivotSegmentLengthMinutes` | Continuous | 0.02% | Segment length in minutes; ⚠ RANGE_MISMATCH |
| `PivotSegmentResidualStd` | Continuous | 0.02% | Std of within-segment residuals |
| `PivotSegmentVolatility` | Continuous | 0.02% | ATR of within-segment bars |

**Model input features (excluding labels)**: 64 features per TF × 4 TFs = **256 total TF features**
(`target_spot` and `position_size_mult` are training labels, not inputs.)

---

## Options Chain Schema (options_2025_v43.csv)

| Column | Type | Range / Notes |
|--------|------|--------------|
| `timestamp` | datetime | Minute-level; join key to TF datasets |
| `contract_id` | str | Unique contract identifier |
| `symbol` | str | "SPY" (constant) |
| `expiration` | date | Contract expiration date |
| `strike` | float | Strike price |
| `type` | str | "call" or "put" |
| `last` | float | Last trade price |
| `mark` | float | (bid + ask) / 2 |
| `bid` | float | Best bid |
| `bid_size` | int | Bid lot size |
| `ask` | float | Best ask |
| `ask_size` | int | Ask lot size |
| `volume` | int | Contracts traded |
| `open_interest` | int | Open interest |
| `implied_volatility` | float | IV; typically [0.10, 0.60] for SPY |
| `delta` | float | Call delta ∈ [0,1]; Put delta ∈ [-1,0] |
| `gamma` | float | ≥ 0 |
| `theta` | float | ≤ 0 (time decay) |
| `vega` | float | ≥ 0 |
| `rho` | float | Interest rate sensitivity |
| `in_the_money` | int | 0 or 1 flag |

**Validation rules**:
- `bid ≤ ask` everywhere
- `last` ∈ [`bid`, `ask`]
- `delta` ∈ [-1, 1]; calls positive, puts negative
- `gamma` ≥ 0; `vega` ≥ 0
- `implied_volatility` > 0
- No duplicate (timestamp, contract_id) rows

---

## Architecture Overview — CondorNet v4.3

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT LAYER                                  │
│                                                                  │
│  M1 Features    M5 Features   M15 Features   H1 Features        │
│  [B, T, 64]     [B, T, 64]    [B, T, 64]     [B, T, 64]        │
│       │              │              │               │             │
│  ┌────┴──────────────┴──────────────┴───────────────┴────┐      │
│  │         Multi-TF Projection + Concat  [B, T, 256]      │      │
│  └──────────────────────────┬──────────────────────────-──┘      │
│                             │                                     │
│  Options Chain [B, N_contracts, 10]                              │
│       │                                                          │
│  ┌────┴────────────────┐                                         │
│  │  OptionsChainEncoder│  → chain_embed [B, T, d_chain=128]     │
│  └────────────────────-┘                                         │
└─────────────────────────────────────────────────────────────────┘
                     │                  │
              TF features          chain embed
                     │                  │
              ┌──────┴──────────────────┴──────┐
              │     JOINT FUSION [B, T, 384]    │
              └─────────────────┬──────────────┘
                                │
              ┌─────────────────▼──────────────────┐
              │     CondorNet Core (unchanged)      │
              │   ETD-1 + TFT Control + CDE +       │
              │   Predicate Logic + FusionGate       │
              │         State: [h, v, m, r]          │
              └───────────────┬─────────────────────┘
                              │
         ┌────────────────────┼─────────────────────────┐
         │                    │                          │
    ┌────▼────┐         ┌─────▼──────┐           ┌──────▼──────┐
    │Strategy │         │  PoP / EV  │           │  Position   │
    │Selector │         │  Regressor │           │  Size Head  │
    │(9 class)│         │(PoP,EV,VaR)│           │ (Fuzzy + NN)│
    └────┬────┘         └─────┬──────┘           └──────┬──────┘
         │                    │                          │
    ┌────▼────────────────────▼──────────────────────────▼──────┐
    │                   STRATEGY CONSTRUCTOR                      │
    │  (Leg builder: Strike, Expiry, Long/Short, Qty)             │
    └────────────────────────────────────────────────────────────┘
```

---

## Files to Create (New)

| # | File | Purpose |
|---|------|---------|
| F1 | `intelligence/schema_v43.py` | V2.2 schema constants: feature names, types, NaN policy, ranges, normalization specs |
| F2 | `intelligence/data_pipeline_v43.py` | ETL: load all 4 TF CSVs + options chain; validate; align timestamps; produce tensors |
| F3 | `intelligence/options_chain_encoder.py` | Options chain → latent representation: moneyness grid, strike attention transformer |
| F4 | `intelligence/strategy_generator.py` | Enumerate/sample strategy candidates per timestamp (9 types × strike × expiry combos) |
| F5 | `intelligence/payoff_calculator.py` | PoP, EV, Max P&L, Net Greeks, VaR/CVaR via Black-Scholes under implied-vol lognormal |
| F6 | `intelligence/target_labeler_v43.py` | Generate training labels: PoP/EV/risk per (timestamp, strategy); ideal strategy per bar |
| F7 | `intelligence/condor_train_net_v43.py` | Complete v43 training loop: multi-input DataLoader + new loss + all Track A–D hooks |
| F8 | `intelligence/training/training_hooks_v43.py` | Modular hooks: deep_observe(), checkpoint(), emit(), crash_sentinel() |

---

## Files to Update (Existing)

| # | File | What Changes | How |
|---|------|-------------|-----|
| U1 | `intelligence/schema_v43.py` | **NEW** — defines V2.2 | See F1 above |
| U2 | `intelligence/condor_brain_net.py` | Multi-TF input projection; OptionsChainEncoder integration; StrategyHead; PoP/EV heads | Add classes before `CondorNet`; update `__init__` and `forward` signatures |
| U3 | `intelligence/condor_brain.py` | Update `CondorSignal` dataclass; update `CondorBrain.forward` for multi-TF; update `CondorBrainEngine` |  New fields in dataclass; new input routing in forward |
| U4 | `intelligence/fuzzy_engine.py` | Add `compute_pop_based_sizing(pop, ev, var)`; add linguistic term definitions | New function block; update `compute_position_size` to accept optional PoP override |
| U5 | `intelligence/canonical_feature_registry.py` | Sync with exact V2.2 66-feature list; add options chain feature catalog | Edit feature list constants to match column profile exactly |
| U6 | `core/config.py` | Add `v43_data_dir`, `options_chain_path`, `strategy_universe`, `max_legs`, `pop_target_min`, `schema_version` to `StrategyConfig`/`RunConfig` | Extend both dataclasses |

---

## TRACK 0 — Dataset Validation & ETL (NEW)

### 0.1 — Schema Audit Module (`intelligence/schema_v43.py`)

**Delivers**: Single source of truth for all V2.2 feature definitions.

```python
# intelligence/schema_v43.py  (structure)

TF_FEATURE_NAMES: list[str]          # 64 input features (excludes target_spot, position_size_mult)
TF_LABEL_NAMES:   list[str]          # ["target_spot", "position_size_mult"]
TF_PIVOT_FEATURES: list[str]         # 13 sparse pivot features — special NaN handling
TF_SPARSE_FEATURES: list[str]        # features that legitimately have NaN (never impute)
TF_BOUNDED_FEATURES: dict[str, tuple] # {name: (min_valid, max_valid)}
TF_TERNARY_FEATURES: list[str]       # ["breakout_score", "psar_trend"]
TF_DTYPE_MAP: dict[str, str]         # column → expected dtype

CHAIN_FEATURE_NAMES: list[str]       # 10 encoded chain features per contract
CHAIN_LABEL_NAMES: list[str]         # ["pop", "ev", "max_profit", "max_loss", "var_95", "cvar_95"]

KNOWN_CROSS_TF_FLAGS: dict[str, list] # {col: ["TYPE_MISMATCH", "RANGE_MISMATCH", ...]}
SCHEMA_VERSION = "v4.3.0"
```

**Validation function**:
```python
def validate_tf_dataframe(df: pd.DataFrame, tf_name: str) -> ValidationReport:
    """
    Checks:
    - All TF_FEATURE_NAMES present (raises if missing)
    - dtypes match TF_DTYPE_MAP
    - Bounded features within valid ranges (warn on violations > 0.01%)
    - Ternary features only in {-1, 0, 1}
    - Timestamp strictly increasing (no duplicates)
    - NaN% for non-pivot features < 5% (warn if exceeded)
    - No NaN in OHLCV columns
    Returns: ValidationReport(passed: bool, warnings: list, errors: list)
    """
```

### 0.2 — Multi-TF ETL Pipeline (`intelligence/data_pipeline_v43.py`)

**Delivers**: Aligned, validated, normalized tensors for all 4 TFs + options chain.

```
DataPipelineV43
├── load_tf_dataset(path, tf_name) → pd.DataFrame
├── validate_tf_dataset(df, tf_name) → ValidationReport
├── align_timestamps(m1_df, m5_df, m15_df, h1_df) → AlignedDataset
│     Logic: M5 bar timestamp = right-edge of 5×M1 bars
│            M15 bar timestamp = right-edge of 15×M1 bars
│            H1 bar timestamp = right-edge of 60×M1 bars
│     Returns dict of aligned DataFrames with same index (M5 timestamps as master clock)
├── normalize_features(df) → pd.DataFrame
│     Per-feature normalization:
│       Continuous → RobustScaler (median/IQR) — handles OSC-Volume's ±1.5M range
│       Bounded [0,1] → PassThrough (already normalized)
│       Ternary {-1,0,1} → PassThrough (keep as integer encoding)
│       Sparse pivot features → normalize non-NaN values; keep NaN as 0.0 with mask
│       Low-cardinality (gap_risk_score) → ordinal encode {0→0, 0.4→0.5, 0.8→1.0}
├── build_pivot_mask(df) → pd.DataFrame  # True where pivot features are NaN
├── load_options_chain(path) → pd.DataFrame
│     Chunked load (310 MB): dask or pandas chunked read
│     Parse timestamps, compute derived fields:
│       moneyness = log(spot_price / strike)
│       days_to_exp = (expiration - timestamp.date()).days
│       width_of_mid = (ask - bid) / mark
├── validate_options_chain(df) → ValidationReport
├── create_chain_snapshot(chain_df, timestamp) → ChainSnapshot
│     Returns dict: {(strike, type, expiry): {iv, delta, gamma, theta, vega, bid, ask, oi}}
└── build_dataset(config: PipelineConfig) → V43Dataset
      Returns torch Dataset for training
```

**Key alignment logic** (no look-ahead):
```
For each M5 bar at time T:
  M5 features: bar [T-5min, T)   → use as-is
  M1 features: last 5 M1 bars ending at T → take M1 row at T
  M15 features: last M15 bar ending at or before T
  H1 features: last H1 bar ending at or before T
  Options chain: snapshot at T (or last available T' < T)
```

### 0.3 — Cross-TF Anomaly Handling

Known flags from `CROSS_TF_COLUMN_CONTRACT_v43.csv`:

| Flag | Affected Columns | Handling in ETL |
|------|-----------------|----------------|
| `TYPE_MISMATCH` | `friction_ratio` (M1=const, M5+=range), `vol_energy` | Cast M1 `friction_ratio` to float64; log warning |
| `RANGE_MISMATCH` | `WeightedAlpha`, `Slope`, `SlopeATR`, `PivotSegmentLengthMinutes` | Scale M5+ values to M1-equivalent range (per-TF normalization handles this via RobustScaler) |
| `SPARSITY_MISMATCH` | All `Pivot*` features (M5 is 73.9% NaN vs M1's 5.9%) | Track NaN mask separately per TF; never impute |
| `CONST_MISMATCH` | `friction_ratio` on M1 | Convert to per-TF feature — M1 gets constant 0.5 encoded as special |

### 0.4 — Target Label Generation (`intelligence/target_labeler_v43.py`)

**Delivers**: Pre-computed training labels saved as `data/Datasetv4/v43/labels_v43.parquet`.

For each (timestamp, strategy_type):

```
1. Get current spot price S from M5 close
2. Get options chain snapshot at timestamp
3. For each candidate strategy in StrategyUniverse.sample(chain_snapshot):
   a. Compute payoff profile (via PayoffCalculator)
   b. Compute PoP = P(payoff > 0 at expiry) via BS normal CDF
   c. Compute EV  = E[payoff at expiry] under risk-neutral measure
   d. Compute max_profit, max_loss, breakeven_lo, breakeven_hi
   e. Compute net_delta, net_gamma, net_theta, net_vega
   f. Compute VaR_95, CVaR_95 under lognormal distribution
4. Store (timestamp, strategy_id, pop, ev, max_profit, max_loss, var_95, cvar_95, net_delta)
5. Also store: ideal_strategy_id = argmax(pop - lambda * cvar_95)  where lambda=0.5
```

Output schema (`labels_v43.parquet`):
```
timestamp | strategy_id | strategy_type | legs_json | pop | ev | max_profit | max_loss
| var_95 | cvar_95 | net_delta | net_gamma | net_theta | net_vega | is_ideal
```

---

## TRACK 1 — Options Strategy Universe (`intelligence/strategy_generator.py`)

### Strategy Grammar

9 strategy types with leg specifications:

| ID | Type | Legs | Construction |
|----|------|------|-------------|
| 0 | `single_call` | 1 | Long 1 call at target delta |
| 1 | `single_put` | 1 | Long 1 put at target delta |
| 2 | `bull_call_spread` | 2 | Long call K1, Short call K2 (K1 < K2, same expiry) |
| 3 | `bear_put_spread` | 2 | Long put K2, Short put K1 (K1 < K2, same expiry) |
| 4 | `straddle` | 2 | Long call + Long put at same ATM strike |
| 5 | `strangle` | 2 | Long OTM call + Long OTM put (equal delta distance) |
| 6 | `butterfly_call` | 3 | Long 1 call K1, Short 2 calls K2, Long 1 call K3 (K1<K2<K3 symmetric) |
| 7 | `iron_condor` | 4 | Short put K2 + Long put K1 + Short call K3 + Long call K4 (K1<K2<K3<K4) |
| 8 | `custom_multi_leg` | N | Model-generated structure up to 4 legs |

### StrategyGenerator class

```python
class StrategyGenerator:
    def __init__(self, config: StrategyConfig):
        self.target_deltas = config.target_deltas  # e.g., [0.15, 0.20, 0.25, 0.30]
        self.target_dtes   = config.target_dtes    # e.g., [0, 1, 2, 7]  days
        self.max_legs      = config.max_legs        # 4

    def sample(self, chain_snapshot: ChainSnapshot, spot: float,
               n_candidates: int = 50) -> list[Strategy]:
        """
        Generates candidate strategies at current bar.
        For each strategy type:
          1. Select strikes by nearest-delta matching
          2. Select expiry by DTE target
          3. Validate leg availability (bid/ask not NaN, OI > 0)
          4. Return Strategy objects with leg specs
        """

    def build_iron_condor(self, chain, spot, short_delta=0.20, wing_width=5.0, dte=0) -> Strategy:
        """Exact same logic as existing options_strategy.py:build_condor — preserves backward compat"""
```

---

## TRACK 2 — Payoff Calculator (`intelligence/payoff_calculator.py`)

### Mathematical Definitions

**Probability of Profit** (Black-Scholes, per Gomes[3]):
```
For a strategy with profit range [B_lo, B_hi] at expiry:
  PoP = P(B_lo ≤ S_T ≤ B_hi)
  S_T ~ LogNormal(μ = log(S) + (r - σ²/2)τ, σ² = IV² × τ)
  PoP = N(d2_hi) - N(d2_lo)  where d2 = (log(S/K) + (r - σ²/2)τ) / (σ√τ)
```

**Expected Value** (risk-neutral):
```
EV = E_Q[payoff(S_T)] = integral of payoff(S_T) × f_Q(S_T) dS_T
   For simple spreads: analytic BS formula
   For complex multi-leg: Monte Carlo (n=10,000 paths)
```

**VaR / CVaR** (95th percentile):
```
VaR_95  = -quantile(payoff_distribution, 0.05)
CVaR_95 = -E[payoff | payoff < -VaR_95]
```

### PayoffCalculator class

```python
class PayoffCalculator:
    def compute_pop(self, strategy: Strategy, spot: float, iv: float,
                    r: float = 0.05, tau: float = None) -> float: ...

    def compute_ev(self, strategy: Strategy, spot: float, iv: float,
                   r: float, tau: float, n_mc: int = 10_000) -> float: ...

    def compute_max_profit_loss(self, strategy: Strategy) -> tuple[float, float]: ...

    def compute_breakevens(self, strategy: Strategy, spot: float) -> list[float]: ...

    def compute_net_greeks(self, strategy: Strategy) -> NetGreeks: ...

    def compute_var_cvar(self, strategy: Strategy, spot: float, iv: float,
                         r: float, tau: float, confidence: float = 0.95) -> tuple[float, float]: ...

    def payoff_at_expiry(self, strategy: Strategy, S_T: float) -> float:
        """Net payoff (credit received - cost to close) at given underlying price."""
```

---

## TRACK 3 — Options Chain Encoder (`intelligence/options_chain_encoder.py`)

### Representation

At each timestamp, the chain is encoded as a **moneyness-indexed grid**:

```
Chain grid per timestamp: [N_strikes × 2 × n_expiries, chain_features]
  - N_strikes: nearest 20 strikes (10 calls + 10 puts) to ATM
  - 2: call and put for each strike
  - n_expiries: nearest 3 expiration dates
  - chain_features (10): [moneyness, days_to_exp, iv, delta, gamma, theta, vega, bid, ask, oi_norm]
```

### OptionsChainEncoder architecture

```python
class OptionsChainEncoder(nn.Module):
    """
    Input:  chain_grid [B, N_contracts, 10]  (N_contracts ≤ 120 after filtering)
    Output: chain_embed [B, d_chain]          (d_chain = 128)

    Architecture:
      1. Linear projection: 10 → d_model (64)
      2. Positional encoding by moneyness rank
      3. Transformer encoder: 2 layers, 4 heads, d_ff=256
      4. Mean pooling over N_contracts → [B, d_model]
      5. Linear projection → [B, d_chain=128]

    Rationale: Transformer handles variable-length chains (missing strikes OK).
               Mean pooling is permutation-invariant (no assumed ordering).
    """
    def forward(self, chain_grid: Tensor, key_padding_mask: Tensor = None) -> Tensor:
        """
        key_padding_mask: [B, N_contracts] bool tensor — True for padded/missing contracts
        Returns: [B, 128] chain embedding
        """
```

---

## TRACK 4 — Model Architecture Updates (`intelligence/condor_brain_net.py`)

### 4.1 — Multi-TF Input Projection

Add `MultiTFProjector` class before `CondorNet`:

```python
class MultiTFProjector(nn.Module):
    """
    Projects 4 TF feature tensors to a common dimension and concatenates.
    Input:  dict of {tf_name: Tensor[B, T, 64]}
    Output: Tensor[B, T, d_joint]  where d_joint = 4 × d_tf_proj

    Each TF gets its own linear projection (64 → d_tf_proj=64)
    because TF features have different statistical scales.
    Concatenated: [B, T, 256]
    """
```

### 4.2 — Strategy Selection Head

Add `StrategyHead` class:

```python
class StrategyHead(nn.Module):
    """
    Output:
      strategy_logits: [B, 9]      — softmax over 9 strategy types
      leg_params:      [B, 4, 5]   — per-leg: [strike_offset, delta_target, expiry_days, long_short, qty]
      entry_signal:    [B, 1]      — binary entry decision
    """
```

### 4.3 — PoP / EV Regression Head

Add `RiskMetricHead` class:

```python
class RiskMetricHead(nn.Module):
    """
    Output:
      pop:        [B, 1]   — predicted probability of profit ∈ [0,1] (sigmoid)
      ev:         [B, 1]   — predicted expected value (unbounded)
      max_loss:   [B, 1]   — predicted max loss ≥ 0 (softplus)
      var_95:     [B, 1]   — predicted VaR ≥ 0 (softplus)
      cvar_95:    [B, 1]   — predicted CVaR ≥ 0 (softplus)
    """
```

### 4.4 — Update `CondorNet.__init__` signature

```python
def __init__(self,
             input_dim: int = 256,          # 4 TFs × 64 features (after projection)
             chain_dim: int = 128,           # OptionsChainEncoder output dim
             d_joint: int = 384,             # Combined joint representation
             n_strategy_types: int = 9,      # Strategy universe size
             max_legs: int = 4,              # Maximum legs per strategy
             # ... all existing params preserved ...
             ):
```

### 4.5 — Update `CondorNet.forward` signature

```python
def forward(self,
            x_m1:  Tensor,          # [B, T, 64]
            x_m5:  Tensor,          # [B, T, 64]
            x_m15: Tensor,          # [B, T, 64]
            x_h1:  Tensor,          # [B, T, 64]
            chain: Tensor,          # [B, N_contracts, 10]
            chain_mask: Tensor = None,  # [B, N_contracts] padding mask
            pivot_features: Tensor = None,  # [B, T, 13] sparse pivot features
            return_diagnostics: bool = False
            ) -> CondorNetOutput:
```

**Backward compatibility**: The existing `forward(x, ...)` single-TF signature is preserved as a
wrapper that broadcasts `x` to all four TFs with zero chain input. This ensures existing backtests
continue to run without modification.

---

## TRACK 5 — Training Loop v4.3 (`intelligence/condor_train_net_v43.py`)

### 5.1 — Multi-Input DataLoader

```python
class V43Dataset(Dataset):
    """
    Yields per-sample:
    {
      "x_m1":  Tensor[seq_len, 64],    # M1 features (normalized)
      "x_m5":  Tensor[seq_len, 64],    # M5 features (normalized)
      "x_m15": Tensor[seq_len, 64],    # M15 features (normalized)
      "x_h1":  Tensor[seq_len, 64],    # H1 features (normalized)
      "chain": Tensor[N_contracts, 10], # Options chain snapshot
      "pivot_mask": Tensor[seq_len, 13], # NaN mask for pivot features
      "labels": {
          "target_spot":     Tensor[1],
          "position_size":   Tensor[1],
          "strategy_type":   Tensor[1, long],   # class label 0-8
          "pop":             Tensor[1],
          "ev":              Tensor[1],
          "max_loss":        Tensor[1],
          "var_95":          Tensor[1],
          "cvar_95":         Tensor[1],
      }
    }
    """
```

### 5.2 — Combined Loss Function

```python
class CondorLossV43(nn.Module):
    """
    L_total = w1 * L_strategy     # Cross-entropy over 9 strategy types
            + w2 * L_pop          # BCE loss for PoP
            + w3 * L_ev           # MSE loss for EV
            + w4 * L_risk         # MSE for VaR, CVaR (combined)
            + w5 * L_size         # MSE for position size multiplier
            + w6 * L_spot         # MSE for target spot price (auxiliary)
            + w7 * L_fuzzy_var    # Existing fuzzy gate variance penalty
            + w8 * L_pattern_ent  # Existing pattern entropy regularization
            + w9 * L_robust       # Robust pricing duality (from INTEGRATION_PLAN_MASTER)

    Default weights: w1=0.25, w2=0.20, w3=0.15, w4=0.15, w5=0.10, w6=0.05, w7=0.05, w8=0.03, w9=0.02
    All weights configurable via CLI.

    Probability calibration: ensure predicted PoP matches empirical frequency
    using ECE (Expected Calibration Error) as soft constraint.
    """
```

### 5.3 — v43 Training Entry Point

CLI:
```bash
python intelligence/condor_train_net_v43.py \
    --data-dir data/Datasetv4/v43 \
    --labels   data/Datasetv4/v43/labels_v43.parquet \
    --epochs 50 --batch-size 256 --seq-len 64 \
    --d-model 256 --d-chain 128 \
    --deep-observe --observe-every 100 \
    --checkpoint-dir models/ckpts/v43 \
    --resume models/ckpts/v43/last.pth
```

### 5.4 — Carry-Forward: All Track A–D Hooks

All hooks from the existing implementation plan (A1.4, A2.4-A2.7, A3.1-A3.4, A4, A5, B1-B4, C1-C4, D0-D6) are implemented in `intelligence/training/training_hooks_v43.py` and called from the v43 training loop. **This is the single place where ALL hook implementations live for v43.**

---

## TRACK 6 — Fuzzy Sizing Expansion (`intelligence/fuzzy_engine.py`)

### 6.1 — PoP-Based Linguistic Terms

```python
# Linguistic term definitions (per spec Section 7)
POP_VERY_HIGH  = lambda pop: trimf(pop, [0.70, 0.85, 1.00])
POP_HIGH       = lambda pop: trimf(pop, [0.55, 0.68, 0.80])
POP_MEDIUM     = lambda pop: trimf(pop, [0.40, 0.50, 0.65])
POP_LOW        = lambda pop: trimf(pop, [0.25, 0.38, 0.55])
POP_VERY_LOW   = lambda pop: trimf(pop, [0.00, 0.15, 0.35])

VOL_LOW  = lambda atr_pct: trimf(atr_pct, [0.000, 0.002, 0.004])
VOL_MED  = lambda atr_pct: trimf(atr_pct, [0.003, 0.005, 0.008])
VOL_HIGH = lambda atr_pct: trimf(atr_pct, [0.006, 0.009, 0.015])
```

### 6.2 — PoP Fuzzy Rule Base

```
Rule 1:  IF PoP is VERY_HIGH AND Vol is LOW  → size = 1.00 (full)
Rule 2:  IF PoP is VERY_HIGH AND Vol is MED  → size = 0.85
Rule 3:  IF PoP is VERY_HIGH AND Vol is HIGH → size = 0.70
Rule 4:  IF PoP is HIGH AND Vol is LOW       → size = 0.80
Rule 5:  IF PoP is HIGH AND Vol is MED       → size = 0.65
Rule 6:  IF PoP is HIGH AND Vol is HIGH      → size = 0.50
Rule 7:  IF PoP is MEDIUM                   → size = 0.40
Rule 8:  IF PoP is LOW                      → size = 0.20
Rule 9:  IF PoP is VERY_LOW                 → size = 0.00 (no trade)
```

### 6.3 — New Function Signature

```python
def compute_pop_based_sizing(
    pop: float,          # Predicted PoP from model [0,1]
    ev: float,           # Predicted EV ($)
    var_95: float,       # Predicted 95% VaR (≥0)
    atr_pct: float,      # Current volatility regime
    base_size: float     # Base position size from existing 10-factor fuzzy
) -> float:
    """
    Combines PoP-based sizing with existing 10-factor fuzzy output.
    Final size = 0.5 * base_size + 0.5 * pop_size
    (Equal weight blend — configurable via config.pop_sizing_weight)
    Returns: float in [0, 1]
    """
```

---

## TRACK 7 — Carry-Forward: Existing Tracks A–D

All items from `task_checklist_v43.md` that are still pending are implemented in v43.
**They are not duplicated here** — the full list with status is in `task_checklist_v43_full.md`.

Pending items at time of writing:
- A1.4, A2.4-2.7, A3.1-3.4, A4.1, A5.1
- B1.7, B1.9, B1.10, B1.F
- B2.1-2.8, B3.1-3.4, B4.1-4.7
- C1.1-1.6, C2.1-2.4, C3.1-3.3, C4.1-4.6
- D0.1-0.2, D1.1-1.5, D2.1-2.3, D3.1-3.2, D4.1-4.2, D5.1, D6.1

Implementation order: unchanged from the original plan's 5-Phase sequence.
All Track A–D hooks live in `intelligence/training/training_hooks_v43.py`.

---

## Execution Order (6 Phases)

### Phase 0 — Schema & Validation Foundation
1. **F1**: `schema_v43.py` — V2.2 constants + validators
2. **F2 (part 1)**: `data_pipeline_v43.py` — load + validate all 4 TF CSVs
3. **U5**: `canonical_feature_registry.py` — sync to V2.2 exactly
4. **U6**: `core/config.py` — add v43 config fields

### Phase 1 — Options Chain & Strategy Engine
5. **F2 (part 2)**: `data_pipeline_v43.py` — add options chain loading + temporal alignment
6. **F3**: `options_chain_encoder.py` — chain grid representation + transformer encoder
7. **F4**: `strategy_generator.py` — 9-strategy grammar + leg construction
8. **F5**: `payoff_calculator.py` — PoP/EV/VaR/CVaR under Black-Scholes

### Phase 2 — Target Label Generation
9. **F6**: `target_labeler_v43.py` — generate `labels_v43.parquet`
   *(This is a one-time offline job — run on Lightning AI with the full dataset)*

### Phase 3 — Architecture Upgrades
10. **U2**: `condor_brain_net.py` — MultiTFProjector + OptionsChainEncoder integration + StrategyHead + RiskMetricHead
11. **U3**: `condor_brain.py` — CondorSignal update + multi-TF forward
12. **U4**: `fuzzy_engine.py` — PoP-based sizing + linguistic terms

### Phase 4 — Training Loop & Hooks
13. **F7**: `condor_train_net_v43.py` — full v43 training loop with multi-input DataLoader
14. **F8**: `training_hooks_v43.py` — all Track A–D hooks
15. A1.4, A3.1–3.4, A2.4–2.7, A4.1 (stability & crash recovery)

### Phase 5 — Observability & Telemetry
16. D0, B1.F, B1.7, B1.9, B1.10 (deep observe core)
17. B2.1–2.8, B3.1–3.4, B4 (predicate intelligence)
18. C1–C4 (subsystem attribution & drift)

### Phase 6 — Reporting & Integration
19. D3, D6, A5 (reports + model DNA + run summary)
20. D4, D5 (GUI telemetry + batch replay)
21. B4.2, B4.6, B4.7 (predicate explanations + logic graph)

---

## Key Invariants & Contracts

1. **No CSV order dependence**: All feature access uses `.loc[:, FEATURE_NAMES]` with named lists from `schema_v43.py`.
2. **No look-ahead bias**: All features at bar T use only data from bars < T. Options chain at T uses snapshot from T (or last T' < T if T unavailable).
3. **Pivot NaN is semantic**: Pivot features are NaN when no event occurred. These are NEVER imputed. The model receives a `pivot_mask` tensor indicating NaN positions.
4. **4-leg Iron Condor invariant preserved**: `build_iron_condor()` in `strategy_generator.py` uses the exact same fail-fast logic as `strategies/options_strategy.py:build_condor` — no partial positions ever created.
5. **Backward compatibility**: All existing backtest commands continue to work unchanged. The multi-TF upgrade is additive only.
6. **Schema version in every output**: All checkpoints, JSON reports, and label files include `"schema_version": "v4.3.0"`.
7. **No GPU required locally**: All code paths include CPU fallback. Training and profiling runs on Lightning AI only.

---

## Cross-Reference: Files ↔ Spec Sections

| Spec Section | Implemented In |
|-------------|---------------|
| §1 Data Inputs & Validation | `schema_v43.py`, `data_pipeline_v43.py` |
| §2 Underlying Market Features | `schema_v43.py` (V2.2 feature list), `data_pipeline_v43.py` (normalization) |
| §3 Options Chain Features | `options_chain_encoder.py`, `data_pipeline_v43.py` |
| §4 Strategy Universe | `strategy_generator.py` |
| §5 Risk/Return Metrics | `payoff_calculator.py` |
| §6 Model Architecture | `condor_brain_net.py` (updated), `options_chain_encoder.py` |
| §7 Fuzzy Logic Sizing | `fuzzy_engine.py` (updated) |
| §8 Implementation Checklist | This document + `task_checklist_v43_full.md` |

---

*Last updated: 2026-02-22 | Next milestone: Phase 0 complete (schema_v43.py + ETL validation)*
