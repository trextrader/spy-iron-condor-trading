# Column Schema Audit - V2.2 Feature Registry

## Summary

| Metric | Count |
|--------|-------|
| **CSV Total Columns** | 65 |
| **V2.2 Model Input Features** | 54 |
| **Metadata/Identifier Columns** | 9 |
| **Legacy/Reference Columns** | 4 |

---

## 1. METADATA / IDENTIFIERS (9 columns)
*Not used as model inputs - provide context only*

| # | Column | Type | Purpose |
|---|--------|------|---------|
| 1 | `timestamp` | datetime | Record timestamp (UTC) |
| 2 | `symbol` | str | Underlying symbol (SPY) |
| 3 | `underlying_price` | float | SPY spot price snapshot |
| 4 | `option_symbol` | str | OCC contract identifier |
| 5 | `expiration` | datetime | Option expiration date |
| 6 | `call_put` | str | "C" or "P" |
| 7 | `rho` | float | Interest rate Greek (not in V2.2 model) |
| 8 | `volume` | int | 1-minute volume (also in model as input) |
| 9 | `open_interest` | int | Open interest (reference only) |

---

## 2. LEGACY / REFERENCE COLUMNS (4 columns)
*Kept for debugging/reference, not model inputs*

| # | Column | Status | Notes |
|---|--------|--------|-------|
| 36 | `sma` | Legacy | SMA20 - superseded by bb_mu_dyn |
| 37 | `psar_mark` | Legacy | Raw PSAR value - superseded by psar_adaptive |
| 54 | `bandwidth` | Legacy | Raw BB width - superseded by bb_percentile, bw_expansion_rate |
| - | `volume_ratio` | **REMOVED** | Replaced by `cmf` in V2.2 |

---

## 3. V2.2 CANONICAL MODEL INPUT FEATURES (52 columns)

### 3.1 Market Primitives (5)

| # | Column | Range | Purpose |
|---|--------|-------|---------|
| 1 | `open` | SPY $ | 1-minute open |
| 2 | `high` | SPY $ | 1-minute high |
| 3 | `low` | SPY $ | 1-minute low |
| 4 | `close` | SPY $ | 1-minute close |
| 5 | `volume` | shares | 1-minute volume |

### 3.2 Options Greeks & Chain State (9)

| # | Column | Range | Purpose |
|---|--------|-------|---------|
| 6 | `delta` | [-1, 1] | Option delta |
| 7 | `gamma` | [0, +) | Option gamma |
| 8 | `vega` | [0, +) | Option vega |
| 9 | `theta` | (-, 0] | Option theta (time decay) |
| 10 | `iv` | [0, 2] | Implied volatility |
| 11 | `ivr` | [0, 1] | IV Rank (percentile) |
| 12 | `spread_ratio` | [0, +) | (ask-bid)/mid |
| 13 | `te` | days | Time to expiration |
| 14 | `strike` | SPY $ | Strike price |

### 3.3 Strategy / Risk (2)

| # | Column | Range | Purpose |
|---|--------|-------|---------|
| 15 | `target_spot` | SPY $ | IC entry target price |
| 16 | `max_dd_60m` | pct | Max drawdown (60m lookback) |

### 3.4 V2.1 Dynamic Features - Kinematics (6)

| # | Column | Formula | Range | Purpose |
|---|--------|---------|-------|---------|
| 17 | `log_return` | ln(close_t/close_t-1) | centered | 1m log return |
| 18 | `vol_ewma` | EWMA(ret^2, span=64) | [0, +) | Realized vol proxy |
| 19 | `ret_z` | log_return / (vol_ewma + eps) | centered | Vol-normalized return |
| 20 | `atr_pct` | ATR / close | [0, 0.05] | ATR as % of price |
| 21 | `kappa_proxy` | 2nd_diff(log_ret) / scale | centered | Price curvature |
| 22 | `vol_energy` | log(1 + alpha*|kappa|) | [0, +) | Vol energy modulator |

### 3.5 V2.1 Dynamic Features - Oscillators (3)

| # | Column | Formula | Range | Purpose |
|---|--------|---------|-------|---------|
| 23 | `rsi_dyn` | Curvature-weighted RSI | [0, 100] | Vol-weighted momentum |
| 24 | `adx_adaptive` | ADX * (1 + vol_energy) | [0, 100] | Vol-adaptive trend |
| 25 | `psar_adaptive` | (price - SAR) / (ATR * price) | centered | ATR-scaled PSAR |

### 3.6 V2.1 Dynamic Features - Bollinger Bands (4)

| # | Column | Formula | Range | Purpose |
|---|--------|---------|-------|---------|
| 26 | `bb_mu_dyn` | Rolling mean(close) | SPY $ | Band center |
| 27 | `bb_sigma_dyn` | Rolling std * (1 + vol_energy) | [0, +) | Dynamic band width |
| 28 | `bb_lower_dyn` | bb_mu - 2*bb_sigma | SPY $ | Lower band |
| 29 | `bb_upper_dyn` | bb_mu + 2*bb_sigma | SPY $ | Upper band |

### 3.7 V2.1 Dynamic Features - Regime (3)

| # | Column | Formula | Range | Purpose |
|---|--------|---------|-------|---------|
| 30 | `stoch_k_dyn` | Stoch %K / (1 + vol_energy) | [0, 100] | Vol-compressed oscillator |
| 31 | `consolidation_score` | 1/(1 + range/mean_bw) | [0, 1] | Consolidation strength |
| 32 | `breakout_score` | Band break detection | {-1, 0, +1} | Breakout direction |

---

## 4. V2.2 PRIMITIVE OUTPUTS (22 columns)

### 4.1 P002: Bandwidth Metrics (2)

| # | Column | Source | Range | Purpose |
|---|--------|--------|-------|---------|
| 33 | `bb_percentile` | bands.py | [0, 100] | Bandwidth percentile rank |
| 34 | `bw_expansion_rate` | bands.py | centered | (curr_bw - prev_bw) / prev_bw |

### 4.2 P003: Money Flow & Pressure (3)
*Replaces deprecated volume_ratio, bid, ask*

| # | Column | Source | Range | Purpose |
|---|--------|--------|-------|---------|
| 35 | `cmf` | bands.py | [-1, 1] | Chaikin Money Flow |
| 36 | `pressure_up` | bands.py | [0, 1] | max(0, close-open)/(high-low) |
| 37 | `pressure_down` | bands.py | [0, 1] | max(0, open-close)/(high-low) |

### 4.3 P004: Execution Gate (2)

| # | Column | Source | Range | Purpose |
|---|--------|--------|-------|---------|
| 38 | `friction_ratio` | bands.py | [0, 1] | Spread / avg_range |
| 39 | `exec_allow` | bands.py | {0, 1} | Binary friction gate |

### 4.4 P005: Gap Risk (2)

| # | Column | Source | Range | Purpose |
|---|--------|--------|-------|---------|
| 40 | `gap_risk_score` | bands.py | [0, 1] | Composite gap risk |
| 41 | `risk_override` | bands.py | {0, 1} | Gap risk override flag |

### 4.5 P006: IV Confidence (1)

| # | Column | Source | Range | Purpose |
|---|--------|--------|-------|---------|
| 42 | `iv_confidence` | bands.py | [0, 1] | exp(-lambda * lag_minutes) |

### 4.6 P007: Multi-Timeframe (1)

| # | Column | Source | Range | Purpose |
|---|--------|--------|-------|---------|
| 43 | `mtf_consensus` | bands.py | centered | Weighted avg of 1m/5m/15m |

### 4.7 M001: Volatility-Adaptive MACD (3)

| # | Column | Source | Range | Purpose |
|---|--------|--------|-------|---------|
| 44 | `macd_norm` | momentum.py | centered | MACD / vol_ewma |
| 45 | `macd_signal_norm` | momentum.py | centered | Signal / vol_ewma |
| 46 | `macd_histogram` | momentum.py | centered | macd_norm - signal_norm |

### 4.8 M002: Directional Indicators (2)

| # | Column | Source | Range | Purpose |
|---|--------|--------|-------|---------|
| 47 | `plus_di` | dynamic_features.py | [0, 100] | +DI (uptrend strength) |
| 48 | `minus_di` | dynamic_features.py | [0, 100] | -DI (downtrend strength) |

### 4.9 M004: PSAR Metrics (2)

| # | Column | Source | Range | Purpose |
|---|--------|--------|-------|---------|
| 49 | `psar_trend` | signals.py | {-1, 0, +1} | PSAR trend direction |
| 50 | `psar_reversion_mu` | momentum.py | [0, 1] | PSAR reversion membership |

### 4.10 T001-T002: Topology (3)

| # | Column | Source | Range | Purpose |
|---|--------|--------|-------|---------|
| 51 | `beta1_norm_stub` | topology.py | 0.0 | TDA stub (V2.3) |
| 52 | `chaos_membership` | topology.py | [0, 1] | Chaos regime membership |
| 53 | `position_size_mult` | topology.py | [0, 1] | 1 - chaos_membership |

### 4.11 F001: Fuzzy (1)

| # | Column | Source | Range | Purpose |
|---|--------|--------|-------|---------|
| 54 | `fuzzy_reversion_11` | fuzzy.py | [0, 1] | 11-factor fuzzy score |

---

## 5. LOCAL FILE ISSUES DETECTED

The local files in `data/processed/` have schema problems:

### 5.1 Merge Artifacts (`_x` / `_y` suffixes)
```
ret_z_x, ret_z_y
atr_pct_x, atr_pct_y
kappa_proxy_x, kappa_proxy_y
vol_energy_x, vol_energy_y
psar_adaptive_x, psar_adaptive_y
bb_mu_dyn_x, bb_mu_dyn_y
bb_sigma_dyn_x, bb_sigma_dyn_y
bb_lower_dyn_x, bb_lower_dyn_y
bb_upper_dyn_x, bb_upper_dyn_y
stoch_k_dyn_x, stoch_k_dyn_y
consolidation_score_x, consolidation_score_y
breakout_score_x, breakout_score_y
```

**Fix**: Choose correct column (`_y` typically) and drop the duplicate.

### 5.2 Deprecated Columns Still Present
```
bid       -> should be pressure_up
ask       -> should be pressure_down
volume_ratio -> should be cmf
```

**Fix**: Run V2.2 primitive computation pipeline.

### 5.3 Missing V2.2 Primitives
Local files lack:
- `cmf`
- `pressure_up`
- `pressure_down`

**Fix**: Run `data_factory/pipeline/compute_v22_primitives.py`

---

## 6. CANONICAL SELECTION PATTERN

```python
from intelligence.canonical_feature_registry import (
    FEATURE_COLS_V22,
    select_feature_frame,
    assert_schema_match_cols,
)

# ALWAYS select by name, never by position
df_features = select_feature_frame(df, FEATURE_COLS_V22, strict=True)

# Validate before training/inference
assert_schema_match_cols(df.columns.tolist(), FEATURE_COLS_V22, strict=True)
```

---

## 7. COLUMN COUNT RECONCILIATION

| Source | V2.1 Base | V2.2 Primitives | Total Model | Metadata | Legacy | CSV Total |
|--------|-----------|-----------------|-------------|----------|--------|-----------|
| Canonical Registry | 32 | 22 | 54 | - | - | - |
| Lightning AI CSV | 32 | 22 | 54 | 7 | 4 | 65 |
| Local CSV (broken) | 32 (duped) | partial | ~45 | 9 | 4 | ~75 |

---

## 8. FILES REQUIRING UPDATE

1. **`intelligence/canonical_feature_registry.py`** - Source of truth (OK)
2. **`scripts/precompute_features.py`** - Must output V2.2 columns
3. **`data_factory/pipeline/finalize_and_verify.py`** - Schema validation
4. **`intelligence/train_condor_brain.py`** - Feature selection
5. **`intelligence/condor_brain.py`** - Inference feature selection

---

*Generated: 2026-01-29*
*Schema Version: V2.2 (54 features)*
