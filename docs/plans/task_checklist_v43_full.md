# CondorNet v4.3 — Full Task Checklist
<!-- schema_version: v4.3.0 | date: 2026-02-22 -->
<!-- Supersedes: task_checklist_v43.md -->

Legend: ✅ = complete | [ ] = pending | 🔄 = in progress

---

## TRACK 0 — Dataset Validation & ETL (NEW)

### F1: schema_v43.py — V2.2 Schema Constants & Validators
- [ ] F1.1 — Define `TF_FEATURE_NAMES` list (64 input features, excluding target_spot + position_size_mult)
- [ ] F1.2 — Define `TF_LABEL_NAMES` = ["target_spot", "position_size_mult"]
- [ ] F1.3 — Define `TF_PIVOT_FEATURES` list (13 sparse pivot columns)
- [ ] F1.4 — Define `TF_SPARSE_FEATURES` set (never-impute columns)
- [ ] F1.5 — Define `TF_BOUNDED_FEATURES` dict with (min, max) per column
- [ ] F1.6 — Define `TF_TERNARY_FEATURES` = ["breakout_score", "psar_trend"]
- [ ] F1.7 — Define `TF_DTYPE_MAP` dict column → expected dtype
- [ ] F1.8 — Define `KNOWN_CROSS_TF_FLAGS` dict (TYPE_MISMATCH, RANGE_MISMATCH, SPARSITY_MISMATCH, CONST_MISMATCH)
- [ ] F1.9 — Define `CHAIN_FEATURE_NAMES` list (10 encoded features per contract)
- [ ] F1.10 — Define `CHAIN_LABEL_NAMES` list (pop, ev, max_profit, max_loss, var_95, cvar_95)
- [ ] F1.11 — Implement `validate_tf_dataframe(df, tf_name) → ValidationReport`
- [ ] F1.12 — Implement `validate_options_chain(df) → ValidationReport`
- [ ] F1.13 — Set `SCHEMA_VERSION = "v4.3.0"`
- [ ] F1.14 — Define `CANONICAL_TF = "m5"` + canonical timestamp contract rules (drop T failing any condition)
- [ ] F1.15 — Define `SPOT_PRICE_RULE = "M5 close at timestamp T"` constant
- [ ] F1.16 — Define `CHAIN_GRID_CONFIG` dict (n_strikes=20, n_expiries=3, max_contracts=120, liquidity_min_oi=100, pad rules)
- [ ] F1.17 — Define `FRICTION_CANDIDATE_WINDOWS = [5, 10, 20, 40, 60]`
- [ ] F1.18 — Define `strategy_json_version = "1.0"` + required/optional field schema
- [ ] F1.19 — Define `ABSTAIN_CONFIDENCE_THRESHOLD = 0.60` + `ABSTAIN_LABEL_SCORE_CUTOFF = 0.40`
- [ ] F1.20 — Define `DTE_AFFINITY` matrix dict (10 types × 4 DTE buckets)
- [ ] F1.21 — Define `MIN_STRIKE_INCREMENT = 0.50` (SPY)
- [ ] F1.22 — Define `PIVOT_MASK_CONTRACT = "four_separate"` (not stacked)
- [ ] F1.23 — Define `TOD_TRADING_MINUTES = 390` (9:30am–4:00pm)

### F2: data_pipeline_v43.py — ETL + Temporal Alignment
- [ ] F2.1 — `load_tf_dataset(path, tf_name)` — reads CSV, validates schema, returns DataFrame
- [ ] F2.2 — `align_timestamps(m1_df, m5_df, m15_df, h1_df)` — no look-ahead, M5 as master clock
- [ ] F2.3 — `normalize_features(df)` — RobustScaler for continuous; passthrough for bounded/ternary; mask for sparse
- [ ] F2.4 — `build_pivot_mask(df)` — [rows, 13] bool DataFrame marking NaN pivot positions
- [ ] F2.5 — `load_options_chain(path)` — chunked load of 310 MB CSV; parse timestamps
- [ ] F2.6 — `compute_chain_derived_fields(df)` — add moneyness=log(S/K), days_to_exp, width_of_mid
- [ ] F2.7 — `validate_options_chain(df)` — bid≤ask, delta∈[-1,1], gamma≥0, no duplicate (ts, contract_id)
- [ ] F2.8 — `create_chain_snapshot(chain_df, timestamp)` — O(1) dict lookup after groupby(timestamp)
- [ ] F2.9 — `V43Dataset.__getitem__` — returns all 4 TF tensors + chain snapshot + pivot mask + labels
- [ ] F2.10 — Handle CROSS_TF anomalies: friction_ratio cast, WeightedAlpha/Slope range normalization
- [ ] F2.11 — Ensure all timestamps are tz-naive (tz_localize(None)) before alignment
- [ ] F2.12 — Train/val/test split: 70/15/15 by date (not random shuffle — temporal integrity)
- [ ] F2.13 — `align_timestamps()`: enforce canonical contract — drop any T failing any TF or chain condition
- [ ] F2.14 — `fit_and_save_normalizer(train_df, path)` — RobustScaler on train only, save pkl, idempotency guard
- [ ] F2.15 — `load_normalizer(path)` — load and apply at inference; raise RuntimeError if not yet fit
- [ ] F2.16 — `compute_friction_features(df)` — 5 binary columns `friction_ok_5/10/20/40/60` (replaces dead friction_ratio)
- [ ] F2.17 — `compute_tod_features(df)` — `tod_sin`, `tod_cos` from minute_of_day / 390
- [ ] F2.18 — `compute_regime_persistence(df)` — consecutive bars in same (vol_bucket, trend_bucket)

---

## TRACK 1 — Options Chain Encoder

### F3: options_chain_encoder.py — Transformer Chain Encoder
- [ ] F3.1 — `build_chain_grid(chain_snapshot, spot, n_strikes=20, n_expiries=3)` → Tensor[N,10]
- [ ] F3.2 — `OptionsChainEncoder.__init__(in_features=10, d_model=64, n_heads=4, n_layers=2, d_chain=128)`
- [ ] F3.3 — Linear projection layer: 10 → d_model
- [ ] F3.4 — Moneyness-based positional encoding (sort contracts by moneyness)
- [ ] F3.5 — Transformer encoder (2 layers, 4 heads, d_ff=256)
- [ ] F3.6 — Mean pooling over N_contracts dimension → [B, d_model]
- [ ] F3.7 — Output projection → [B, d_chain=128]
- [ ] F3.8 — Handle padding mask for missing contracts (key_padding_mask)
- [ ] F3.9 — Handle empty chain (all-zero fallback with warning log)
- [ ] F3.10 — Target label generation integration (`target_labeler_v43.py`)
- [ ] F3.11 — Apply `liquidity_min_oi=100` mask: OI < 100 → mask=True in chain_mask (not dropped)
- [ ] F3.12 — Extract skew signal: `skew = put_iv_25d - call_iv_25d` appended to chain encoder output
- [ ] F3.13 — Chain encoder diagnostics hook (E1): contract count, expiry/strike/IV distributions per batch

## TRACK 2 — Options Strategy Universe

### F4: strategy_generator.py — 9-Strategy Grammar
- [ ] F4.1 — `Strategy` dataclass: legs, strategy_type, net_credit, breakevens
- [ ] F4.2 — `Leg` dataclass: strike, expiry, type (call/put), long_short, qty, greeks
- [ ] F4.3 — `build_single_call(chain, spot, target_delta, dte)` → Strategy
- [ ] F4.4 — `build_single_put(chain, spot, target_delta, dte)` → Strategy
- [ ] F4.5 — `build_bull_call_spread(chain, spot, short_delta, wing_width, dte)` → Strategy
- [ ] F4.6 — `build_bear_put_spread(chain, spot, short_delta, wing_width, dte)` → Strategy
- [ ] F4.7 — `build_straddle(chain, spot, dte)` → Strategy
- [ ] F4.8 — `build_strangle(chain, spot, target_delta, dte)` → Strategy
- [ ] F4.9 — `build_butterfly_call(chain, spot, target_delta, wing_width, dte)` → Strategy
- [ ] F4.10 — `build_iron_condor(chain, spot, short_delta, wing_width, dte)` → Strategy (mirrors existing build_condor)
- [ ] F4.11 — `build_custom_multi_leg(chain, spot, leg_specs: list)` → Strategy (model-driven)
- [ ] F4.12 — `StrategyGenerator.sample(chain_snapshot, spot, n_candidates)` — sample N from universe
- [ ] F4.13 — Fail-fast validation: all 4 legs must have valid bid/ask for iron condor (same rule as existing code)
- [ ] F4.14 — Leg serialization: `Strategy.to_json()` and `Strategy.from_json()`
- [ ] F4.15 — Strategy ID: `sha256(timestamp|type|sorted_legs)[:16]` — deterministic, globally consistent
- [ ] F4.16 — Per-leg friction gate: each leg's `spread < HL_N` for at least one N in candidate windows
- [ ] F4.17 — `build_abstain()` → Strategy with strategy_type="abstain", all risk metrics None
- [ ] F4.18 — Apply `DTE_AFFINITY` prior score to each candidate strategy at generation time
- [ ] F4.19 — Strategy JSON validation: assert all required fields present before serialize/save
- [ ] F4.20 — `StrategyGenerator.compare_all(chain, spot, dte)` → ranked list by composite score

---

## TRACK 3 — Payoff Calculator

### F5: payoff_calculator.py — PoP / EV / VaR
- [ ] F5.1 — `payoff_at_expiry(strategy, S_T)` → float: net payoff for given underlying price
- [ ] F5.2 — `compute_breakevens(strategy, spot)` → list[float]: underlying prices where payoff=0
- [ ] F5.3 — `compute_max_profit_loss(strategy)` → tuple[float, float]
- [ ] F5.4 — `compute_pop(strategy, spot, iv, r, tau)` → float via BS normal CDF (N(d2_hi) - N(d2_lo))
- [ ] F5.5 — `compute_ev(strategy, spot, iv, r, tau, n_mc=10_000)` → float via MC or analytic BS
- [ ] F5.6 — `compute_net_greeks(strategy)` → NetGreeks dataclass (delta, gamma, theta, vega, rho)
- [ ] F5.7 — `compute_var_cvar(strategy, spot, iv, r, tau, confidence=0.95)` → tuple[float, float]
- [ ] F5.8 — Validate: PoP ∈ [0,1], EV finite, VaR ≥ 0, CVaR ≥ VaR
- [ ] F5.9 — Unit tests: verify Iron Condor PoP ≈ expected values for known IV/spot scenarios
- [ ] F5.10 — Per-leg IV: always use `leg.implied_volatility`; fallback to M5 `vol_ewma` with warning log
- [ ] F5.11 — `compute_pin_risk(strategy, spot)` → bool, True when spot within 0.5 strikes of short leg with DTE≤2
- [ ] F5.12 — `compute_net_greeks()` validation: net_delta within regime-appropriate bounds

---

## TRACK 4 — Target Label Generation

### F6: target_labeler_v43.py — Training Label Generation
- [ ] F6.1 — Load all 4 TF datasets and options chain
- [ ] F6.2 — For each M5 timestamp, generate strategy candidates via StrategyGenerator
- [ ] F6.3 — Compute PoP, EV, max_profit, max_loss, var_95, cvar_95 for each candidate
- [ ] F6.4 — Identify `is_ideal = argmax(pop - 0.5 * cvar_95 / max_loss)` per timestamp
- [ ] F6.5 — Save to `data/Datasetv4/v43/labels_v43.parquet` (schema: timestamp, strategy_id, strategy_type, legs_json, pop, ev, max_profit, max_loss, var_95, cvar_95, net_delta, net_gamma, net_theta, net_vega, is_ideal)
- [ ] F6.6 — Validate label distribution: PoP histogram, strategy type frequency, EV distribution
- [ ] F6.7 — Log label generation summary (n_timestamps, n_strategies_per_ts, % valid labels)
- [ ] F6.8 — Generate abstain labels: when best candidate score < 0.40, label = abstain
- [ ] F6.9 — Add `dte_affinity_score` column to labels parquet
- [ ] F6.10 — Add `pin_risk_flag` column to labels parquet
- [ ] F6.11 — Add `friction_gate_ok` column to labels parquet (True only when ALL legs pass)

---

## TRACK 5 — Model Architecture Updates

### U2: condor_brain_net.py — New Architecture Components
- [ ] U2.1 — Add `MultiTFProjector(nn.Module)` class: 4 × [B,T,64] → [B,T,256]
- [ ] U2.2 — Add integration hook for `OptionsChainEncoder` output into joint representation
- [ ] U2.3 — Add `StrategyHead(nn.Module)`: strategy_logits [B,9] + leg_params [B,4,5] + entry_signal [B,1]
- [ ] U2.4 — Add `RiskMetricHead(nn.Module)`: pop [B,1] + ev [B,1] + max_loss [B,1] + var_95 [B,1] + cvar_95 [B,1]
- [ ] U2.5 — Update `CondorNet.__init__` with new params (input_dim, chain_dim, d_joint, n_strategy_types, max_legs)
- [ ] U2.6 — Update `CondorNet.forward` to accept (x_m1, x_m5, x_m15, x_h1, chain, chain_mask, pivot_features)
- [ ] U2.7 — Add backward-compat wrapper: single `x` arg routes to all 4 TFs (for existing backtest engine)
- [ ] U2.8 — Add `CondorNetOutput` dataclass: all outputs in a single typed return
- [ ] U2.9 — `PivotProjector(nn.Module)`: Linear(13→16) + ReLU; output fused with TF before chain concat
- [ ] U2.10 — `TFFusionBlock`: Linear(272→256) + LayerNorm after cat(TF[256], PivotProj[16])
- [ ] U2.11 — Chain embed broadcast: `chain_embed.unsqueeze(1).expand(-1, T, -1)` then cat → [B,T,384]
- [ ] U2.12 — `MultiTFContributionRatios` hook: log Frobenius norms of M1/M5/M15/H1 projections per forward

### U3: condor_brain.py — Facade Updates
- [ ] U3.1 — Update `CondorSignal` dataclass: add strategy_type, strategy_legs, pop, ev, max_loss, var_95, cvar_95
- [ ] U3.2 — Update `CondorBrain.__init__` to accept multi-TF input config and chain_dim
- [ ] U3.3 — Update `CondorBrain.forward` signature to accept 4-TF input + chain
- [ ] U3.4 — Update `CondorBrainEngine.predict` to return full CondorSignal (including new risk fields)
- [ ] U3.5 — Preserve backward compat: existing single-TF callers still work via routing wrapper
- [ ] U3.6 — `CondorBrainEngine` inference chain snapshot builder: raw chain DataFrame → build_chain_grid()
- [ ] U3.7 — `CondorBrainEngine` inference strategy constructor: leg_params → real strikes/expiries
- [ ] U3.8 — Chain encoder inference cache: key=(date, bar_time, round(spot,2)), invalidate on new bar

### U4: fuzzy_engine.py — PoP-Based Sizing
- [ ] U4.1 — Define linguistic terms: POP_VERY_HIGH, POP_HIGH, POP_MEDIUM, POP_LOW, POP_VERY_LOW (triangular MFs)
- [ ] U4.2 — Define vol linguistic terms: VOL_LOW, VOL_MED, VOL_HIGH (using atr_pct)
- [ ] U4.3 — Implement `compute_pop_based_sizing(pop, ev, var_95, atr_pct, base_size)` → float [0,1]
- [ ] U4.4 — Implement fuzzy rule base (9 rules from §6.2 of master plan)
- [ ] U4.5 — Add `pop_sizing_weight` parameter to `compute_position_size` (default 0.5)
- [ ] U4.6 — Update docstring with linguistic term definitions and rule base

### U5: canonical_feature_registry.py — V2.2 Sync
- [ ] U5.1 — Verify all 64 TF_FEATURE_NAMES present in registry
- [ ] U5.2 — Add/update entries for all Group 7 pivot features (13 columns)
- [ ] U5.3 — Add options chain feature catalog entries (CHAIN_FEATURE_NAMES)
- [ ] U5.4 — Flag `friction_ratio` TYPE_MISMATCH handling in registry
- [ ] U5.5 — Add `schema_version = "v4.3.0"` constant

### U6: core/config.py — v43 Config Fields
- [ ] U6.1 — Add to `RunConfig`: `v43_data_dir`, `options_chain_path`, `labels_path`, `schema_version = "v4.3.0"`
- [ ] U6.2 — Add to `StrategyConfig`: `strategy_universe` (list of enabled types), `max_legs = 4`, `pop_target_min = 0.55`
- [ ] U6.3 — Add to `StrategyConfig`: `pop_sizing_weight = 0.5`, `ev_weight = 0.15`, `var_weight = 0.15`
- [ ] U6.4 — Add to `RunConfig`: `chain_n_strikes = 20`, `chain_n_expiries = 3`, `d_chain = 128`
- [ ] U6.5 — Update `config.template.py` to match
- [ ] U6.6 — Add `risk_free_rate: float = 0.05` to RunConfig
- [ ] U6.7 — Add `grad_accum_steps: int = 4` to RunConfig
- [ ] U6.8 — Add `chain_liquidity_min_oi: int = 100` to RunConfig
- [ ] U6.9 — Add `abstain_confidence_threshold: float = 0.60` to StrategyConfig
- [ ] U6.10 — Add `abstain_label_score_cutoff: float = 0.40` to StrategyConfig

### U7: configs/loss_weights_v43.json — Loss Weight Schedule
- [ ] U7.1 — Create JSON with initial 9-component weights (strategy_ce=1.0, pop_bce=0.8, ev_mse=0.5, risk_mse=0.5, size_mse=0.4, spot_mse=0.3, fuzzy_var=0.2, pattern_ent=0.1, robust=0.2)
- [ ] U7.2 — Implement linear annealing loader in training loop (pop_bce→1.2 by ep10, ev_mse→0.8 by ep20, fuzzy_var→0.05 by ep30)

### U8: tests/test_backward_compat_v42.py — Backward Compatibility
- [ ] U8.1 — Test `CondorBrain(x)` single-tensor call (old v42 signature) still runs without exception
- [ ] U8.2 — Test `CondorBrainEngine.predict(features_dict)` still returns valid CondorSignal
- [ ] U8.3 — Test all v42 CondorSignal fields (direction, confidence, size) still present in v43 output

---

## TRACK 6 — Training Loop v4.3

### F7: condor_train_net_v43.py — Full Training Loop
- [ ] F7.1 — Argument parser: --data-dir, --labels, --d-chain, --n-strategy-types, and all existing v42 flags
- [ ] F7.2 — `V43Dataset` class with `__getitem__` returning all 4 TF tensors + chain + mask + labels
- [ ] F7.3 — Multi-input DataLoader with custom collate_fn for ragged chain grids
- [ ] F7.4 — Model instantiation: `CondorNet(input_dim=256, chain_dim=128, ...)`
- [ ] F7.5 — `CondorLossV43` class with 9-component loss (strategy CE + PoP BCE + EV MSE + risk MSE + size MSE + spot MSE + fuzzy_var + pattern_ent + robust)
- [ ] F7.6 — Training loop with all 7-step hook order (forward→loss→backward→step→observe→checkpoint→emit)
- [ ] F7.7 — Validation loop with loss logging and PoP calibration check (ECE)
- [ ] F7.8 — `--resume` flag: load checkpoint and continue from correct epoch/batch
- [ ] F7.9 — Crash sentinel: write `.training_active` on start, delete on clean exit
- [ ] F7.10 — Load `configs/loss_weights_v43.json`; implement linear weight annealing per epoch
- [ ] F7.11 — `collate_fn`: right-pad chain grids to batch max N_contracts, generate `chain_mask` BoolTensor
- [ ] F7.12 — Gradient accumulation: `loss /= grad_accum_steps`, zero_grad every N steps

### F8: training/training_hooks_v43.py — All Track A–D Hooks
- [ ] F8.1 — `save_convergence_checkpoint(model, optimizer, epoch, batch, loss, config)` (A1.4)
- [ ] F8.2 — `gpu_memory_autotuner(model, dataloader, start_bs=256)` (A2.4)
- [ ] F8.3 — `amp_scaler_diagnostics(scaler)` → log scale factor + overflow (A2.5)
- [ ] F8.4 — `cpu_bottleneck_profiler(dataloader_time, compute_time)` (A2.6)
- [ ] F8.5 — `DataLoaderHealthMonitor` class: shape checks, NaN/Inf, timing, worker crash (A2.7)
- [ ] F8.6 — `auto_resume_from_checkpoint(ckpt_dir)` → loads full state (A3.1, A3.2)
- [ ] F8.7 — `resume_safe_logger(log_path)` → appends to existing log (A3.3)
- [ ] F8.8 — `write_run_metadata(config, model, dataset)` → `reports/run_metadata.json` (A4.1)
- [ ] F8.9 — `write_run_summary(results)` → `reports/run_summary.json` (A5.1)
- [ ] F8.10 — `deep_observe(model, outputs, loss, batch_idx, epoch)` → 7-section JSON (B1.F, D0.1)
- [ ] F8.11 — B-matrix influence ratio: ‖Bu‖ / ‖Ax‖ (B1.7)
- [ ] F8.12 — Gradient norms per module (B1.9)
- [ ] F8.13 — Param drift deltas: compare state_dict snapshots (B1.10)
- [ ] F8.14 — Predicate taste testing: fired/ignored/reinforced/rejected per batch (B2.1)
- [ ] F8.15 — Per-predicate sentiment: correlate activation with loss Δ (B2.2)
- [ ] F8.16 — Set/superset loss correlation (B2.3)
- [ ] F8.17 — Rejected logic trace: predicates gated out by superset < 0.1 (B2.4)
- [ ] F8.18 — Set membership distributions (B2.5)
- [ ] F8.19 — Predicate Journal JSONL writer (B2.6)
- [ ] F8.20 — Predicate pruning rules (B2.7)
- [ ] F8.21 — Predicate merge on resume (B2.8)
- [ ] F8.22 — Pivot proximity per batch (B3.1)
- [ ] F8.23 — Pivot influence on predicates/sets/loss (B3.2)
- [ ] F8.24 — Reversal specialization tracking (B3.3)
- [ ] F8.25 — Pivot influence heatmap generator (B3.4)
- [ ] F8.26 — Reasoning trace chain: features→predicates→sets→superset→direction (B4.1)
- [ ] F8.27 — Predicate explanation engine (B4.2)
- [ ] F8.28 — Memory update logs: ‖x_k - x_{k-1}‖ per block (B4.3)
- [ ] F8.29 — Memory drift tracking: spectral radius Δ + hidden state norm (B4.4)
- [ ] F8.30 — Logic chain graph persistence (edge list) (B4.5)
- [ ] F8.31 — Logic chain graph pruning (B4.6)
- [ ] F8.32 — Logic chain graph visualization hooks (B4.7)
- [ ] F8.33 — TFT contribution ratio: ‖u_k‖/‖x_k‖ (C1.1)
- [ ] F8.34 — CDE contribution ratio: ‖G·dX‖/‖x_k‖ (C1.2)
- [ ] F8.35 — ETD drift contribution: ‖F·x_{k-1}‖/‖x_k‖ (C1.3)
- [ ] F8.36 — Forcing contribution: ‖D‖/‖x_k‖ (C1.4)
- [ ] F8.37 — Winner subsystem per batch (C1.5)
- [ ] F8.38 — Attribution normalization (clipping 1e-8, zero-norm fallback, seq aggregation) (C1.6)
- [ ] F8.39 — TFT variable selection weights (C2.1)
- [ ] F8.40 — CDE signal absorption by feature dim (C2.2)
- [ ] F8.41 — ETD memory retention: cos(x_0, x_T) (C2.3)
- [ ] F8.42 — Predicate→subsystem gradient influence (C2.4)
- [ ] F8.43 — Disagreement index (sign disagreement between F·x, B·u, G·dX) (C3.1)
- [ ] F8.44 — Conflict logs (C3.2)
- [ ] F8.45 — Resolution mechanism trace (C3.3)
- [ ] F8.46 — Parameter drift per module (C4.1)
- [ ] F8.47 — Activation drift running mean/var (C4.2)
- [ ] F8.48 — Specialization drift: contribution ratios per epoch (C4.4)
- [ ] F8.49 — Drift anomaly detection + alerts (C4.5)
- [ ] F8.50 — Specialization drift visualization (C4.6)
- [ ] F8.51 — PyTorch profiler instrumentation (D1.1)
- [ ] F8.52 — Training speed logs: wall-clock per batch/epoch, peak memory (D1.2)
- [ ] F8.53 — TensorBoard logging (D2.2)
- [ ] F8.54 — Epoch-level metric summaries (D3.1)
- [ ] F8.55 — Combined interpretability report (D3.2)
- [ ] F8.56 — GUI telemetry stream (D4.1)
- [ ] F8.57 — Persistent logging across sessions (D4.2)
- [ ] F8.58 — Batch replay tool: replay forward pass for any batch index (D5.1)
- [ ] F8.59 — Model DNA export: `model_dna.json` (D6.1)
- [ ] F8.60 — Validation deep-observe mode (--deep-observe-val) (D0.2)
- [ ] F8.61 — Strategy distribution drift telemetry (E2): type frequency per epoch + >20% drift alert
- [ ] F8.62 — Chain encoder diagnostics (E1): contract count, IV/strike/expiry histograms per batch
- [ ] F8.63 — Multi-TF contribution ratios (E3): M1/M5/M15/H1 Frobenius norm ratios per batch

---

## Previously Completed (from task_checklist_v43.md)

- ✅ A1.1 — Default save dir → `models/ckpts/`
- ✅ A1.2 — `--checkpoint-dir` CLI flag
- ✅ A1.3 — Atomic save (tmp → validate → rename)
- ✅ A1.5 — Rotate last N checkpoints (`--keep-ckpts`)
- ✅ A2.1 — CLI: `--num-workers`, `--prefetch-factor`, `--persistent-workers`
- ✅ A2.2 — DataLoader hardcoded (workers=4, pin_memory=True, persistent=True, prefetch=2)
- ✅ A2.3 — Duplicate DataLoader block removed
- ✅ B1.1 — Predicate activation map (diag dict)
- ✅ B1.3 — SuperSet routing (diag dict)
- ✅ B1.4 — Fuzzy gate variance (in loss logs)
- ✅ B1.5 — Pattern entropy (in loss logs)
- ✅ B1.6 — A-matrix spectral radius (diag dict)
- ✅ B1.8 — State norms h/v/m/r (diag dict)

---

## Execution Priority Order

### Phase 0 — Schema & Validation Foundation (Do First)
1. [ ] F1.1–F1.13 (schema_v43.py)
2. [ ] F2.1–F2.4 (ETL: TF loading + alignment)
3. [ ] U5.1–U5.5 (canonical_feature_registry.py sync)
4. [ ] U6.1–U6.5 (config.py v43 fields)

### Phase 1 — Options Chain & Strategy Engine
5. [ ] F2.5–F2.11 (ETL: options chain)
6. [ ] F3.1–F3.10 (options_chain_encoder.py)
7. [ ] F4.1–F4.14 (strategy_generator.py)
8. [ ] F5.1–F5.9 (payoff_calculator.py)

### Phase 2 — Target Label Generation (Lightning AI job)
9. [ ] F6.1–F6.7 (target_labeler_v43.py)

### Phase 3 — Architecture Upgrades
10. [ ] U2.1–U2.8 (condor_brain_net.py)
11. [ ] U3.1–U3.5 (condor_brain.py)
12. [ ] U4.1–U4.6 (fuzzy_engine.py)

### Phase 4 — Training Loop & Stability Hooks
13. [ ] F7.1–F7.9 (condor_train_net_v43.py)
14. [ ] F8.1–F8.9 (stability & crash recovery hooks)

### Phase 5 — Observability & Telemetry
15. [ ] F8.10–F8.32 (deep observe + predicate intelligence)
16. [ ] F8.33–F8.50 (subsystem attribution & drift)

### Phase 6 — Reporting & Integration
17. [ ] F8.51–F8.60 (profiling, telemetry, DNA, replay)
