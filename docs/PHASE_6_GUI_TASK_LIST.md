# Phase 6: CondorBrain GUI - Complete Task List (Revised)

**Created:** 2026-02-06
**Revised:** 2026-02-06
**Status:** APPROVED FOR IMPLEMENTATION
**Technology Stack:** FastAPI (Backend) + React/TypeScript/TailwindCSS/Zustand (Frontend)

---

## Overview

This document contains the complete, step-by-step task list for building the CondorBrain/CondorNet GUI from first line of code to production-ready deployment.

**Total Phases:** 14 sub-phases (6.0 - 6.13)
**Total Tasks:** 280+
**Architecture:** Monorepo with `gui/backend/` and `gui/frontend/`

---

## Revised Phase Order

```
6.0  → Scaffolding (CRITICAL)
6.1  → Core Infrastructure (CRITICAL)
6.2  → Dashboard (CRITICAL) ← Moved early for developer feedback
6.3  → Intelligence Control Matrix (HIGH)
6.4  → Execution Reality Engine (HIGH)
6.5  → Backtest Control Panel (CRITICAL)
6.6  → Tape Viewer (MEDIUM)
6.7  → Model Introspection (MEDIUM)
6.8  → Trade Explorer (MEDIUM)
6.9  → Optimization Suite (HIGH)
6.10 → Polish & Testing (MEDIUM)
6.11 → Deployment Readiness (HIGH) ← Moved to final gate
6.12 → Performance & GPU Optimization (HIGH) ← NEW
6.13 → Security & Access Control (MEDIUM) ← NEW
```

---

## PHASE 6.0 — Project Scaffolding (CRITICAL)

**Duration:** 1 day
**Dependencies:** None
**Deliverable:** Empty but structured project ready for development

### 6.0.1 Directory Structure
- [ ] Create `gui/` directory at project root
- [ ] Create `gui/backend/` for FastAPI server
- [ ] Create `gui/frontend/` for React application
- [ ] Create `gui/shared/` for shared types/schemas
- [ ] Create `gui/scripts/` for build/deploy scripts

### 6.0.2 Backend Scaffolding
- [ ] Initialize FastAPI project in `gui/backend/`
- [ ] Create `gui/backend/requirements.txt`
- [ ] Create `gui/backend/main.py` (entry point)
- [ ] Create `gui/backend/config.py` (settings)
- [ ] Create `gui/backend/routers/` directory
- [ ] Create `gui/backend/services/` directory
- [ ] Create `gui/backend/schemas/` directory
- [ ] Create `gui/backend/utils/` directory

### 6.0.3 Frontend Scaffolding
- [ ] Initialize React+TypeScript project with Vite
- [ ] Install TailwindCSS + shadcn/ui components
- [ ] Create `gui/frontend/src/components/` directory
- [ ] Create `gui/frontend/src/pages/` directory
- [ ] Create `gui/frontend/src/hooks/` directory
- [ ] Create `gui/frontend/src/services/` directory
- [ ] Create `gui/frontend/src/types/` directory
- [ ] Create `gui/frontend/src/stores/` directory (Zustand)

### 6.0.4 Development Environment
- [ ] Create `gui/docker-compose.yml` for local dev
- [ ] Create `gui/Makefile` with common commands
- [ ] Create `gui/.env.example` template
- [ ] Create `gui/README.md` with setup instructions

---

## PHASE 6.1 — Core Infrastructure (CRITICAL)

**Duration:** 3 days
**Dependencies:** 6.0
**Deliverable:** Working backend API + frontend shell

### 6.1.1 Configuration Engine
- [ ] Create `gui/backend/services/config_engine.py`
  - [ ] `ConfigManager` class
  - [ ] `generate_config_hash()` function (SHA256)
  - [ ] `save_config()` function
  - [ ] `load_config()` function
  - [ ] `validate_config()` function
  - [ ] `get_config_history()` function
  - [ ] `diff_configs()` function
- [ ] Create `gui/backend/schemas/config.py`
  - [ ] `IntelligenceConfig` Pydantic model
  - [ ] `ExecutionRealityConfig` Pydantic model
  - [ ] `BacktestConfig` Pydantic model
  - [ ] `FullConfig` Pydantic model
- [ ] Create `gui/backend/services/config_storage.py`
  - [ ] JSON file-based storage
  - [ ] Config versioning
  - [ ] Audit trail logging
- [ ] Create `gui/backend/services/config_export.py`
  - [ ] Export to JSON
  - [ ] Export to YAML
  - [ ] Export to PDF summary

### 6.1.2 Backend API Layer - Core Endpoints
- [ ] Create `gui/backend/routers/backtest.py`
  - [ ] `POST /api/backtest/run` endpoint
  - [ ] `POST /api/backtest/replay` endpoint
  - [ ] `GET /api/backtest/status/{run_id}` endpoint
  - [ ] `GET /api/backtest/results/{run_id}` endpoint
  - [ ] `GET /api/backtest/compare` endpoint (NEW)
- [ ] Create `gui/backend/routers/config.py`
  - [ ] `GET /api/config/current` endpoint
  - [ ] `POST /api/config/update` endpoint
  - [ ] `POST /api/config/toggle` endpoint
  - [ ] `GET /api/config/history` endpoint
  - [ ] `POST /api/config/export` endpoint (NEW)
  - [ ] `GET /api/config/diff/{hash1}/{hash2}` endpoint (NEW)
- [ ] Create `gui/backend/routers/model.py`
  - [ ] `GET /api/model/list` endpoint
  - [ ] `GET /api/model/logic/{model_id}` endpoint
  - [ ] `GET /api/model/info/{model_id}` endpoint
  - [ ] `GET /api/model/diff/{model_a}/{model_b}` endpoint (NEW)
- [ ] Create `gui/backend/routers/tape.py`
  - [ ] `GET /api/tape/list` endpoint
  - [ ] `GET /api/tape/data/{tape_id}` endpoint
  - [ ] `GET /api/tape/info/{tape_id}` endpoint
  - [ ] `POST /api/tape/annotate` endpoint (NEW)
  - [ ] `GET /api/tape/annotations/{tape_id}` endpoint (NEW)
- [ ] Create `gui/backend/routers/trades.py`
  - [ ] `GET /api/trades/list/{run_id}` endpoint
  - [ ] `GET /api/trades/details/{trade_id}` endpoint
- [ ] Create `gui/backend/routers/execution.py`
  - [ ] `POST /api/execution/simulate` endpoint
  - [ ] `GET /api/execution/profiles` endpoint
- [ ] Create `gui/backend/routers/optimization.py`
  - [ ] `POST /api/optimization/run` endpoint
  - [ ] `GET /api/optimization/status/{opt_id}` endpoint
  - [ ] `GET /api/optimization/results/{opt_id}` endpoint
- [ ] Create `gui/backend/routers/readiness.py`
  - [ ] `POST /api/readiness/check` endpoint
  - [ ] `GET /api/readiness/status` endpoint
- [ ] Create `gui/backend/routers/websocket.py`
  - [ ] `/ws/backtest/{run_id}` endpoint
  - [ ] `/ws/optimization/{opt_id}` endpoint

### 6.1.3 Backend Services Layer
- [ ] Create `gui/backend/services/backtest_runner.py`
  - [ ] `BacktestRunner` class
  - [ ] Integration with `kaggle/condor_brain_backtest_v2.py`
  - [ ] Async execution support
  - [ ] Progress tracking
  - [ ] Result comparison
- [ ] Create `gui/backend/services/model_loader.py`
  - [ ] `ModelLoader` class
  - [ ] Model discovery
  - [ ] Model metadata extraction
  - [ ] Model diff computation
- [ ] Create `gui/backend/services/tape_loader.py`
  - [ ] `TapeLoader` class
  - [ ] Tape discovery
  - [ ] Tape metadata extraction
  - [ ] Data pagination
  - [ ] Annotation storage
- [ ] Create `gui/backend/services/execution_simulator.py`
  - [ ] Integration with `ExecutionRealityEngine`
  - [ ] Single-fill simulation
  - [ ] Batch simulation

### 6.1.4 Application Shell (Frontend)
- [ ] Create `gui/frontend/src/App.tsx`
  - [ ] Router setup (react-router-dom)
  - [ ] Theme provider (dark/light)
  - [ ] Global state provider (Zustand)
  - [ ] Toast notifications
- [ ] Create `gui/frontend/src/components/layout/Sidebar.tsx`
  - [ ] Navigation items
  - [ ] Active state highlighting
  - [ ] Collapse/expand
  - [ ] Icons for each section
- [ ] Create `gui/frontend/src/components/layout/TopBar.tsx`
  - [ ] App title
  - [ ] Model info display
  - [ ] Seed display
  - [ ] Config hash display (clickable)
  - [ ] Theme toggle
- [ ] Create `gui/frontend/src/components/layout/MainContent.tsx`
  - [ ] Content routing
  - [ ] Loading states
  - [ ] Error boundaries
- [ ] Create `gui/frontend/src/stores/configStore.ts`
  - [ ] Zustand store for global config
  - [ ] Config hash tracking
  - [ ] Toggle actions
  - [ ] History tracking

### 6.1.5 API Client (Frontend)
- [ ] Create `gui/frontend/src/services/api.ts`
  - [ ] Axios instance with base URL
  - [ ] Error handling interceptors
  - [ ] Request/response logging
- [ ] Create `gui/frontend/src/services/backtest.ts`
- [ ] Create `gui/frontend/src/services/config.ts`
- [ ] Create `gui/frontend/src/services/websocket.ts`
- [ ] Create `gui/frontend/src/types/api.ts`

---

## PHASE 6.2 — Dashboard (CRITICAL)

**Duration:** 2 days
**Dependencies:** 6.1
**Deliverable:** Home page with status overview and quick actions

### 6.2.1 Backend - Dashboard Endpoints
- [ ] Create `gui/backend/routers/dashboard.py`
  - [ ] `GET /api/dashboard/summary` endpoint
  - [ ] `GET /api/dashboard/recent-runs` endpoint
  - [ ] `GET /api/dashboard/determinism-status` endpoint

### 6.2.2 Frontend - Dashboard Components
- [ ] Create `gui/frontend/src/pages/Dashboard.tsx`
  - [ ] Main page layout (grid)
  - [ ] Summary cards row
  - [ ] Quick actions row
  - [ ] Recent activity row
- [ ] Create `gui/frontend/src/components/dashboard/ModelSummaryCard.tsx`
  - [ ] Model fingerprint (truncated hash)
  - [ ] Version string
  - [ ] Param count (formatted)
  - [ ] Last modified date
  - [ ] Click to view details
- [ ] Create `gui/frontend/src/components/dashboard/DatasetSummaryCard.tsx`
  - [ ] Dataset fingerprint
  - [ ] Row count (formatted)
  - [ ] Feature count
  - [ ] Date range
  - [ ] Click to view tape
- [ ] Create `gui/frontend/src/components/dashboard/DeterminismStatusCard.tsx`
  - [ ] Replay Determinism: PASS/FAIL badge
  - [ ] Batch Invariance: PASS/FAIL badge
  - [ ] Execution Physics: PASS/FAIL badge
  - [ ] Last verified timestamp
  - [ ] Click to run checks
- [ ] Create `gui/frontend/src/components/dashboard/LastBacktestCard.tsx`
  - [ ] Equity curve sparkline (mini chart)
  - [ ] Trade count
  - [ ] Sharpe ratio
  - [ ] Max DD percentage
  - [ ] NPDD score
  - [ ] View full results link
- [ ] Create `gui/frontend/src/components/dashboard/ConfigHashCard.tsx`
  - [ ] Current config hash (full)
  - [ ] Copy to clipboard button
  - [ ] View config link
  - [ ] Export button
- [ ] Create `gui/frontend/src/components/dashboard/QuickActions.tsx`
  - [ ] Run Backtest button (primary)
  - [ ] Open ICM button
  - [ ] View Last Trades button
  - [ ] Run Optimization button
- [ ] Create `gui/frontend/src/components/dashboard/RecentActivity.tsx`
  - [ ] List of recent runs
  - [ ] Status indicators
  - [ ] Quick view links

---

## PHASE 6.3 — Intelligence Control Matrix (HIGH)

**Duration:** 3 days
**Dependencies:** 6.2
**Deliverable:** Full ICM panel with all toggles and dependencies

### 6.3.1 Backend - ICM Data Models
- [ ] Create `gui/backend/schemas/intelligence.py`
  - [ ] `ModelCoreConfig` (CondorNet, Predicate, ETD, CDE, SuperSets, GroupInvariance)
  - [ ] `FuzzyComponentConfig` (11 components)
  - [ ] `DynamicIndicatorConfig` (BB, RSI, ATR, MACD, etc.)
  - [ ] `DiffusionPredictorConfig` (7 predictors + auto-select)
- [ ] Create `gui/backend/services/intelligence_manager.py`
  - [ ] `get_available_components()` function
  - [ ] `get_component_state()` function
  - [ ] `toggle_component()` function
  - [ ] `get_dependency_graph()` function
  - [ ] `validate_component_combination()` function

### 6.3.2 Backend - ICM Endpoints
- [ ] Add to `gui/backend/routers/config.py`
  - [ ] `GET /api/config/intelligence` endpoint
  - [ ] `POST /api/config/intelligence/toggle` endpoint
  - [ ] `GET /api/config/intelligence/dependencies` endpoint
  - [ ] `POST /api/config/intelligence/validate` endpoint

### 6.3.3 Frontend - ICM Components
- [ ] Create `gui/frontend/src/pages/IntelligenceControlMatrix.tsx`
  - [ ] Main page layout (sections)
  - [ ] Collapsible sections
  - [ ] Save/Reset buttons
- [ ] Create `gui/frontend/src/components/icm/ModelCoreSection.tsx`
  - [ ] CondorNet master toggle
  - [ ] Predicate Layer toggle (indented)
  - [ ] ETD toggle (indented)
  - [ ] CDE toggle (indented)
  - [ ] Super Sets toggle (indented)
  - [ ] Group Invariance toggle (indented)
  - [ ] Visual tree structure with lines
  - [ ] Dependency indicators
- [ ] Create `gui/frontend/src/components/icm/FuzzyEngineSection.tsx`
  - [ ] Master Fuzzy Engine toggle
  - [ ] 11 component toggles in grid
  - [ ] Component 1 auto-link indicator (locked when Model=ON)
  - [ ] Component descriptions on hover
  - [ ] Activation preview
- [ ] Create `gui/frontend/src/components/icm/DynamicIndicatorsSection.tsx`
  - [ ] BB toggle + params (period, std, offset)
  - [ ] RSI toggle + params (length)
  - [ ] ATR toggle + params (window)
  - [ ] MACD toggle + params (fast, slow, signal)
  - [ ] Manifold Volatility toggle
  - [ ] TDA Signature toggle
  - [ ] Realized Vol toggle
  - [ ] Skew toggle
  - [ ] Momentum primitives toggle
  - [ ] Topology primitives toggle
  - [ ] Param inputs with validation
  - [ ] Reset to defaults button per indicator
- [ ] Create `gui/frontend/src/components/icm/DiffusionPredictorsSection.tsx`
  - [ ] 7 predictor toggles
  - [ ] Auto-select best predictor toggle
  - [ ] Epsilon display for each predictor
  - [ ] Visual comparison chart
- [ ] Create `gui/frontend/src/components/ui/ToggleSwitch.tsx`
  - [ ] Reusable toggle component
  - [ ] Disabled state styling
  - [ ] Dependency lock icon
  - [ ] Loading state
- [ ] Create `gui/frontend/src/components/ui/ParamInput.tsx`
  - [ ] Number input with validation
  - [ ] Min/max bounds
  - [ ] Reset to default button
  - [ ] Tooltip with description
- [ ] Create `gui/frontend/src/hooks/useICM.ts`
  - [ ] ICM state management
  - [ ] Toggle handlers with dependency checks
  - [ ] Validation logic
  - [ ] Dirty state tracking

---

## PHASE 6.4 — Execution Reality Engine Panel (HIGH)

**Duration:** 2 days
**Dependencies:** 6.3
**Deliverable:** Full execution reality configuration panel

### 6.4.1 Backend - Execution Reality Data Models
- [ ] Create `gui/backend/schemas/execution.py`
  - [ ] `LatencyModelConfig`
  - [ ] `QueuePositionConfig`
  - [ ] `SpreadDynamicsConfig`
  - [ ] `VolatilityShockConfig`
  - [ ] `BrokenSpreadConfig`
  - [ ] `MicrostructureConfig`
  - [ ] `QuoteStalenessConfig`
  - [ ] `TimeOfDayLiquidityConfig`
  - [ ] `ExecutionRealityProfile` (combined)

### 6.4.2 Backend - Execution Reality Endpoints
- [ ] Enhance `gui/backend/routers/execution.py`
  - [ ] `GET /api/execution/config` endpoint
  - [ ] `POST /api/execution/config/update` endpoint
  - [ ] `GET /api/execution/profiles/list` endpoint
  - [ ] `POST /api/execution/profiles/save` endpoint
  - [ ] `DELETE /api/execution/profiles/{id}` endpoint
  - [ ] `POST /api/execution/simulate` endpoint

### 6.4.3 Frontend - Execution Reality Components
- [ ] Create `gui/frontend/src/pages/ExecutionRealityEngine.tsx`
  - [ ] Main page layout
  - [ ] 8 component sections (accordion)
  - [ ] Profile selector
  - [ ] Simulator panel
- [ ] Create `gui/frontend/src/components/execution/LatencyModelSection.tsx`
- [ ] Create `gui/frontend/src/components/execution/QueuePositionSection.tsx`
- [ ] Create `gui/frontend/src/components/execution/SpreadDynamicsSection.tsx`
- [ ] Create `gui/frontend/src/components/execution/VolatilityShockSection.tsx`
- [ ] Create `gui/frontend/src/components/execution/BrokenSpreadSection.tsx`
- [ ] Create `gui/frontend/src/components/execution/MicrostructureSection.tsx`
- [ ] Create `gui/frontend/src/components/execution/QuoteStalenessSection.tsx`
- [ ] Create `gui/frontend/src/components/execution/TODLiquiditySection.tsx`
- [ ] Create `gui/frontend/src/components/execution/ExecutionSimulator.tsx`
  - [ ] Quick test panel
  - [ ] Bid/Ask/Size inputs
  - [ ] Market state inputs
  - [ ] Simulate button
  - [ ] Result display with all diagnostics
- [ ] Create `gui/frontend/src/components/execution/ProfileManager.tsx`
  - [ ] Profile dropdown
  - [ ] Save as new profile
  - [ ] Delete profile
  - [ ] Set as default
- [ ] Create `gui/frontend/src/hooks/useExecutionReality.ts`

---

## PHASE 6.5 — Backtest Control Panel (CRITICAL)

**Duration:** 4 days
**Dependencies:** 6.4
**Deliverable:** Full backtest execution and results viewing

### 6.5.1 Backend - Backtest Services
- [ ] Enhance `gui/backend/services/backtest_runner.py`
  - [ ] Full integration with `condor_brain_backtest_v2.py`
  - [ ] Progress callback support
  - [ ] WebSocket progress streaming
  - [ ] Result serialization
  - [ ] Run comparison logic
- [ ] Create `gui/backend/services/run_storage.py`
  - [ ] Run result persistence (JSON)
  - [ ] Run comparison
  - [ ] Run export (CSV, JSON, Excel)
- [ ] Create `gui/backend/schemas/backtest.py`
  - [ ] All request/response models

### 6.5.2 Backend - WebSocket Support
- [ ] Enhance `gui/backend/routers/websocket.py`
  - [ ] Progress streaming
  - [ ] Real-time equity updates
  - [ ] Trade notifications

### 6.5.3 Frontend - Backtest Control Components
- [ ] Create `gui/frontend/src/pages/BacktestControlPanel.tsx`
- [ ] Create `gui/frontend/src/components/backtest/TapeSelector.tsx`
- [ ] Create `gui/frontend/src/components/backtest/ModelSelector.tsx`
- [ ] Create `gui/frontend/src/components/backtest/SeedInput.tsx`
- [ ] Create `gui/frontend/src/components/backtest/DeviceSelector.tsx`
- [ ] Create `gui/frontend/src/components/backtest/BatchSizeInput.tsx`
- [ ] Create `gui/frontend/src/components/backtest/ExecutionProfileSelector.tsx`
- [ ] Create `gui/frontend/src/components/backtest/ActionButtons.tsx`
- [ ] Create `gui/frontend/src/components/backtest/ProgressIndicator.tsx`
- [ ] Create `gui/frontend/src/components/backtest/EquityCurveChart.tsx`
  - [ ] Line chart with Recharts
  - [ ] Zoom/pan controls
  - [ ] Trade entry/exit markers
  - [ ] Drawdown shading
  - [ ] Hover tooltips
- [ ] Create `gui/frontend/src/components/backtest/MetricsSummary.tsx`
- [ ] Create `gui/frontend/src/components/backtest/TradeListTable.tsx`
- [ ] Create `gui/frontend/src/components/backtest/DeterminismBadge.tsx`
- [ ] Create `gui/frontend/src/components/backtest/BacktestComparison.tsx` (NEW)
  - [ ] Side-by-side equity curves
  - [ ] Metrics diff table
  - [ ] Trade list diff
  - [ ] Config diff viewer
- [ ] Create `gui/frontend/src/hooks/useBacktest.ts`
- [ ] Create `gui/frontend/src/hooks/useWebSocket.ts`

---

## PHASE 6.6 — Tape Viewer (MEDIUM)

**Duration:** 3 days
**Dependencies:** 6.5
**Deliverable:** Full market data explorer with overlays

### 6.6.1 Backend - Tape Data Endpoints
- [ ] Enhance `gui/backend/routers/tape.py`
  - [ ] `GET /api/tape/ohlc/{tape_id}` (paginated, binary option)
  - [ ] `GET /api/tape/quotes/{tape_id}` (bid/ask)
  - [ ] `GET /api/tape/greeks/{tape_id}`
  - [ ] `GET /api/tape/microstructure/{tape_id}`
- [ ] Create `gui/backend/services/indicator_calculator.py`
- [ ] Create `gui/backend/services/annotation_store.py` (NEW)

### 6.6.2 Frontend - Tape Viewer Components
- [ ] Create `gui/frontend/src/pages/TapeViewer.tsx`
- [ ] Create `gui/frontend/src/components/tape/CandlestickChart.tsx`
  - [ ] Use lightweight-charts library
  - [ ] OHLC candles
  - [ ] Volume bars (subplot)
  - [ ] Zoom/pan/scroll
  - [ ] Crosshair with values
- [ ] Create `gui/frontend/src/components/tape/BidAskOverlay.tsx`
- [ ] Create `gui/frontend/src/components/tape/SpreadHeatmap.tsx`
- [ ] Create `gui/frontend/src/components/tape/GreeksPanel.tsx`
- [ ] Create `gui/frontend/src/components/tape/MicrostructureEvents.tsx`
- [ ] Create `gui/frontend/src/components/tape/IndicatorOverlays.tsx`
- [ ] Create `gui/frontend/src/components/tape/DiffusionCurves.tsx`
- [ ] Create `gui/frontend/src/components/tape/IndicatorToggles.tsx`
- [ ] Create `gui/frontend/src/components/tape/AnnotationLayer.tsx` (NEW)
  - [ ] Click to add annotation
  - [ ] Annotation types (shock, regime, anomaly, etc.)
  - [ ] Annotation markers on chart
  - [ ] Annotation list sidebar
- [ ] Create `gui/frontend/src/hooks/useTapeData.ts`

---

## PHASE 6.7 — Model Introspection Panel (MEDIUM)

**Duration:** 3 days
**Dependencies:** 6.6
**Deliverable:** Deep model visualization and debugging

### 6.7.1 Backend - Model Logic Endpoints
- [ ] Create `gui/backend/services/model_introspector.py`
  - [ ] Extract predicate activations
  - [ ] Extract fuzzy memberships
  - [ ] Extract ETD/CDE states
  - [ ] Extract super set activations
  - [ ] Extract loss components
- [ ] Enhance `gui/backend/routers/model.py`
  - [ ] `GET /api/model/predicates/{run_id}`
  - [ ] `GET /api/model/fuzzy/{run_id}`
  - [ ] `GET /api/model/states/{run_id}`
  - [ ] `GET /api/model/loss/{run_id}`

### 6.7.2 Frontend - Model Introspection Components
- [ ] Create `gui/frontend/src/pages/ModelIntrospection.tsx`
- [ ] Create `gui/frontend/src/components/introspection/PredicateHeatmap.tsx`
- [ ] Create `gui/frontend/src/components/introspection/FuzzyMembershipCurves.tsx`
- [ ] Create `gui/frontend/src/components/introspection/ETDStateViewer.tsx`
- [ ] Create `gui/frontend/src/components/introspection/CDEStateViewer.tsx`
- [ ] Create `gui/frontend/src/components/introspection/SuperSetActivations.tsx`
- [ ] Create `gui/frontend/src/components/introspection/LossBreakdown.tsx`
- [ ] Create `gui/frontend/src/components/introspection/ModelVersionDiff.tsx` (NEW)
  - [ ] Select two model versions
  - [ ] Predicate diff
  - [ ] Fuzzy set diff
  - [ ] Super-set diff
  - [ ] Weight diff visualization
- [ ] Create `gui/frontend/src/hooks/useModelIntrospection.ts`

---

## PHASE 6.8 — Trade Explorer (MEDIUM)

**Duration:** 2 days
**Dependencies:** 6.7
**Deliverable:** Deep trade audit and analysis

### 6.8.1 Backend - Trade Detail Endpoints
- [ ] Enhance `gui/backend/routers/trades.py`
  - [ ] `GET /api/trades/execution/{trade_id}`
  - [ ] `GET /api/trades/decision/{trade_id}`
  - [ ] `GET /api/trades/reality/{trade_id}`
- [ ] Create `gui/backend/services/trade_analyzer.py`

### 6.8.2 Frontend - Trade Explorer Components
- [ ] Create `gui/frontend/src/pages/TradeExplorer.tsx`
- [ ] Create `gui/frontend/src/components/trades/TradeList.tsx`
- [ ] Create `gui/frontend/src/components/trades/TradeDetails.tsx`
- [ ] Create `gui/frontend/src/components/trades/EntryExitDetails.tsx`
- [ ] Create `gui/frontend/src/components/trades/BidAskFillDetails.tsx`
- [ ] Create `gui/frontend/src/components/trades/AtomicityCheck.tsx`
- [ ] Create `gui/frontend/src/components/trades/DecisionBreakdown.tsx`
- [ ] Create `gui/frontend/src/components/trades/ExecutionRealityEffects.tsx`
- [ ] Create `gui/frontend/src/hooks/useTradeExplorer.ts`

---

## PHASE 6.9 — Optimization Suite (HIGH)

**Duration:** 4 days
**Dependencies:** 6.8
**Deliverable:** Full optimization engine with multiple modes

### 6.9.1 Backend - Optimization Services
- [ ] Create `gui/backend/services/optimizer.py`
  - [ ] `AblationOptimizer` class
  - [ ] `BayesianOptimizer` class (optuna)
  - [ ] `EvolutionaryOptimizer` class
  - [ ] `CertificationRunner` class
- [ ] Create `gui/backend/schemas/optimization.py`
- [ ] Enhance `gui/backend/routers/optimization.py`

### 6.9.2 Frontend - Optimization Components
- [ ] Create `gui/frontend/src/pages/OptimizationSuite.tsx`
- [ ] Create `gui/frontend/src/components/optimization/ModeSelector.tsx`
- [ ] Create `gui/frontend/src/components/optimization/ObjectiveSelector.tsx`
- [ ] Create `gui/frontend/src/components/optimization/SearchSpaceConfig.tsx`
- [ ] Create `gui/frontend/src/components/optimization/OptimizationProgress.tsx`
- [ ] Create `gui/frontend/src/components/optimization/ResultsTable.tsx`
- [ ] Create `gui/frontend/src/components/optimization/BestConfigViewer.tsx`
- [ ] Create `gui/frontend/src/components/optimization/ComponentImportance.tsx`
- [ ] Create `gui/frontend/src/components/optimization/SynergyMap.tsx`
- [ ] Create `gui/frontend/src/hooks/useOptimization.ts`

---

## PHASE 6.10 — Polish & Testing (MEDIUM)

**Duration:** 3 days
**Dependencies:** 6.9
**Deliverable:** Production-ready UI with tests

### 6.10.1 Error Handling & Loading States
- [ ] Create `gui/frontend/src/components/ui/ErrorBoundary.tsx`
- [ ] Create `gui/frontend/src/components/ui/LoadingSpinner.tsx`
- [ ] Create `gui/frontend/src/components/ui/SkeletonLoader.tsx`
- [ ] Create `gui/frontend/src/components/ui/Toast.tsx`
- [ ] Add error boundaries to all pages
- [ ] Add loading states to all data-fetching components

### 6.10.2 Keyboard Shortcuts
- [ ] Create `gui/frontend/src/hooks/useKeyboardShortcuts.ts`
- [ ] Ctrl+R: Run backtest
- [ ] Ctrl+S: Save config
- [ ] Ctrl+E: Export results
- [ ] Escape: Close modals
- [ ] ?: Show shortcut help

### 6.10.3 Responsive Design
- [ ] Mobile-friendly sidebar (collapsible)
- [ ] Responsive charts
- [ ] Touch-friendly controls
- [ ] Breakpoint testing

### 6.10.4 Documentation
- [ ] Create `gui/README.md`
- [ ] Create `gui/backend/README.md`
- [ ] Create `gui/frontend/README.md`
- [ ] Add JSDoc/docstrings

### 6.10.5 Testing
- [ ] Backend unit tests (pytest)
- [ ] Backend integration tests
- [ ] Frontend unit tests (vitest)
- [ ] Frontend component tests
- [ ] E2E tests (Playwright)

---

## PHASE 6.11 — Deployment Readiness (HIGH)

**Duration:** 2 days
**Dependencies:** 6.10
**Deliverable:** Final deployment gate with all checks

### 6.11.1 Backend - Readiness Check Services
- [ ] Create `gui/backend/services/readiness_checker.py`
  - [ ] `check_replay_determinism()`
  - [ ] `check_batch_invariance()`
  - [ ] `check_execution_physics()`
  - [ ] `check_microstructure_realism()`
  - [ ] `check_predicate_stability()`
  - [ ] `check_fuzzy_stability()`
  - [ ] `check_numerical_stability()`
  - [ ] `check_state_leaks()`
  - [ ] `run_all_checks()`
- [ ] Enhance `gui/backend/routers/readiness.py`

### 6.11.2 Frontend - Readiness Components
- [ ] Create `gui/frontend/src/pages/DeploymentReadiness.tsx`
- [ ] Create `gui/frontend/src/components/readiness/CheckList.tsx`
- [ ] Create `gui/frontend/src/components/readiness/CheckDetail.tsx`
- [ ] Create `gui/frontend/src/components/readiness/OverallStatus.tsx`
- [ ] Create `gui/frontend/src/components/readiness/RunChecksButton.tsx`
- [ ] Create `gui/frontend/src/components/readiness/ReportExport.tsx`
- [ ] Create `gui/frontend/src/hooks/useReadiness.ts`

---

## PHASE 6.12 — Performance & GPU Optimization (HIGH) [NEW]

**Duration:** 3 days
**Dependencies:** 6.11
**Deliverable:** Optimized compute and rendering

### 6.12.1 Backend - Compute Optimization
- [ ] Implement TorchScript/TorchDynamo compilation
- [ ] Implement CUDA Graphs for repeated inference
- [ ] Implement pinned memory + async transfers
- [ ] Implement batch inference in backtester
- [ ] Implement FP16 inference mode
- [ ] Add memory profiling utilities
- [ ] Add inference timing metrics

### 6.12.2 Frontend - Graphics Optimization
- [ ] Implement WebGL charts (lightweight-charts)
- [ ] Implement OffscreenCanvas for heavy rendering
- [ ] Implement Web Workers for indicator calculation
- [ ] Implement binary data streams for large tapes
- [ ] Add virtual scrolling for large lists
- [ ] Add chart data decimation
- [ ] Add lazy loading for components

### 6.12.3 Caching Strategies
- [ ] Backend response caching (Redis optional)
- [ ] Frontend query caching (React Query)
- [ ] Indicator computation caching
- [ ] Model inference caching

---

## PHASE 6.13 — Security & Access Control (MEDIUM) [NEW]

**Duration:** 2 days
**Dependencies:** 6.12
**Deliverable:** Secure deployment-ready system

### 6.13.1 Backend Security
- [ ] Implement API key authentication
- [ ] Implement rate limiting
- [ ] Implement request validation
- [ ] Implement safe execution sandboxing
- [ ] Add audit logging for all actions
- [ ] Add input sanitization
- [ ] Add CORS configuration

### 6.13.2 Frontend Security
- [ ] Implement secure token storage
- [ ] Implement API key management UI
- [ ] Add session timeout
- [ ] Add activity logging

### 6.13.3 Build & Deployment
- [ ] Create `gui/scripts/build.sh`
- [ ] Create `gui/scripts/start.sh`
- [ ] Create `gui/Dockerfile`
- [ ] Create `gui/docker-compose.prod.yml`
- [ ] Production configuration
- [ ] Environment variable management
- [ ] Health check endpoints

---

## Summary Statistics (Revised)

| Phase | Description | Tasks | Priority | Duration |
|-------|-------------|-------|----------|----------|
| 6.0 | Scaffolding | 16 | CRITICAL | 1 day |
| 6.1 | Core Infrastructure | 48 | CRITICAL | 3 days |
| 6.2 | Dashboard | 16 | CRITICAL | 2 days |
| 6.3 | Intelligence Control Matrix | 26 | HIGH | 3 days |
| 6.4 | Execution Reality Engine | 24 | HIGH | 2 days |
| 6.5 | Backtest Control Panel | 32 | CRITICAL | 4 days |
| 6.6 | Tape Viewer | 22 | MEDIUM | 3 days |
| 6.7 | Model Introspection | 18 | MEDIUM | 3 days |
| 6.8 | Trade Explorer | 18 | MEDIUM | 2 days |
| 6.9 | Optimization Suite | 22 | HIGH | 4 days |
| 6.10 | Polish & Testing | 20 | MEDIUM | 3 days |
| 6.11 | Deployment Readiness | 16 | HIGH | 2 days |
| 6.12 | GPU Optimization | 18 | HIGH | 3 days |
| 6.13 | Security | 16 | MEDIUM | 2 days |
| **TOTAL** | | **312** | | **37 days** |

---

## Technology Stack (Final)

### Backend
- **Framework:** FastAPI
- **Async:** asyncio + uvicorn
- **Validation:** Pydantic v2
- **WebSocket:** fastapi-websockets
- **Testing:** pytest + pytest-asyncio

### Frontend
- **Framework:** React 18 + TypeScript
- **Build:** Vite
- **Styling:** TailwindCSS + shadcn/ui
- **State:** Zustand
- **Charts:** Recharts + lightweight-charts (WebGL)
- **API:** Axios + React Query
- **Testing:** Vitest + Playwright

### Infrastructure
- **Container:** Docker
- **Orchestration:** docker-compose
- **Caching:** Redis (optional)
