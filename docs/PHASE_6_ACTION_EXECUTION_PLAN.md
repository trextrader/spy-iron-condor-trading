# Phase 6 GUI - Timeline Action Execution Plan

## Overview

This document provides the **day-by-day execution plan** for implementing the CondorBrain GUI. Each action item includes specific deliverables, verification criteria, and rollback procedures.

---

## Week 1: Foundation (Days 1-7)

### Day 1: Project Scaffolding Start
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 1 ACTIONS                                                          │
├────────────────────────────────────────────────────────────────────────┤
│ Morning (4h)                                                           │
│ ☐ Create monorepo structure                                            │
│   └── mkdir -p gui/{backend,frontend,shared}                           │
│ ☐ Initialize Python backend project                                    │
│   └── Create pyproject.toml with FastAPI, Pydantic v2                 │
│ ☐ Initialize React frontend project                                    │
│   └── npx create-vite gui-frontend --template react-ts                │
│                                                                        │
│ Afternoon (4h)                                                         │
│ ☐ Configure TypeScript strict mode                                     │
│ ☐ Set up TailwindCSS                                                   │
│ ☐ Install core dependencies:                                           │
│   └── zustand, @tanstack/react-query, recharts, lightweight-charts    │
│ ☐ Create shared types directory                                        │
│                                                                        │
│ VERIFICATION:                                                          │
│ ✓ `pip install -e gui/backend` succeeds                               │
│ ✓ `npm run dev` starts frontend on localhost:5173                     │
│ ✓ TypeScript compiles with zero errors                                │
└────────────────────────────────────────────────────────────────────────┘
```

### Day 2: Tooling & CI/CD
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 2 ACTIONS                                                          │
├────────────────────────────────────────────────────────────────────────┤
│ Morning (4h)                                                           │
│ ☐ Configure ESLint + Prettier                                          │
│   └── .eslintrc.cjs with React hooks rules                            │
│ ☐ Set up Black + Ruff for Python                                      │
│   └── pyproject.toml [tool.black], [tool.ruff]                        │
│ ☐ Create pre-commit hooks                                              │
│   └── .pre-commit-config.yaml                                          │
│                                                                        │
│ Afternoon (4h)                                                         │
│ ☐ Create GitHub Actions workflow                                       │
│   └── .github/workflows/gui-ci.yml                                    │
│   └── Jobs: lint, typecheck, test, build                              │
│ ☐ Create Docker development configs                                    │
│   └── docker-compose.dev.yml                                          │
│   └── Dockerfile.backend, Dockerfile.frontend                         │
│                                                                        │
│ VERIFICATION:                                                          │
│ ✓ `pre-commit run --all-files` passes                                 │
│ ✓ `docker-compose -f docker-compose.dev.yml up` starts both services │
│ ✓ CI workflow runs on push (dry run with act)                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Day 3: Core Infrastructure Begin
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 3 ACTIONS                                                          │
├────────────────────────────────────────────────────────────────────────┤
│ Morning (4h)                                                           │
│ ☐ Complete scaffolding documentation                                   │
│   └── docs/GUI_DEVELOPMENT.md                                         │
│   └── docs/API_REFERENCE.md (skeleton)                                │
│ ☐ Create FastAPI application skeleton                                  │
│   └── gui/backend/main.py with app factory                            │
│   └── gui/backend/config.py with Settings class                       │
│                                                                        │
│ Afternoon (4h)                                                         │
│ ☐ Define all Pydantic schemas (Phase 6.1)                             │
│   └── gui/backend/schemas/config.py                                   │
│   └── gui/backend/schemas/backtest.py                                 │
│   └── gui/backend/schemas/intelligence.py                             │
│   └── gui/backend/schemas/execution.py                                │
│   └── gui/backend/schemas/optimization.py                             │
│                                                                        │
│ VERIFICATION:                                                          │
│ ✓ `uvicorn gui.backend.main:app` starts on port 8000                  │
│ ✓ OpenAPI docs available at /docs                                     │
│ ✓ All schema imports succeed without circular dependencies            │
└────────────────────────────────────────────────────────────────────────┘
```

### Day 4: Backend Services
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 4 ACTIONS                                                          │
├────────────────────────────────────────────────────────────────────────┤
│ Morning (4h)                                                           │
│ ☐ Implement ConfigManager service                                      │
│   └── gui/backend/services/config_engine.py                           │
│   └── Methods: get_current_config, set_config, toggle_component       │
│   └── Methods: get_config_hash, get_config_by_hash                    │
│ ☐ Implement BacktestRunner service                                     │
│   └── gui/backend/services/backtest_runner.py                         │
│   └── Integration with core/backtest_engine.py                        │
│                                                                        │
│ Afternoon (4h)                                                         │
│ ☐ Create config router                                                 │
│   └── gui/backend/routers/config.py                                   │
│   └── Endpoints: GET/POST /config/current, /config/toggle             │
│ ☐ Create backtest router                                               │
│   └── gui/backend/routers/backtest.py                                 │
│   └── Endpoints: POST /backtest/run, GET /backtest/status/{id}        │
│                                                                        │
│ VERIFICATION:                                                          │
│ ✓ curl localhost:8000/api/config/current returns valid JSON           │
│ ✓ Config hash is deterministic (same config = same hash)             │
│ ✓ POST /api/backtest/run returns run_id                               │
└────────────────────────────────────────────────────────────────────────┘
```

### Day 5: WebSocket & More Routers
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 5 ACTIONS                                                          │
├────────────────────────────────────────────────────────────────────────┤
│ Morning (4h)                                                           │
│ ☐ Implement WebSocketManager                                           │
│   └── gui/backend/services/websocket_manager.py                       │
│   └── Channels: backtest.progress, optimization.progress, logs        │
│   └── Connection management with heartbeat                            │
│ ☐ Create websocket router                                              │
│   └── gui/backend/routers/websocket.py                                │
│                                                                        │
│ Afternoon (4h)                                                         │
│ ☐ Create remaining routers:                                            │
│   └── gui/backend/routers/model.py (model info, weights)              │
│   └── gui/backend/routers/tape.py (tape listing, data access)         │
│   └── gui/backend/routers/trades.py (trade history, analysis)         │
│   └── gui/backend/routers/execution.py (exec reality controls)        │
│                                                                        │
│ VERIFICATION:                                                          │
│ ✓ WebSocket connects at ws://localhost:8000/ws                        │
│ ✓ Can subscribe to backtest.progress channel                          │
│ ✓ All router modules import without error                             │
└────────────────────────────────────────────────────────────────────────┘
```

### Day 6: React Foundation
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 6 ACTIONS                                                          │
├────────────────────────────────────────────────────────────────────────┤
│ Morning (4h)                                                           │
│ ☐ Create React app shell                                               │
│   └── App.tsx with React Router                                       │
│   └── Layout components: Sidebar, Header, MainContent                 │
│ ☐ Set up Zustand stores                                                │
│   └── stores/configStore.ts                                           │
│   └── stores/backtestStore.ts                                         │
│   └── stores/uiStore.ts                                               │
│                                                                        │
│ Afternoon (4h)                                                         │
│ ☐ Create API client layer                                              │
│   └── api/client.ts (axios instance with interceptors)                │
│   └── api/config.ts (config endpoints)                                │
│   └── api/backtest.ts (backtest endpoints)                            │
│ ☐ Set up React Query hooks                                             │
│   └── hooks/useConfig.ts                                              │
│   └── hooks/useBacktest.ts                                            │
│                                                                        │
│ VERIFICATION:                                                          │
│ ✓ App renders with sidebar navigation                                 │
│ ✓ Zustand devtools shows state updates                                │
│ ✓ API calls reach backend (check network tab)                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Day 7: Integration & Dashboard Start
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 7 ACTIONS                                                          │
├────────────────────────────────────────────────────────────────────────┤
│ Morning (4h)                                                           │
│ ☐ Implement WebSocket hook                                             │
│   └── hooks/useWebSocket.ts                                           │
│   └── Auto-reconnect, channel subscription                            │
│ ☐ Create error boundary components                                     │
│   └── components/ErrorBoundary.tsx                                    │
│   └── components/LoadingState.tsx                                     │
│                                                                        │
│ Afternoon (4h)                                                         │
│ ☐ Begin Dashboard panel (6.2)                                          │
│   └── pages/Dashboard.tsx                                             │
│   └── components/dashboard/MetricCard.tsx                             │
│   └── components/dashboard/StatusIndicator.tsx                        │
│                                                                        │
│ MILESTONE CHECK:                                                       │
│ ✓ M1: Infrastructure Ready                                            │
│ ✓ Backend: All routers responding                                     │
│ ✓ Frontend: App shell + API integration working                       │
│ ✓ WebSocket: Connection established                                   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Week 2: Control Panels (Days 8-14)

### Day 8: Dashboard Core
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 8 ACTIONS                                                          │
├────────────────────────────────────────────────────────────────────────┤
│ Morning (4h)                                                           │
│ ☐ Complete MetricCard component                                        │
│   └── Variants: standard, highlight, warning                          │
│   └── Props: label, value, change, icon                               │
│ ☐ Create StatusGrid component                                          │
│   └── 4-column grid layout                                            │
│   └── Metrics: Total Trades, Win Rate, Net P&L, Sharpe                │
│                                                                        │
│ Afternoon (4h)                                                         │
│ ☐ Implement EquityCurve mini-chart                                     │
│   └── components/dashboard/EquityCurveMini.tsx                        │
│   └── Using Recharts AreaChart                                        │
│   └── 7-day rolling view                                              │
│ ☐ Add responsive breakpoints                                           │
│   └── Mobile: 1 column, Tablet: 2 columns, Desktop: 4 columns        │
│                                                                        │
│ VERIFICATION:                                                          │
│ ✓ Dashboard shows live metric data                                    │
│ ✓ Charts render without console errors                                │
│ ✓ Responsive layout works at all breakpoints                          │
└────────────────────────────────────────────────────────────────────────┘
```

### Day 9: Dashboard Advanced
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 9 ACTIONS                                                          │
├────────────────────────────────────────────────────────────────────────┤
│ Morning (4h)                                                           │
│ ☐ Create ActivityFeed component                                        │
│   └── Real-time log entries via WebSocket                             │
│   └── Virtualized list for performance                                │
│ ☐ Create RecentRunsList component                                      │
│   └── Last 10 backtest runs                                           │
│   └── Quick status indicators                                         │
│                                                                        │
│ Afternoon (4h)                                                         │
│ ☐ Implement QuickActions panel                                         │
│   └── "New Backtest" button                                           │
│   └── "View Results" shortcut                                         │
│   └── "Export Config" button                                          │
│ ☐ Add loading skeletons for all components                            │
│                                                                        │
│ VERIFICATION:                                                          │
│ ✓ Activity feed updates in real-time                                  │
│ ✓ Recent runs list populated from API                                 │
│ ✓ Skeleton loaders appear during data fetch                           │
└────────────────────────────────────────────────────────────────────────┘
```

### Day 10: Dashboard Complete + ICM Start
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 10 ACTIONS                                                         │
├────────────────────────────────────────────────────────────────────────┤
│ Morning (4h)                                                           │
│ ☐ Dashboard polish                                                     │
│   └── Animation transitions                                           │
│   └── Tooltip enhancements                                            │
│   └── Color coding for positive/negative values                       │
│ ☐ Dashboard E2E tests                                                  │
│   └── tests/e2e/dashboard.spec.ts                                     │
│                                                                        │
│ Afternoon (4h)                                                         │
│ ☐ Begin ICM Panel (6.3)                                                │
│   └── pages/IntelligencePanel.tsx                                     │
│   └── components/icm/ComponentTree.tsx                                │
│ ☐ Create tree data structure                                           │
│   └── CondorNet → [ETD-1, TFT, CDE, MoE]                              │
│   └── Fuzzy → [10 factors]                                            │
│                                                                        │
│ VERIFICATION:                                                          │
│ ✓ Dashboard fully functional                                          │
│ ✓ ICM tree renders with all components                                │
│ ✓ E2E tests pass                                                       │
└────────────────────────────────────────────────────────────────────────┘
```

### Day 11: ICM Panel Core
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 11 ACTIONS                                                         │
├────────────────────────────────────────────────────────────────────────┤
│ Morning (4h)                                                           │
│ ☐ Implement ToggleSwitch component                                     │
│   └── Animated on/off state                                           │
│   └── Disabled state for dependencies                                 │
│ ☐ Implement ComponentNode component                                    │
│   └── Expandable/collapsible                                          │
│   └── Status indicator (active, inactive, error)                      │
│                                                                        │
│ Afternoon (4h)                                                         │
│ ☐ Wire toggle to ConfigManager backend                                 │
│   └── POST /api/config/toggle                                         │
│   └── Optimistic UI update + rollback on error                        │
│ ☐ Implement dependency resolution                                      │
│   └── Disable child toggles when parent disabled                      │
│   └── Visual indication of dependency chain                           │
│                                                                        │
│ VERIFICATION:                                                          │
│ ✓ Toggle updates persist to backend                                   │
│ ✓ Config hash changes after toggle                                    │
│ ✓ Dependencies correctly enforced                                     │
└────────────────────────────────────────────────────────────────────────┘
```

### Day 12: ICM Advanced Features
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 12 ACTIONS                                                         │
├────────────────────────────────────────────────────────────────────────┤
│ Morning (4h)                                                           │
│ ☐ Implement BatchToggle component                                      │
│   └── "Enable All", "Disable All" buttons                             │
│   └── Category-level batch operations                                 │
│ ☐ Create ConfigDiff component                                          │
│   └── Show changes since last save                                    │
│   └── Highlight modified components                                   │
│                                                                        │
│ Afternoon (4h)                                                         │
│ ☐ Implement save/restore presets                                       │
│   └── Local storage persistence                                       │
│   └── Named presets (e.g., "Full CondorNet", "Fuzzy Only")           │
│ ☐ Add keyboard navigation                                              │
│   └── Arrow keys for tree navigation                                  │
│   └── Space to toggle                                                 │
│                                                                        │
│ VERIFICATION:                                                          │
│ ✓ Batch operations work correctly                                     │
│ ✓ Presets persist across sessions                                     │
│ ✓ Full keyboard accessibility                                         │
│                                                                        │
│ MILESTONE CHECK:                                                       │
│ ✓ M2: Dashboard Live                                                  │
└────────────────────────────────────────────────────────────────────────┘
```

### Day 13: Execution Reality Panel
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 13 ACTIONS                                                         │
├────────────────────────────────────────────────────────────────────────┤
│ Morning (4h)                                                           │
│ ☐ Create ExecutionPanel page                                           │
│   └── pages/ExecutionPanel.tsx                                        │
│ ☐ Implement 8 ComponentCard components:                                │
│   └── SlippageCard, SpreadDynamicsCard                                │
│   └── FillProbabilityCard, PartialFillCard                           │
│   └── LatencyCard, QueuePositionCard                                 │
│   └── RejectionsCard, MarketImpactCard                               │
│                                                                        │
│ Afternoon (4h)                                                         │
│ ☐ Create SliderControl with numeric input                              │
│   └── Range validation                                                │
│   └── Step increments                                                 │
│   └── Real-time preview                                               │
│ ☐ Wire controls to ExecutionRealityConfig                             │
│                                                                        │
│ VERIFICATION:                                                          │
│ ✓ All 8 components render with controls                               │
│ ✓ Slider values persist to config                                     │
│ ✓ Validation prevents invalid inputs                                  │
└────────────────────────────────────────────────────────────────────────┘
```

### Day 14: Execution Reality Complete
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 14 ACTIONS                                                         │
├────────────────────────────────────────────────────────────────────────┤
│ Morning (4h)                                                           │
│ ☐ Implement Monte Carlo visualization                                  │
│   └── components/execution/MCDistribution.tsx                         │
│   └── Histogram of expected slippage                                  │
│   └── Confidence interval bands                                       │
│ ☐ Create SimulateButton with preview                                   │
│   └── POST /api/execution/simulate                                    │
│   └── Display expected impact summary                                 │
│                                                                        │
│ Afternoon (4h)                                                         │
│ ☐ Implement profile management                                         │
│   └── ProfileSelector dropdown                                        │
│   └── Save/Load named profiles                                        │
│   └── Presets: "Ideal", "Conservative", "Realistic"                  │
│ ☐ ICM + Exec Reality integration tests                                 │
│                                                                        │
│ VERIFICATION:                                                          │
│ ✓ MC visualization updates on parameter change                        │
│ ✓ Profile save/load works correctly                                   │
│ ✓ Integration with ICM panel (exec reality toggle)                   │
│                                                                        │
│ MILESTONE CHECK:                                                       │
│ ✓ M3: Control Panels Functional                                       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Week 3: Backtest Ecosystem (Days 15-21)

### Day 15-16: Backtest Form
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 15-16 ACTIONS                                                      │
├────────────────────────────────────────────────────────────────────────┤
│ Day 15 Morning (4h)                                                    │
│ ☐ Create BacktestPanel page                                            │
│   └── pages/BacktestPanel.tsx                                         │
│ ☐ Implement TapeSelector component                                     │
│   └── Dropdown with available tapes                                   │
│   └── Date range display                                              │
│   └── Record count                                                    │
│                                                                        │
│ Day 15 Afternoon (4h)                                                  │
│ ☐ Implement ModelSelector component                                    │
│   └── Available model versions                                        │
│   └── Checkpoint info display                                         │
│ ☐ Create SeedInput with randomize button                              │
│                                                                        │
│ Day 16 Morning (4h)                                                    │
│ ☐ Implement DeviceSelector (CPU/CUDA/MPS)                             │
│ ☐ Create BatchSizeInput                                                │
│ ☐ Add LimitInput (number of bars)                                     │
│                                                                        │
│ Day 16 Afternoon (4h)                                                  │
│ ☐ Implement form validation                                            │
│   └── Required field indicators                                       │
│   └── Cross-field validation                                          │
│ ☐ Create RunButton with loading state                                  │
│                                                                        │
│ VERIFICATION:                                                          │
│ ✓ Form validates all required fields                                  │
│ ✓ POST /api/backtest/run succeeds                                     │
│ ✓ Run ID returned and stored                                          │
└────────────────────────────────────────────────────────────────────────┘
```

### Day 17-18: Progress & WebSocket
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 17-18 ACTIONS                                                      │
├────────────────────────────────────────────────────────────────────────┤
│ Day 17 Morning (4h)                                                    │
│ ☐ Create ProgressTracker component                                     │
│   └── components/backtest/ProgressTracker.tsx                         │
│ ☐ Implement ProgressBar with ETA                                       │
│   └── Percentage complete                                             │
│   └── Bars processed / Total bars                                     │
│   └── Estimated time remaining                                        │
│                                                                        │
│ Day 17 Afternoon (4h)                                                  │
│ ☐ Implement LogStream component                                        │
│   └── Real-time log output via WebSocket                              │
│   └── Auto-scroll with pause button                                   │
│   └── Log level filtering (INFO, WARN, ERROR)                         │
│ ☐ Create CancelButton with confirmation                                │
│                                                                        │
│ Day 18 Morning (4h)                                                    │
│ ☐ Backend: Emit progress events from BacktestRunner                   │
│   └── Connect to WebSocketManager                                     │
│   └── Emit every N bars (configurable)                                │
│ ☐ Frontend: Subscribe to backtest.progress channel                    │
│                                                                        │
│ Day 18 Afternoon (4h)                                                  │
│ ☐ Implement StatusBadge component                                      │
│   └── States: pending, running, completed, failed, cancelled          │
│ ☐ Add notification on completion                                       │
│   └── Toast notification                                              │
│   └── Sound option (configurable)                                     │
│                                                                        │
│ VERIFICATION:                                                          │
│ ✓ Progress updates arrive via WebSocket                               │
│ ✓ Progress bar reflects actual completion                             │
│ ✓ Cancel stops the backtest immediately                               │
└────────────────────────────────────────────────────────────────────────┘
```

### Day 19-20: Results Display
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 19-20 ACTIONS                                                      │
├────────────────────────────────────────────────────────────────────────┤
│ Day 19 Morning (4h)                                                    │
│ ☐ Create ResultsPanel component                                        │
│   └── components/backtest/ResultsPanel.tsx                            │
│ ☐ Implement MetricsGrid                                                │
│   └── Net P&L, Sharpe, Max DD, Win Rate                              │
│   └── NPDD ratio (key metric)                                        │
│   └── Trade count, Avg trade duration                                │
│                                                                        │
│ Day 19 Afternoon (4h)                                                  │
│ ☐ Implement full EquityCurve chart                                     │
│   └── Using lightweight-charts (WebGL)                                │
│   └── Zoom, pan, crosshair                                            │
│   └── Drawdown overlay option                                         │
│ ☐ Add benchmark comparison line                                        │
│                                                                        │
│ Day 20 Morning (4h)                                                    │
│ ☐ Create DrawdownChart component                                       │
│   └── Underwater equity chart                                         │
│   └── Max DD markers                                                  │
│ ☐ Implement MonthlyReturnsHeatmap                                      │
│   └── Color-coded monthly P&L                                         │
│                                                                        │
│ Day 20 Afternoon (4h)                                                  │
│ ☐ Create TradeList summary                                             │
│   └── Top 10 winners/losers                                           │
│   └── Click to expand details                                         │
│ ☐ Add export buttons                                                   │
│   └── Export as PNG, CSV, JSON                                        │
│                                                                        │
│ VERIFICATION:                                                          │
│ ✓ Results display immediately after completion                        │
│ ✓ Charts render correctly with real data                              │
│ ✓ Export produces valid files                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Day 21: Comparison & Replay
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 21 ACTIONS                                                         │
├────────────────────────────────────────────────────────────────────────┤
│ Morning (4h)                                                           │
│ ☐ Implement ComparisonPanel (NEW)                                      │
│   └── components/backtest/ComparisonPanel.tsx                         │
│   └── Side-by-side metrics comparison                                 │
│   └── Overlay equity curves                                           │
│   └── Diff highlighting for config changes                            │
│ ☐ Backend: Add /api/backtest/compare endpoint                          │
│                                                                        │
│ Afternoon (4h)                                                         │
│ ☐ Implement ReplayControls                                             │
│   └── POST /api/backtest/replay                                       │
│   └── Determinism verification badge                                  │
│   └── Diff fingerprint display                                        │
│ ☐ Add ConfigSnapshotExport (NEW)                                       │
│   └── Export full config as JSON                                      │
│   └── Include model checkpoint hash                                   │
│                                                                        │
│ VERIFICATION:                                                          │
│ ✓ Can compare two backtest runs                                       │
│ ✓ Replay produces identical results                                   │
│ ✓ Config export/import works correctly                                │
│                                                                        │
│ MILESTONE CHECK:                                                       │
│ ✓ M4: Backtest Workflow Complete                                      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Week 4: Data Views (Days 22-28)

### Day 22-23: Tape Viewer
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 22-23 ACTIONS                                                      │
├────────────────────────────────────────────────────────────────────────┤
│ Day 22 Morning (4h)                                                    │
│ ☐ Create TapeViewer page                                               │
│   └── pages/TapeViewer.tsx                                            │
│ ☐ Implement VirtualizedDataGrid                                        │
│   └── Using @tanstack/react-virtual                                   │
│   └── 100k+ row support                                               │
│   └── Column resizing                                                 │
│                                                                        │
│ Day 22 Afternoon (4h)                                                  │
│ ☐ Backend: Implement paginated tape API                                │
│   └── GET /api/tape/{tape_id}/data?offset=&limit=                    │
│ ☐ Frontend: Implement infinite scroll                                  │
│                                                                        │
│ Day 23 Morning (4h)                                                    │
│ ☐ Implement FilterPanel                                                │
│   └── Date range filter                                               │
│   └── Column value filters                                            │
│   └── Search box                                                      │
│                                                                        │
│ Day 23 Afternoon (4h)                                                  │
│ ☐ Implement AnnotationLayer (NEW)                                      │
│   └── Right-click context menu                                        │
│   └── Add notes to specific rows                                      │
│   └── Highlight important bars                                        │
│   └── Persist annotations to local storage                            │
│                                                                        │
│ VERIFICATION:                                                          │
│ ✓ Grid handles 100k rows smoothly                                     │
│ ✓ Filtering updates in <100ms                                         │
│ ✓ Annotations persist across page refresh                             │
└────────────────────────────────────────────────────────────────────────┘
```

### Day 24-25: Model Introspection
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 24-25 ACTIONS                                                      │
├────────────────────────────────────────────────────────────────────────┤
│ Day 24 Morning (4h)                                                    │
│ ☐ Create ModelIntrospection page                                       │
│   └── pages/ModelIntrospection.tsx                                    │
│ ☐ Implement ArchitectureViewer                                         │
│   └── Network diagram visualization                                   │
│   └── Layer information on hover                                      │
│   └── Zoom/pan controls                                               │
│                                                                        │
│ Day 24 Afternoon (4h)                                                  │
│ ☐ Backend: Implement model introspection API                           │
│   └── GET /api/model/{model_id}/architecture                         │
│   └── GET /api/model/{model_id}/weights-summary                      │
│ ☐ Implement WeightHeatmap component                                    │
│   └── Layer selection dropdown                                        │
│   └── Heatmap visualization                                           │
│                                                                        │
│ Day 25 Morning (4h)                                                    │
│ ☐ Implement VersionDiffViewer (NEW)                                    │
│   └── Compare two model versions                                      │
│   └── Weight difference visualization                                 │
│   └── Parameter change summary                                        │
│ ☐ Backend: Add /api/model/compare endpoint                             │
│                                                                        │
│ Day 25 Afternoon (4h)                                                  │
│ ☐ Implement AttentionVisualization                                     │
│   └── TFT attention heads                                             │
│   └── Feature importance display                                      │
│ ☐ Add model info panel                                                 │
│   └── Training date, checkpoint hash                                  │
│   └── Validation metrics                                              │
│                                                                        │
│ VERIFICATION:                                                          │
│ ✓ Architecture diagram matches actual model                           │
│ ✓ Weight heatmaps load for all layers                                 │
│ ✓ Version diff shows meaningful changes                               │
│                                                                        │
│ MILESTONE CHECK:                                                       │
│ ✓ M5: Data Views Ready                                                │
└────────────────────────────────────────────────────────────────────────┘
```

### Day 26-28: Trade Explorer
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 26-28 ACTIONS                                                      │
├────────────────────────────────────────────────────────────────────────┤
│ Day 26 Morning (4h)                                                    │
│ ☐ Create TradeExplorer page                                            │
│   └── pages/TradeExplorer.tsx                                         │
│ ☐ Implement TradeTable component                                       │
│   └── Sortable columns                                                │
│   └── Multi-column filtering                                          │
│   └── Row grouping by date/outcome                                    │
│                                                                        │
│ Day 26 Afternoon (4h)                                                  │
│ ☐ Backend: Implement trades API                                        │
│   └── GET /api/trades/{run_id}?sort=&filter=                         │
│   └── GET /api/trades/{run_id}/{trade_id}                            │
│                                                                        │
│ Day 27 Morning (4h)                                                    │
│ ☐ Implement TradeDetailModal                                           │
│   └── 4-leg Iron Condor display                                       │
│   └── Greeks at entry/exit                                            │
│   └── P&L breakdown by leg                                            │
│                                                                        │
│ Day 27 Afternoon (4h)                                                  │
│ ☐ Create P&L visualization components                                  │
│   └── TradeP&LChart (entry to exit)                                   │
│   └── LegBreakdown component                                          │
│                                                                        │
│ Day 28 Morning (4h)                                                    │
│ ☐ Implement TradeAnalytics                                             │
│   └── Win/loss distribution chart                                     │
│   └── Duration histogram                                              │
│   └── Time-of-day analysis                                            │
│                                                                        │
│ Day 28 Afternoon (4h)                                                  │
│ ☐ Add export capabilities                                              │
│   └── Export selected trades                                          │
│   └── CSV/JSON formats                                                │
│ ☐ Trade Explorer E2E tests                                             │
│                                                                        │
│ VERIFICATION:                                                          │
│ ✓ All trades from backtest visible                                    │
│ ✓ Detail modal shows correct 4-leg structure                          │
│ ✓ Analytics charts render correctly                                   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Week 5: Optimization & Polish (Days 29-35)

### Day 29-30: Optimization Suite Start
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 29-30 ACTIONS                                                      │
├────────────────────────────────────────────────────────────────────────┤
│ Day 29 Morning (4h)                                                    │
│ ☐ Create OptimizationPanel page                                        │
│   └── pages/OptimizationPanel.tsx                                     │
│ ☐ Implement ModeSelector tabs                                          │
│   └── Ablation, Bayesian, Evolutionary, Certification                │
│                                                                        │
│ Day 29 Afternoon (4h)                                                  │
│ ☐ Implement AblationStudyPanel                                         │
│   └── Component selection checklist                                   │
│   └── Metrics to evaluate                                             │
│   └── Run button                                                      │
│                                                                        │
│ Day 30 Morning (4h)                                                    │
│ ☐ Backend: Implement optimization runner                               │
│   └── gui/backend/services/optimization_runner.py                     │
│   └── Integrate with core/optimizer.py                                │
│                                                                        │
│ Day 30 Afternoon (4h)                                                  │
│ ☐ Implement ComponentImportanceChart                                   │
│   └── Horizontal bar chart                                            │
│   └── Color coding by importance                                      │
│ ☐ WebSocket integration for optimization progress                      │
│                                                                        │
│ VERIFICATION:                                                          │
│ ✓ Ablation study runs and produces results                            │
│ ✓ Progress updates via WebSocket                                      │
│ ✓ Importance chart displays correctly                                 │
└────────────────────────────────────────────────────────────────────────┘
```

### Day 31-32: Optimization Advanced
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 31-32 ACTIONS                                                      │
├────────────────────────────────────────────────────────────────────────┤
│ Day 31 Morning (4h)                                                    │
│ ☐ Implement BayesianPanel                                              │
│   └── Search space definition                                         │
│   └── Acquisition function selector                                   │
│   └── Convergence visualization                                       │
│                                                                        │
│ Day 31 Afternoon (4h)                                                  │
│ ☐ Implement ParetoChart                                                │
│   └── Multi-objective visualization                                   │
│   └── Interactive point selection                                     │
│   └── Pareto frontier highlight                                       │
│                                                                        │
│ Day 32 Morning (4h)                                                    │
│ ☐ Implement SynergyMatrix                                              │
│   └── Component interaction heatmap                                   │
│   └── Positive/negative synergy colors                                │
│   └── Tooltip with details                                            │
│                                                                        │
│ Day 32 Afternoon (4h)                                                  │
│ ☐ Implement CertificationWorkflow                                      │
│   └── Step-by-step process                                            │
│   └── Validation checks                                               │
│   └── Report generation                                               │
│                                                                        │
│ VERIFICATION:                                                          │
│ ✓ Bayesian optimization runs end-to-end                               │
│ ✓ Pareto chart interactive and accurate                               │
│ ✓ Certification workflow produces report                              │
│                                                                        │
│ MILESTONE CHECK:                                                       │
│ ✓ M6: Full Feature Parity                                             │
└────────────────────────────────────────────────────────────────────────┘
```

### Day 33-34: Polish & Testing
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 33-34 ACTIONS                                                      │
├────────────────────────────────────────────────────────────────────────┤
│ Day 33 Morning (4h)                                                    │
│ ☐ Accessibility audit                                                  │
│   └── WCAG 2.1 AA compliance check                                   │
│   └── Screen reader testing                                           │
│   └── Color contrast verification                                     │
│                                                                        │
│ Day 33 Afternoon (4h)                                                  │
│ ☐ Keyboard navigation polish                                           │
│   └── Tab order verification                                          │
│   └── Focus indicators                                                │
│   └── Keyboard shortcuts documentation                                │
│                                                                        │
│ Day 34 Morning (4h)                                                    │
│ ☐ Error handling comprehensive test                                    │
│   └── Network failure scenarios                                       │
│   └── Invalid data responses                                          │
│   └── WebSocket disconnection                                         │
│                                                                        │
│ Day 34 Afternoon (4h)                                                  │
│ ☐ E2E test suite                                                       │
│   └── Full workflow tests                                             │
│   └── Cross-browser testing                                           │
│   └── Mobile viewport testing                                         │
│                                                                        │
│ VERIFICATION:                                                          │
│ ✓ All a11y issues resolved                                            │
│ ✓ E2E tests pass on Chrome, Firefox, Safari                          │
│ ✓ Mobile layout functional                                            │
└────────────────────────────────────────────────────────────────────────┘
```

### Day 35: Deployment Prep
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 35 ACTIONS                                                         │
├────────────────────────────────────────────────────────────────────────┤
│ Morning (4h)                                                           │
│ ☐ Create production Docker builds                                      │
│   └── Multi-stage Dockerfile.backend.prod                             │
│   └── Multi-stage Dockerfile.frontend.prod                            │
│   └── docker-compose.prod.yml                                         │
│ ☐ Optimize bundle sizes                                                │
│   └── Code splitting                                                  │
│   └── Tree shaking verification                                       │
│                                                                        │
│ Afternoon (4h)                                                         │
│ ☐ Environment configuration                                            │
│   └── .env.example with all variables                                 │
│   └── Environment validation on startup                               │
│ ☐ Begin TorchScript compilation (6.12)                                 │
│   └── scripts/compile_torchscript.py                                  │
│   └── Model optimization for inference                                │
│                                                                        │
│ VERIFICATION:                                                          │
│ ✓ Production build succeeds                                           │
│ ✓ Bundle size < 500KB (gzipped)                                       │
│ ✓ TorchScript model loads correctly                                   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Week 6: Finalization (Days 36-37)

### Day 36: Performance & Security
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 36 ACTIONS                                                         │
├────────────────────────────────────────────────────────────────────────┤
│ Morning (4h)                                                           │
│ ☐ CUDA Graphs implementation                                           │
│   └── Inference path optimization                                     │
│   └── Memory pre-allocation                                           │
│ ☐ FP16 inference mode                                                  │
│   └── torch.cuda.amp integration                                      │
│   └── Accuracy validation                                             │
│                                                                        │
│ Afternoon (4h)                                                         │
│ ☐ API key management (6.13)                                            │
│   └── Secure storage                                                  │
│   └── Key rotation support                                            │
│ ☐ Input sanitization                                                   │
│   └── All user inputs validated                                       │
│   └── Path traversal prevention                                       │
│ ☐ Rate limiting                                                        │
│   └── Per-endpoint limits                                             │
│   └── Configurable thresholds                                         │
│                                                                        │
│ VERIFICATION:                                                          │
│ ✓ GPU inference 2x faster with CUDA Graphs                           │
│ ✓ FP16 results match FP32 within tolerance                           │
│ ✓ Rate limiting blocks excessive requests                             │
└────────────────────────────────────────────────────────────────────────┘
```

### Day 37: Final Integration
```
┌────────────────────────────────────────────────────────────────────────┐
│ DAY 37 ACTIONS                                                         │
├────────────────────────────────────────────────────────────────────────┤
│ Morning (4h)                                                           │
│ ☐ WebGL optimization                                                   │
│   └── OffscreenCanvas for chart rendering                             │
│   └── Web Worker for data processing                                  │
│ ☐ Audit logging                                                        │
│   └── All config changes logged                                       │
│   └── All backtest runs logged                                        │
│   └── Structured log format (JSON)                                    │
│                                                                        │
│ Afternoon (4h)                                                         │
│ ☐ Health check endpoints                                               │
│   └── GET /health (basic)                                             │
│   └── GET /health/deep (with dependencies)                            │
│ ☐ Monitoring setup                                                     │
│   └── Prometheus metrics endpoint                                     │
│   └── Basic Grafana dashboard                                         │
│ ☐ Final documentation review                                           │
│   └── API reference complete                                          │
│   └── Deployment guide                                                │
│   └── User manual                                                     │
│                                                                        │
│ VERIFICATION:                                                          │
│ ✓ Charts render at 60fps                                              │
│ ✓ Health checks return correct status                                 │
│ ✓ Metrics exposed for monitoring                                      │
│                                                                        │
│ MILESTONE CHECK:                                                       │
│ ✓ M7: Production Ready                                                │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Post-Completion Checklist

### Release Criteria
- [ ] All 312 tasks completed
- [ ] Zero high/critical bugs open
- [ ] E2E test coverage > 80%
- [ ] Performance benchmarks met
- [ ] Security audit passed
- [ ] Documentation complete
- [ ] Docker images published
- [ ] Health checks operational

### Rollback Procedures
| Component | Rollback Method |
|-----------|-----------------|
| Frontend | Revert to previous Docker image |
| Backend | Revert to previous Docker image |
| Config | Restore from config history |
| Model | Load previous checkpoint |

### Support Contacts
| Role | Responsibility |
|------|----------------|
| Backend Lead | API issues, WebSocket problems |
| Frontend Lead | UI bugs, Performance issues |
| DevOps | Deployment, Monitoring |
| ML Engineer | Model loading, CUDA issues |
