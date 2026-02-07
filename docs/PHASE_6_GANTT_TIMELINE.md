# Phase 6 GUI Implementation - Gantt Timeline

## Project Overview
- **Total Duration**: 37 Working Days
- **Total Tasks**: 312 Tasks
- **Start Date**: Day 1 (T+0)
- **Target Completion**: Day 37 (T+36)

---

## Visual Gantt Chart

```
PHASE                          | Day 1-5 | Day 6-10 | Day 11-15 | Day 16-20 | Day 21-25 | Day 26-30 | Day 31-37 |
-------------------------------|---------|----------|-----------|-----------|-----------|-----------|-----------|
6.0  Scaffolding (CRITICAL)    | ████░░  |          |           |           |           |           |           |
6.1  Core Infra (CRITICAL)     |   ██████|██░░░░░░░ |           |           |           |           |           |
6.2  Dashboard (CRITICAL)      |         |  ████████|█░░░░░░░░░ |           |           |           |           |
6.3  ICM Panel (HIGH)          |         |          | ██████░░░ |           |           |           |           |
6.4  Exec Reality (HIGH)       |         |          |    ███████|░░░░░░░░░░ |           |           |           |
6.5  Backtest Panel (CRITICAL) |         |          |           | ██████████|░░░░░░░░░░ |           |           |
6.6  Tape Viewer (MEDIUM)      |         |          |           |           | ████░░░░░ |           |           |
6.7  Model Introspect (MEDIUM) |         |          |           |           |   ████░░░ |           |           |
6.8  Trade Explorer (MEDIUM)   |         |          |           |           |      █████|░░░░░░░░░░ |           |
6.9  Optimization (HIGH)       |         |          |           |           |           | ██████████|░░░░░░░░░░ |
6.10 Polish & Test (MEDIUM)    |         |          |           |           |           |           | ████░░░░░ |
6.11 Deployment (HIGH)         |         |          |           |           |           |           |    ███░░░ |
6.12 GPU Optimization (HIGH)   |         |          |           |           |           |           |      ██░░ |
6.13 Security (MEDIUM)         |         |          |           |           |           |           |       ██░ |

Legend: █ = Active Development  ░ = Buffer/Overlap
```

---

## Detailed Phase Timeline

### Phase 6.0 - Project Scaffolding [CRITICAL]
```
Days 1-3 (3 days)
├── Day 1: Repository structure, Monorepo setup, CI/CD scaffolding
├── Day 2: Docker configs, ESLint/Prettier, Git hooks
└── Day 3: Documentation templates, Dev environment scripts
```
| Task Group | Est. Hours | Day Start | Day End |
|------------|------------|-----------|---------|
| Repo Structure | 4h | 1 | 1 |
| Tool Config | 6h | 1 | 2 |
| CI/CD Setup | 6h | 2 | 3 |
| Documentation | 4h | 3 | 3 |

---

### Phase 6.1 - Core Infrastructure [CRITICAL]
```
Days 3-7 (5 days)
├── Day 3-4: FastAPI app skeleton, Pydantic schemas
├── Day 5: Config service, WebSocket manager
├── Day 6: React app bootstrap, Zustand stores
└── Day 7: API client layer, Error boundaries
```
| Task Group | Est. Hours | Day Start | Day End |
|------------|------------|-----------|---------|
| Backend Skeleton | 8h | 3 | 4 |
| Pydantic Schemas | 8h | 4 | 5 |
| Config Engine | 6h | 5 | 5 |
| WebSocket Manager | 4h | 5 | 6 |
| React Bootstrap | 6h | 6 | 6 |
| State Management | 6h | 6 | 7 |
| API Integration | 6h | 7 | 7 |

---

### Phase 6.2 - Dashboard Panel [CRITICAL]
```
Days 7-11 (5 days)
├── Day 7-8: Metric cards, Status indicators
├── Day 9: Equity curve mini-chart
├── Day 10: Activity feed, Recent runs list
└── Day 11: Responsive layout, Loading states
```
| Task Group | Est. Hours | Day Start | Day End |
|------------|------------|-----------|---------|
| Metric Cards | 6h | 7 | 8 |
| Status Grid | 6h | 8 | 8 |
| Equity Mini-Chart | 8h | 9 | 9 |
| Activity Components | 6h | 10 | 10 |
| Layout Polish | 6h | 11 | 11 |

---

### Phase 6.3 - Intelligence Control Matrix [HIGH]
```
Days 11-14 (4 days)
├── Day 11-12: Component toggle tree, Enable/disable logic
├── Day 13: Dependency resolution, Config persistence
└── Day 14: Visual feedback, Batch operations
```
| Task Group | Est. Hours | Day Start | Day End |
|------------|------------|-----------|---------|
| Toggle Tree UI | 8h | 11 | 12 |
| State Management | 6h | 12 | 13 |
| Dependency Logic | 6h | 13 | 13 |
| Batch Controls | 6h | 14 | 14 |

---

### Phase 6.4 - Execution Reality Engine [HIGH]
```
Days 14-17 (4 days)
├── Day 14-15: 8-component configuration cards
├── Day 16: Monte Carlo visualization
└── Day 17: Profile management, Presets
```
| Task Group | Est. Hours | Day Start | Day End |
|------------|------------|-----------|---------|
| Component Cards | 10h | 14 | 15 |
| Slider Controls | 6h | 15 | 16 |
| MC Visualization | 8h | 16 | 17 |
| Profile CRUD | 6h | 17 | 17 |

---

### Phase 6.5 - Backtest Control Panel [CRITICAL]
```
Days 17-21 (5 days)
├── Day 17-18: Run configuration form
├── Day 19: Progress tracking, WebSocket updates
├── Day 20: Results display, Metrics dashboard
├── Day 21: Comparison panel, Replay controls
```
| Task Group | Est. Hours | Day Start | Day End |
|------------|------------|-----------|---------|
| Config Form | 8h | 17 | 18 |
| Progress UI | 6h | 19 | 19 |
| WebSocket Integration | 6h | 19 | 20 |
| Results Display | 8h | 20 | 20 |
| Comparison Panel | 6h | 21 | 21 |
| Replay Controls | 4h | 21 | 21 |

---

### Phase 6.6 - Tape Viewer [MEDIUM]
```
Days 21-23 (3 days)
├── Day 21-22: Data grid with virtualization
├── Day 22: Filtering, Search
└── Day 23: Annotation layer, Export
```
| Task Group | Est. Hours | Day Start | Day End |
|------------|------------|-----------|---------|
| Data Grid | 8h | 21 | 22 |
| Virtualization | 4h | 22 | 22 |
| Filter/Search | 6h | 22 | 23 |
| Annotation Layer | 6h | 23 | 23 |

---

### Phase 6.7 - Model Introspection [MEDIUM]
```
Days 23-25 (3 days)
├── Day 23-24: Architecture viewer, Layer diagram
├── Day 24: Weight heatmaps
└── Day 25: Version diff viewer, Attention visualization
```
| Task Group | Est. Hours | Day Start | Day End |
|------------|------------|-----------|---------|
| Architecture View | 8h | 23 | 24 |
| Weight Heatmaps | 6h | 24 | 24 |
| Version Diff | 6h | 25 | 25 |
| Attention Viz | 4h | 25 | 25 |

---

### Phase 6.8 - Trade Explorer [MEDIUM]
```
Days 25-28 (4 days)
├── Day 25-26: Trade table with sorting/filtering
├── Day 27: Trade detail modal, P&L breakdown
└── Day 28: Statistical analysis, Distribution charts
```
| Task Group | Est. Hours | Day Start | Day End |
|------------|------------|-----------|---------|
| Trade Table | 8h | 25 | 26 |
| Detail Modal | 6h | 27 | 27 |
| P&L Components | 6h | 27 | 28 |
| Analytics Charts | 6h | 28 | 28 |

---

### Phase 6.9 - Optimization Suite [HIGH]
```
Days 28-32 (5 days)
├── Day 28-29: Ablation study UI
├── Day 30: Bayesian optimization controls
├── Day 31: Result visualization, Pareto charts
└── Day 32: Certification workflow
```
| Task Group | Est. Hours | Day Start | Day End |
|------------|------------|-----------|---------|
| Ablation UI | 8h | 28 | 29 |
| Bayesian Controls | 8h | 30 | 30 |
| Result Viz | 8h | 31 | 31 |
| Pareto Charts | 6h | 31 | 32 |
| Certification | 6h | 32 | 32 |

---

### Phase 6.10 - Polish & Testing [MEDIUM]
```
Days 32-34 (3 days)
├── Day 32-33: Accessibility audit, Keyboard navigation
├── Day 33: Error handling, Edge cases
└── Day 34: E2E tests, Integration tests
```
| Task Group | Est. Hours | Day Start | Day End |
|------------|------------|-----------|---------|
| A11y Audit | 6h | 32 | 33 |
| Error Handling | 6h | 33 | 33 |
| E2E Tests | 8h | 34 | 34 |
| Integration Tests | 6h | 34 | 34 |

---

### Phase 6.11 - Deployment Readiness [HIGH]
```
Days 34-36 (3 days)
├── Day 34-35: Docker production builds
├── Day 35: Environment configuration
└── Day 36: Health checks, Monitoring setup
```
| Task Group | Est. Hours | Day Start | Day End |
|------------|------------|-----------|---------|
| Docker Builds | 8h | 34 | 35 |
| Env Config | 6h | 35 | 35 |
| Health Checks | 4h | 36 | 36 |
| Monitoring | 6h | 36 | 36 |

---

### Phase 6.12 - Performance & GPU Optimization [HIGH]
```
Days 35-37 (3 days)
├── Day 35: TorchScript compilation, Model optimization
├── Day 36: CUDA Graphs, FP16 inference
└── Day 37: WebGL optimization, Web Workers
```
| Task Group | Est. Hours | Day Start | Day End |
|------------|------------|-----------|---------|
| TorchScript | 6h | 35 | 35 |
| CUDA Graphs | 6h | 36 | 36 |
| FP16 Inference | 4h | 36 | 36 |
| WebGL Optimize | 6h | 37 | 37 |
| Web Workers | 4h | 37 | 37 |

---

### Phase 6.13 - Security & Access Control [MEDIUM]
```
Days 36-37 (2 days)
├── Day 36: API key management, Input sanitization
└── Day 37: Rate limiting, Audit logging
```
| Task Group | Est. Hours | Day Start | Day End |
|------------|------------|-----------|---------|
| API Key Mgmt | 6h | 36 | 36 |
| Input Sanitation | 4h | 36 | 37 |
| Rate Limiting | 4h | 37 | 37 |
| Audit Logging | 4h | 37 | 37 |

---

## Milestone Summary

| Milestone | Target Day | Dependencies Met |
|-----------|------------|------------------|
| M1: Infrastructure Ready | Day 7 | 6.0 + 6.1 complete |
| M2: Dashboard Live | Day 11 | M1 + 6.2 complete |
| M3: Control Panels Functional | Day 17 | M2 + 6.3 + 6.4 complete |
| M4: Backtest Workflow Complete | Day 21 | M3 + 6.5 complete |
| M5: Data Views Ready | Day 25 | M4 + 6.6 + 6.7 complete |
| M6: Full Feature Parity | Day 32 | M5 + 6.8 + 6.9 complete |
| M7: Production Ready | Day 37 | All phases complete |

---

## Critical Path Analysis

The **critical path** (longest sequence determining project duration):

```
6.0 Scaffolding (3d)
    └── 6.1 Core Infra (5d)
        └── 6.2 Dashboard (5d)
            └── 6.5 Backtest Panel (5d) ← Depends on Dashboard + ICM
                └── 6.9 Optimization (5d) ← Depends on Backtest
                    └── 6.11 Deployment (3d)
                        └── 6.12 GPU Opt (3d)

Critical Path Duration: 29 days
Float Available: 8 days
```

---

## Resource Allocation (Recommended)

| Role | Phases | Utilization |
|------|--------|-------------|
| Backend Lead | 6.0, 6.1, 6.4, 6.9, 6.11, 6.12 | 100% |
| Frontend Lead | 6.1, 6.2, 6.3, 6.5, 6.8, 6.10 | 100% |
| Full-Stack Dev | 6.5, 6.6, 6.7, 6.8 | 85% |
| DevOps | 6.0, 6.11, 6.12, 6.13 | 60% |

---

## Risk Buffers

| Risk Area | Buffer Days | Applied To |
|-----------|-------------|------------|
| WebSocket complexity | +1 day | Phase 6.1 |
| Chart performance | +1 day | Phases 6.2, 6.9 |
| GPU integration | +2 days | Phase 6.12 |
| Security audit findings | +1 day | Phase 6.13 |

**Total Risk Buffer**: 5 days (built into 37-day estimate)
