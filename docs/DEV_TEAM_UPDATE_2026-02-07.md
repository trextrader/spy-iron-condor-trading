# Dev Team Update - February 7, 2026

## Release: v2.3.0 - CondorBrain GUI & Training Visualization

---

## Executive Summary

This release introduces the **CondorBrain GUI** - a full-stack real-time training dashboard with streaming metrics, loss charts, and fuzzy gate heatmaps. Key enhancements include environment-aware telemetry, epoch checkpointing, and comprehensive documentation updates.

---

## What's New

### 1. Training Page (Real-time Visualization)

A dedicated `/training` page in the GUI sidebar provides real-time visualization during model training:

| Component | Description |
|-----------|-------------|
| **TrainingHeader** | Large status display with epoch/step counters, circular progress ring, ETA countdown |
| **MetricGrid** | Grid of all 12 loss components with live sparklines |
| **StreamingLossChart** | Real-time multi-line Recharts LineChart with component toggles |
| **StreamingHeatmap** | Canvas-based fuzzy gate activation heatmap (magma colormap) |
| **DiagnosticsPanel** | Mini AreaCharts for learning rate, gradient norm, scaler scale |
| **TrainingControls** | Start/Stop simulation buttons for testing |

**12 Loss Components Tracked:**
- `loss`, `mse`, `npdd`, `sharpe`, `dd`, `turnover`
- `fuzzy`, `pattern_ent`, `group_inv`, `rho`, `energy`, `growth`

### 2. Environment-Aware Telemetry

The `--gui-telemetry` flag now accepts environment arguments for automatic URL construction:

```bash
--gui-telemetry local     # http://localhost:8000
--gui-telemetry lightai   # Auto-detects from LIGHTNING_CLOUDSPACE_HOST
--gui-telemetry kaggle    # Uses KAGGLE_BACKEND_URL env var
--gui-telemetry colab     # Uses COLAB_BACKEND_URL env var
```

### 3. Epoch Checkpointing

New flags for saving model checkpoints every N epochs (regardless of improvement):

```bash
--checkpoint-every N      # Save checkpoint every N epochs
--checkpoint-dir PATH     # Directory for checkpoints (default: models/checkpoints)
```

Checkpoints include full state for training resumption:
- Model weights
- Optimizer state
- Scheduler state
- Epoch number

### 4. Diagnostics Export

```bash
--save-diagnostics        # Saves JSON for Model Introspection page
```

Saves to: `models/diagnostics/condornet_{timestamp}.json`

---

## Quick Start Commands

### Local Development

**Terminal 1 - Backend:**
```bash
cd C:\SPYOptionTrader_test
uvicorn gui.backend.main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd C:\SPYOptionTrader_test\gui\frontend
npm install  # First time only
npm run dev
```

**Terminal 3 - Training with Telemetry:**
```bash
python intelligence/condor_train_net.py \
  --config configs/condor_net_config_v46.yaml \
  --gui-telemetry local \
  --save-diagnostics \
  --checkpoint-every 1
```

**Access:**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Lightning AI Deployment

**Start GUI:**
```bash
# Terminal 1: Backend
cd /teamspace/studios/this_studio/SPYOptionTrader_test
uvicorn gui.backend.main:app --host 0.0.0.0 --port 8000 &

# Terminal 2: Frontend
cd /teamspace/studios/this_studio/SPYOptionTrader_test/gui/frontend
npm run dev -- --host 0.0.0.0
```

**Access URLs (auto-detected):**
- Frontend: `https://5173-{LIGHTNING_CLOUDSPACE_HOST}/`
- Backend: `https://8000-{LIGHTNING_CLOUDSPACE_HOST}/`

**Training with Telemetry:**
```bash
python intelligence/condor_train_net.py \
  --config configs/condor_net_config_v46.yaml \
  --gui-telemetry lightai \
  --save-diagnostics \
  --checkpoint-every 1 \
  --checkpoint-dir models/checkpoints
```

### Kaggle/Colab (with ngrok)

```bash
# First, set up ngrok tunnel to your backend
export KAGGLE_BACKEND_URL="https://your-ngrok-url.ngrok.io"

python intelligence/condor_train_net.py \
  --config configs/condor_net_config_v46.yaml \
  --gui-telemetry kaggle \
  --save-diagnostics
```

---

## Full Training Command Reference

### Standard Training (No GUI)
```bash
python intelligence/condor_train_net.py \
  --config configs/condor_net_config_v46.yaml
```

### Training with All Features
```bash
python intelligence/condor_train_net.py \
  --config configs/condor_net_config_v46.yaml \
  --gui-telemetry lightai \
  --save-diagnostics \
  --checkpoint-every 1 \
  --checkpoint-dir models/checkpoints
```

### Fast-Track Training (3M+ rows)
```bash
bash run_training.sh
# Or manually:
python intelligence/train_condor_brain.py \
  --local-data "data/processed/mamba_institutional_2025_1m_v22.csv" \
  --output models/cb_v22.pth \
  --max-rows 3000000 \
  --cde \
  --use-predicate-discovery \
  --predicate-slots 2048 \
  --epochs 20 \
  --batch-size 128 \
  --accum-steps 2 \
  --save-on-batch-loss 1.0 \
  --val-limit 200
```

---

## New CLI Flags Summary

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--gui-telemetry` | `local\|lightai\|kaggle\|colab` | None | Enable real-time telemetry to GUI |
| `--save-diagnostics` | flag | False | Save JSON for Model Introspection |
| `--checkpoint-every` | int | None | Save checkpoint every N epochs |
| `--checkpoint-dir` | path | `models/checkpoints` | Directory for epoch checkpoints |

---

## WebSocket Telemetry Endpoints

The training script sends HTTP POST requests to the backend, which broadcasts via WebSocket:

| Endpoint | Frequency | Data |
|----------|-----------|------|
| `POST /api/training/telemetry/step` | Every 10 steps | 12 loss components, LR, grad norm |
| `POST /api/training/telemetry/status` | On change | Epoch, step, progress %, ETA |
| `POST /api/training/telemetry/epoch` | End of epoch | Epoch summary metrics |
| `POST /api/training/telemetry/fuzzy` | Every 10 steps | Fuzzy gate activations (10 gates) |
| `POST /api/training/telemetry/complete` | End of training | Final summary |

**WebSocket Channels:**
- `training.step` - Step metrics
- `training.status` - Progress updates
- `training.epoch` - Epoch summaries
- `training.fuzzy` - Fuzzy gate activations
- `training.complete` - Training completion

---

## Files Changed

### New Files Created

```
gui/frontend/src/components/training/
├── index.ts              # Barrel exports
├── TrainingHeader.tsx    # Status header with progress
├── LiveMetricCard.tsx    # Single metric with sparkline
├── MetricGrid.tsx        # 12-metric grid
├── StreamingLossChart.tsx # Real-time loss chart
├── StreamingHeatmap.tsx  # Fuzzy gate heatmap
├── DiagnosticsPanel.tsx  # LR/grad/scaler charts
└── TrainingControls.tsx  # Start/Stop buttons

gui/frontend/src/pages/
└── Training.tsx          # Main Training page
```

### Files Modified

| File | Change |
|------|--------|
| `gui/frontend/src/App.tsx` | Added lazy import + route for `/training` |
| `gui/frontend/src/components/layout/Sidebar.tsx` | Added Training nav item |
| `gui/frontend/src/hooks/useWebSocket.ts` | Lightning AI URL auto-detection |
| `gui/backend/routers/training.py` | Added `/telemetry/fuzzy` endpoint |
| `gui/backend/services/training_emitter.py` | URL change detection |
| `intelligence/condor_train_net.py` | Environment telemetry + checkpointing |

### Documentation Updated

| File | Change |
|------|--------|
| `CHANGELOG.md` | Added v2.3.0 release notes |
| `gui/README.md` | Complete rewrite with all features |
| `docs/LIGHTNING_AI.md` | Added GUI section |
| `docs/walkthrough.md` | Added Phase 6 section |
| `docs/PHASE_6_GUI_TASK_LIST.md` | Updated completion status |
| `docs/scientific_spec.md` | Renumbered section 13 → 19 |

---

## Architecture Diagrams

### New Diagrams Created

| Diagram | Description |
|---------|-------------|
| [gui_architecture.png](architecture/gui_architecture.png) | Complete GUI system architecture showing frontend, backend, WebSocket |
| [training_telemetry_flow.png](architecture/training_telemetry_flow.png) | Real-time data flow from training script to frontend |
| [training_components.png](architecture/training_components.png) | Detailed React component breakdown for Training page |
| [complete_system_v23.png](architecture/complete_system_v23.png) | Full system architecture v2.3 with GUI layer |
| [epoch_checkpointing.png](architecture/epoch_checkpointing.png) | Epoch checkpointing system for training resumption |

### Data Flow Overview

```
Training Script (Python)
    │ HTTP POST every 10 steps
    ▼
Backend /api/training/telemetry/*
    │ WebSocket broadcast
    ▼
useTrainingTelemetry hook
    │ Maintains stepHistory (max 1000)
    ▼
Training.tsx page
    │ Distributes to components
    ▼
┌─────────────┬─────────────┬─────────────┐
│ MetricGrid  │ LossChart   │ Heatmap     │
│ (12 cards)  │ (Recharts)  │ (Canvas)    │
└─────────────┴─────────────┴─────────────┘
```

---

## Testing the Training Page

1. Start backend and frontend (see Quick Start above)
2. Navigate to http://localhost:5173/training
3. Click **"Start Simulation"** to see mock training data
4. Or run actual training with `--gui-telemetry local`

**Verify:**
- [ ] All 12 metric cards update with sparklines
- [ ] Loss trajectory chart streams new data points
- [ ] Heatmap shows fuzzy gate activations (10 rows)
- [ ] Progress bar and ETA update correctly
- [ ] Diagnostics panel shows LR, gradient norm, scaler

---

## Phase Completion Status

| Phase | Description | Status |
|-------|-------------|--------|
| 2.0 | Multi-Timeframe Data Pipeline | ✅ Complete |
| 2.5 | Lag-Aware Alignment System | ✅ Complete |
| 3.0 | Neural CDE Architecture | ✅ Complete |
| 4.0 | CondorNet Unified Architecture | ✅ Complete |
| 5.0 | Training Stability & Audit | ✅ Complete |
| 6.0 | CondorBrain GUI | ✅ Complete |
| 6.8 | Training Monitor | ✅ Complete |
| 6.11 | Polish & Testing | 🔲 Pending |

---

## Known Issues / Next Steps

1. **Tape Viewer** (6.6) - Still stub, not prioritized
2. **Polish & Testing** (6.11) - Unit tests for frontend components
3. **Mobile Responsiveness** - Training page optimized for desktop

---

## Questions?

Refer to:
- `gui/README.md` - Full GUI documentation
- `docs/LIGHTNING_AI.md` - Cloud deployment instructions
- `docs/walkthrough.md` - System overview

---

*Report generated: 2026-02-07*
*Version: v2.3.0*
