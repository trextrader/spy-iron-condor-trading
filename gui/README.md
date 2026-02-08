# CondorBrain GUI

**Phase 6 - Neural Trading System Dashboard**

A full-stack dashboard for visualizing, configuring, and controlling the CondorBrain/CondorNet trading system with real-time training telemetry.

## Quick Start

### Lightning AI (Recommended)

```bash
# Start backend
uvicorn gui.backend.main:app --host 0.0.0.0 --port 8000 &

# Start frontend
cd gui/frontend && npm run dev -- --host 0.0.0.0

# Access via Lightning AI URLs:
# Frontend: https://5173-{cloudspace}.cloudspaces.litng.ai
# Backend:  https://8000-{cloudspace}.cloudspaces.litng.ai
```

### Local Development

**Backend:**
```bash
cd gui/backend
pip install -e ".[dev]"
uvicorn gui.backend.main:app --reload --port 8000
```

**Frontend:**
```bash
cd gui/frontend
npm install
npm run dev
```

**Access:**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Architecture

```
gui/
├── backend/                 # FastAPI Python backend
│   ├── main.py             # Application entry point
│   ├── config.py           # Settings and configuration
│   ├── routers/
│   │   ├── backtest.py     # Backtest endpoints
│   │   ├── config.py       # Configuration endpoints
│   │   ├── training.py     # Training telemetry endpoints
│   │   └── websocket.py    # WebSocket manager
│   ├── schemas/            # Pydantic models
│   └── services/
│       └── training_emitter.py  # HTTP-based telemetry emitter
├── frontend/               # React TypeScript frontend
│   ├── src/
│   │   ├── api/           # API client layer
│   │   ├── components/
│   │   │   ├── dashboard/  # Dashboard components
│   │   │   ├── introspection/  # Model introspection charts
│   │   │   ├── training/   # Real-time training components
│   │   │   └── layout/     # Sidebar, Header, Layout
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts
│   │   │   └── useTrainingTelemetry.ts
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── Training.tsx
│   │   │   ├── ModelIntrospection.tsx
│   │   │   └── ...
│   │   ├── providers/
│   │   │   └── WebSocketProvider.tsx
│   │   └── stores/        # Zustand state management
│   └── ...
└── shared/                 # Shared TypeScript types
```

## Features (Phase 6)

| Phase | Feature | Status |
|-------|---------|--------|
| 6.0 | Project Scaffolding | ✅ Complete |
| 6.1 | Core Infrastructure | ✅ Complete |
| 6.2 | Dashboard Panel | ✅ Complete |
| 6.3 | Intelligence Control Matrix | ✅ Complete |
| 6.4 | Execution Reality Engine | ✅ Complete |
| 6.5 | Backtest Control Panel | ✅ Complete |
| 6.6 | Tape Viewer | 🔲 Stub |
| 6.7 | Model Introspection | ✅ Complete |
| 6.8 | Training Monitor (NEW) | ✅ Complete |
| 6.9 | Trade Explorer | ✅ Complete |
| 6.10 | Optimization Suite | ✅ Complete |
| 6.11 | Polish & Testing | 🔲 Pending |

## Training Page Components

The Training page (`/training`) provides real-time visualization during model training:

| Component | Description |
|-----------|-------------|
| `TrainingHeader` | Large status display with epoch/step/ETA/circular progress |
| `MetricGrid` | Grid of all 12 loss components with live sparklines |
| `StreamingLossChart` | Real-time multi-line loss trajectory (Recharts) |
| `StreamingHeatmap` | Real-time fuzzy gate activation heatmap (Canvas) |
| `DiagnosticsPanel` | Mini charts for LR, gradient norm, scaler |
| `TrainingControls` | Start/Stop simulation buttons |

## Training Telemetry

The training script sends telemetry via HTTP POST to the backend, which broadcasts to WebSocket clients.

### Enable Telemetry

```bash
# Local
python intelligence/condor_train_net.py --gui-telemetry local

# Lightning AI (auto-detects cloudspace URL)
python intelligence/condor_train_net.py --gui-telemetry lightai

# Kaggle/Colab (set environment variable first)
export KAGGLE_BACKEND_URL="https://your-ngrok-url.ngrok.io"
python intelligence/condor_train_net.py --gui-telemetry kaggle
```

### Telemetry Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /api/training/telemetry/step` | Per-step metrics (12 loss components) |
| `POST /api/training/telemetry/status` | Progress updates |
| `POST /api/training/telemetry/epoch` | Epoch summaries |
| `POST /api/training/telemetry/complete` | Training complete |
| `POST /api/training/telemetry/fuzzy` | Fuzzy gate activations |

### WebSocket Channels

| Channel | Data |
|---------|------|
| `training.step` | Step metrics |
| `training.status` | Progress |
| `training.epoch` | Epoch summary |
| `training.complete` | Completion |
| `training.fuzzy` | Fuzzy activations |

## Model Introspection

The Model Introspection page (`/model`) displays post-training diagnostics from saved JSON files:

| Tab | Description |
|-----|-------------|
| Loss Trajectories | Multi-line chart of all 12 loss components with smoothing |
| Fuzzy Heatmap | Canvas-based heatmap of fuzzy gate activations |
| Gate Distributions | Histogram distributions per gate per epoch |
| Epoch Summary | Per-epoch aggregated metrics with trends |

### Enable Diagnostics

```bash
python intelligence/condor_train_net.py --save-diagnostics
# Saves to: models/diagnostics/condornet_{timestamp}.json
```

## Epoch Checkpointing

Save model checkpoints every N epochs:

```bash
python intelligence/condor_train_net.py \
  --checkpoint-every 1 \
  --checkpoint-dir models/checkpoints

# Creates:
# models/checkpoints/condornet_epoch_001.pt
# models/checkpoints/condornet_epoch_002.pt
# ...
```

Checkpoints include optimizer and scheduler state for training resumption.

## API Endpoints

### Configuration
- `GET /api/config/current` - Get current configuration
- `POST /api/config/update` - Update configuration
- `POST /api/config/toggle` - Toggle a component

### Backtest
- `POST /api/backtest/run` - Start a backtest
- `GET /api/backtest/status/{run_id}` - Get status
- `GET /api/backtest/results/{run_id}` - Get results

### Training Diagnostics
- `GET /api/training/diagnostics/list` - List saved runs
- `GET /api/training/diagnostics/{filename}` - Get full diagnostics
- `GET /api/training/diagnostics/{filename}/loss-trajectory` - Smoothed loss
- `GET /api/training/diagnostics/{filename}/fuzzy-heatmap` - Gate data
- `GET /api/training/diagnostics/{filename}/epoch-summary` - Per-epoch stats

### Simulation (Testing)
- `POST /api/training/simulate/start` - Start mock training
- `POST /api/training/simulate/stop` - Stop simulation
- `GET /api/training/simulate/status` - Check status

### WebSocket
- `ws://localhost:8000/ws` - Real-time updates

## Tech Stack

### Backend
- **FastAPI** - Modern async Python web framework
- **Pydantic v2** - Data validation and settings
- **WebSockets** - Real-time progress updates

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type safety
- **TailwindCSS** - Utility-first styling
- **Zustand** - Lightweight state management
- **React Query** - Server state management
- **Recharts** - Charting (LineChart, AreaChart, BarChart)
- **Canvas API** - High-performance heatmaps

## Development

### Linting & Formatting

```bash
# Backend
cd gui/backend
black .
ruff check .

# Frontend
cd gui/frontend
npm run lint
npm run format:check
```

### Testing

```bash
# Backend
cd gui/backend
pytest tests/ -v

# Frontend
cd gui/frontend
npm run test
```

## License

MIT
