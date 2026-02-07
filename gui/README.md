# CondorBrain GUI

**Phase 6 - Neural Trading System Dashboard**

A full-stack dashboard for visualizing, configuring, and controlling the CondorBrain/CondorNet trading system.

## Quick Start

### Development (Docker)

```bash
# Start both backend and frontend
cd gui
docker-compose -f docker-compose.dev.yml up

# Access:
# - Frontend: http://localhost:5173
# - Backend API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

### Development (Local)

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

## Architecture

```
gui/
├── backend/           # FastAPI Python backend
│   ├── main.py       # Application entry point
│   ├── config.py     # Settings and configuration
│   ├── routers/      # API endpoints
│   ├── schemas/      # Pydantic models
│   └── services/     # Business logic
├── frontend/          # React TypeScript frontend
│   ├── src/
│   │   ├── api/      # API client layer
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── pages/
│   │   └── stores/   # Zustand state management
│   └── ...
├── shared/            # Shared TypeScript types
└── docker-compose.dev.yml
```

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
- **Recharts** - Charting (basic charts)
- **lightweight-charts** - WebGL trading charts

## Features (Phase 6)

| Phase | Feature | Status |
|-------|---------|--------|
| 6.0 | Project Scaffolding | ✅ Complete |
| 6.1 | Core Infrastructure | ✅ Complete |
| 6.2 | Dashboard Panel | 🔲 Pending |
| 6.3 | Intelligence Control Matrix | 🔲 Pending |
| 6.4 | Execution Reality Engine | 🔲 Pending |
| 6.5 | Backtest Control Panel | 🔲 Pending |
| 6.6 | Tape Viewer | 🔲 Pending |
| 6.7 | Model Introspection | 🔲 Pending |
| 6.8 | Trade Explorer | 🔲 Pending |
| 6.9 | Optimization Suite | 🔲 Pending |
| 6.10 | Polish & Testing | 🔲 Pending |
| 6.11 | Deployment Readiness | 🔲 Pending |
| 6.12 | GPU Optimization | 🔲 Pending |
| 6.13 | Security & Access | 🔲 Pending |

## API Endpoints

### Configuration
- `GET /api/config/current` - Get current configuration
- `POST /api/config/update` - Update configuration
- `POST /api/config/toggle` - Toggle a component

### Backtest
- `POST /api/backtest/run` - Start a backtest
- `GET /api/backtest/status/{run_id}` - Get status
- `GET /api/backtest/results/{run_id}` - Get results
- `DELETE /api/backtest/{run_id}` - Cancel backtest

### WebSocket
- `ws://localhost:8000/ws` - Real-time updates
  - Subscribe to channels: `backtest.progress`, `optimization.progress`

## Development

### Linting & Formatting

```bash
# Backend
cd gui/backend
black .
ruff check .
mypy .

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
