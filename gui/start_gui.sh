#!/bin/bash
# CondorBrain GUI Startup Script
# Run from project root: ./gui/start_gui.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🦅 Starting CondorBrain GUI..."
echo "Project root: $PROJECT_ROOT"

# Kill any existing processes on ports 8000 and 5173
echo "Cleaning up old processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:5173 | xargs kill -9 2>/dev/null || true

# Start backend in background
echo "🚀 Starting backend on port 8000..."
uvicorn gui.backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to be ready
echo "Waiting for backend..."
sleep 3

# Check backend health
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend healthy"
else
    echo "❌ Backend failed to start"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# Start frontend
echo "🎨 Starting frontend on port 5173..."
cd gui/frontend
npm run dev -- --host 0.0.0.0 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

cd "$PROJECT_ROOT"

echo ""
echo "=========================================="
echo "🦅 CondorBrain GUI is running!"
echo "=========================================="
echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Lightning AI URLs:"
echo "Frontend: https://5173-\$LIGHTNING_CLOUDSPACE_HOST/"
echo "Backend:  https://8000-\$LIGHTNING_CLOUDSPACE_HOST/"
echo ""
echo "To test training simulation:"
echo "curl -X POST 'http://localhost:8000/api/training/simulate/start?epochs=3&steps_per_epoch=50&delay_ms=200'"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Trap to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

# Wait for processes
wait
