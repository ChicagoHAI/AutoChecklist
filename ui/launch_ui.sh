#!/bin/bash
# Launch AutoChecklist UI (production by default, --dev for development mode)
# Run from ui/ directory: ./launch_ui.sh [--dev]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEV_MODE=false

if [ "$1" = "--dev" ]; then
    DEV_MODE=true
fi

# Clear any inherited VIRTUAL_ENV so uv uses the project's own .venv
unset VIRTUAL_ENV

# Load .env from parent directory
if [ -f "$SCRIPT_DIR/../.env" ]; then
    set -a
    source "$SCRIPT_DIR/../.env"
    set +a
fi

if [ "$DEV_MODE" = true ]; then
    MODE_LABEL="Development"
else
    MODE_LABEL="Production"
fi

echo "Starting AutoChecklist UI ($MODE_LABEL Mode)"
echo "================================================"

# Kill any existing processes on these ports
lsof -ti:7770 | xargs kill -9 2>/dev/null || true
lsof -ti:7771 | xargs kill -9 2>/dev/null || true
fuser -k 7770/tcp 2>/dev/null || true
fuser -k 7771/tcp 2>/dev/null || true
sleep 1

# Install frontend dependencies if needed
if [ ! -d "$SCRIPT_DIR/frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    if [ "$DEV_MODE" = true ]; then
        (cd "$SCRIPT_DIR/frontend" && npm install)
    else
        (cd "$SCRIPT_DIR/frontend" && npm ci)
    fi
fi

# Production: build frontend first
if [ "$DEV_MODE" = false ]; then
    echo "Building frontend for production..."
    (cd "$SCRIPT_DIR/frontend" && npm run build)
fi

# Start backend
echo "Starting backend on http://0.0.0.0:7771..."
if [ "$DEV_MODE" = true ]; then
    (cd "$SCRIPT_DIR/backend" && uv run python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 7771) &
else
    (cd "$SCRIPT_DIR/backend" && uv run python -m uvicorn app.main:app --host 0.0.0.0 --port 7771) &
fi
BACKEND_PID=$!

sleep 2

# Start frontend
echo "Starting frontend on http://localhost:7770..."
if [ "$DEV_MODE" = true ]; then
    (cd "$SCRIPT_DIR/frontend" && npm run dev -- -p 7770) &
else
    (cd "$SCRIPT_DIR/frontend" && npm run start -- -p 7770) &
fi
FRONTEND_PID=$!

# Trap Ctrl+C and kill both processes
cleanup() {
    echo ""
    echo "Shutting down services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

echo ""
echo "================================================"
echo "Services running ($MODE_LABEL mode):"
echo "  Backend:  http://0.0.0.0:7771"
echo "  Frontend: http://localhost:7770"
echo "================================================"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

wait
