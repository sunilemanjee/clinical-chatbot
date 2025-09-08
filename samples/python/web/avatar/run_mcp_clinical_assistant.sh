#!/bin/bash

# MCP Clinical Assistant Startup Script
# This script starts both the MCP server and Flask application
# Usage: ./run_mcp_clinical_assistant.sh [-p PORT]

# Default port
DEFAULT_PORT=8080
PORT=$DEFAULT_PORT

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-p PORT]"
            echo "  -p, --port PORT    Specify the port for Flask application (default: $DEFAULT_PORT)"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "🚀 Starting MCP Clinical Assistant on port $PORT..."

# Check if virtual environment exists
if [ ! -d "avatar-env" ]; then
    echo "❌ Virtual environment not found. Please run setup_env.sh first."
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source avatar-env/bin/activate

# Install MCP dependencies if not already installed
echo "📥 Installing MCP dependencies..."
pip install -r requirements_mcp.txt

# Check environment variables
echo "🔍 Checking environment variables..."
if [ -z "$ELASTIC_URL" ] || [ -z "$ELASTIC_API_KEY" ] || [ -z "$ELASTIC_INDEX_NAME" ]; then
    echo "⚠️  Warning: Some Elasticsearch environment variables are not set."
    echo "   Please ensure ELASTIC_URL, ELASTIC_API_KEY, and ELASTIC_INDEX_NAME are configured."
fi

# Start simplified clinical server in background
echo "🔧 Starting clinical server..."
python simple_mcp_server.py &
MCP_PID=$!

# Wait a moment for MCP server to start
sleep 3

# Check if MCP server started successfully
if ! kill -0 $MCP_PID 2>/dev/null; then
    echo "❌ Failed to start MCP server"
    exit 1
fi

echo "✅ Clinical server started (PID: $MCP_PID)"

# Start Flask application
echo "🌐 Starting Flask application on port $PORT..."
export FLASK_APP=app.py
export FLASK_ENV=development
python -m flask run -h 0.0.0.0 -p $PORT &
FLASK_PID=$!

# Wait a moment for Flask to start
sleep 2

# Check if Flask started successfully
if ! kill -0 $FLASK_PID 2>/dev/null; then
    echo "❌ Failed to start Flask application"
    kill $MCP_PID 2>/dev/null
    exit 1
fi

echo "✅ Flask application started (PID: $FLASK_PID)"
echo ""
echo "🎉 Clinical Assistant is running!"
echo "   - Clinical Server: PID $MCP_PID"
echo "   - Flask App: PID $FLASK_PID"
echo "   - Web Interface: http://localhost:$PORT/chat"
echo ""
echo "Press Ctrl+C to stop both services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $MCP_PID 2>/dev/null
    kill $FLASK_PID 2>/dev/null
    echo "✅ Services stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for processes
wait
