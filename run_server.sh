#!/bin/bash
# Simple script to run the HoldOn API server

cd "$(dirname "$0")"
echo "🚀 Starting HoldOn API Server..."
echo "📍 Server will be available at: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 -m src.api.main

