#!/bin/bash

echo "Stopping CUDA Matrix Multiplication Benchmark application..."

# Kill any existing servers
kill $(lsof -t -i:3000) 2>/dev/null || true
kill $(lsof -t -i:3005) 2>/dev/null || true

echo "All servers stopped."