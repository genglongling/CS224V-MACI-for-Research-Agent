#!/bin/bash
# Stop all running litgpt models

echo "🛑 Stopping all litgpt models..."

# Find and kill all litgpt processes
PIDS=$(ps aux | grep "litgpt serve" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "✅ No litgpt processes found"
    exit 0
fi

echo "Found litgpt processes: $PIDS"

# Kill each process
for PID in $PIDS; do
    echo "   Killing process $PID..."
    kill $PID
    sleep 1
    
    # Check if process is still running
    if kill -0 $PID 2>/dev/null; then
        echo "   Process $PID still running, using SIGKILL..."
        kill -9 $PID
    else
        echo "   ✅ Process $PID stopped successfully"
    fi
done

# Wait and verify
sleep 2

# Check if any processes are still running
REMAINING=$(ps aux | grep "litgpt serve" | grep -v grep)

if [ -z "$REMAINING" ]; then
    echo "✅ All litgpt models stopped successfully!"
else
    echo "⚠️ Some processes may still be running:"
    echo "$REMAINING"
fi
