#!/bin/bash
echo "Monitoring started. Logging CPU/RAM for build_index.py and train.py..."
echo "Timestamp | PID | CPU% | RAM(MB) | Command"

while true; do
    # Get processes, filter for our scripts, format output
    # -e: all processes
    # -o: output format
    processes=$(ps -eo pid,pcpu,rss,command | grep -E "build_index.py|train.py" | grep -v grep)
    
    if [ ! -z "$processes" ]; then
        timestamp=$(date '+%H:%M:%S')
        echo "$processes" | awk -v ts="$timestamp" '
        {
            # rss is in KB. Convert to MB.
            ram_mb = int($3 / 1024);
            # Print simplified line
            printf "[%s] PID: %s | CPU: %s%% | RAM: %s MB | %s %s\n", ts, $1, $2, ram_mb, $4, $5
        }'
    fi
    sleep 5
done
