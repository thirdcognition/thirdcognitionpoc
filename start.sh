#!/bin/bash

# Check if we are in a Docker container
if [ -f /.dockerenv ]; then
    echo "Running inside Docker. Skipping venv creation."
else
    # python3 -m venv .venv
    echo "Not running inside Docker. Creating virtual environment..."
    source .venv/bin/activate
fi

# Start the first process
streamlit run poctimeline/timeline/timeline.py --server.port 3500 --server.headless true --server.address=0.0.0.0 --client.showErrorDetails=False --client.toolbarMode=minimal --server.fileWatcherType=none &

# Start the second process
streamlit run poctimeline/admin/Admin_Home.py --server.port 4000 --server.headless true --server.address=0.0.0.0 --client.showErrorDetails=False --client.toolbarMode=minimal --server.fileWatcherType=none &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?