#!/bin/bash

# Start the first process
streamlit run timeline.py --server.port 3000 --server.headless true &

# Start the second process
streamlit run admin.py --server.port 4000 --server.headless true &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?