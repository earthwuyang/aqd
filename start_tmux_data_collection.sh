#!/bin/bash
# Start tmux session for long-running AQD data collection

echo "=== Starting AQD Data Collection in tmux ==="

# Set target records from environment or default
TARGET_RECORDS=${AQD_TARGET_RECORDS:-20000}
echo "Target records: $TARGET_RECORDS"

# Create tmux session for data collection
SESSION_NAME="aqd_data_collection"

# Check if session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "⚠️ tmux session '$SESSION_NAME' already exists"
    echo "Attaching to existing session..."
    tmux attach-session -t $SESSION_NAME
    exit 0
fi

# Create new tmux session
echo "Creating new tmux session: $SESSION_NAME"

tmux new-session -d -s $SESSION_NAME -x 120 -y 40

# Set up the environment in the tmux session
tmux send-keys -t $SESSION_NAME "cd /home/wuy/DB/pg_duckdb_postgres" C-m
tmux send-keys -t $SESSION_NAME "export AQD_TARGET_RECORDS=$TARGET_RECORDS" C-m
tmux send-keys -t $SESSION_NAME "export PATH=\$HOME/postgresql/bin:\$PATH" C-m
tmux send-keys -t $SESSION_NAME "export LD_LIBRARY_PATH=\$HOME/postgresql/lib:\$LD_LIBRARY_PATH" C-m

# Create multiple windows for different tasks
# Window 0: Data Collection
tmux rename-window -t $SESSION_NAME:0 "DataCollection"

# Window 1: Progress Monitoring  
tmux new-window -t $SESSION_NAME -n "Monitor"
tmux send-keys -t $SESSION_NAME:Monitor "cd /home/wuy/DB/pg_duckdb_postgres" C-m
tmux send-keys -t $SESSION_NAME:Monitor "echo 'Progress monitoring window - use tail -f logs/real_data_collection.log'" C-m

# Window 2: PostgreSQL Status
tmux new-window -t $SESSION_NAME -n "PostgreSQL"
tmux send-keys -t $SESSION_NAME:PostgreSQL "cd /home/wuy/DB/pg_duckdb_postgres" C-m
tmux send-keys -t $SESSION_NAME:PostgreSQL "export PATH=\$HOME/postgresql/bin:\$PATH" C-m
tmux send-keys -t $SESSION_NAME:PostgreSQL "echo 'PostgreSQL status window - use: pg_ctl -D ~/postgresql_data status'" C-m

# Window 3: Results
tmux new-window -t $SESSION_NAME -n "Results"
tmux send-keys -t $SESSION_NAME:Results "cd /home/wuy/DB/pg_duckdb_postgres" C-m
tmux send-keys -t $SESSION_NAME:Results "echo 'Results window - data files will appear in data/ directory'" C-m

# Go back to data collection window
tmux select-window -t $SESSION_NAME:DataCollection

# Wait for PostgreSQL installation to complete
tmux send-keys -t $SESSION_NAME:DataCollection "echo 'Waiting for PostgreSQL installation to complete...'" C-m
tmux send-keys -t $SESSION_NAME:DataCollection "while [ ! -f \$HOME/postgresql/bin/psql ]; do echo 'PostgreSQL not ready, waiting...'; sleep 10; done" C-m
tmux send-keys -t $SESSION_NAME:DataCollection "echo 'PostgreSQL installation complete!'" C-m

# Start the data collection script
tmux send-keys -t $SESSION_NAME:DataCollection "echo 'Starting real data collection with $TARGET_RECORDS target records...'" C-m
tmux send-keys -t $SESSION_NAME:DataCollection "python3 collect_real_training_data.py" C-m

# Set up status line
tmux set-option -t $SESSION_NAME status-left "[AQD Data Collection] "
tmux set-option -t $SESSION_NAME status-right "Target: $TARGET_RECORDS records | %H:%M %d-%b"

# Display instructions
echo ""
echo "🚀 tmux session '$SESSION_NAME' created and data collection started!"
echo ""
echo "Commands:"
echo "  tmux attach -t $SESSION_NAME    # Attach to session"  
echo "  tmux list-sessions              # List all sessions"
echo "  tmux kill-session -t $SESSION_NAME  # Kill session"
echo ""
echo "Windows:"
echo "  0: DataCollection - Main data collection script"
echo "  1: Monitor - Progress monitoring (use: tail -f logs/real_data_collection.log)"
echo "  2: PostgreSQL - Database status and management"
echo "  3: Results - View collected data and results"
echo ""
echo "Navigation:"
echo "  Ctrl+b, 0-3    # Switch between windows"
echo "  Ctrl+b, d      # Detach from session (keeps running)"
echo "  Ctrl+b, c      # Create new window"
echo "  Ctrl+b, x      # Kill current window"
echo ""
echo "To attach to the session:"
echo "  tmux attach -t $SESSION_NAME"
echo ""

# Optionally auto-attach
read -p "Attach to session now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    tmux attach-session -t $SESSION_NAME
fi