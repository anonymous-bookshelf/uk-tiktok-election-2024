#!/bin/bash
# Waits for the current fetch_music_lookup.py run to finish, then starts the next one.
# Run with: nohup bash auto_rerun_music.sh &

WAIT_PID=15064
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OLD_PATH="$ROOT_DIR/data/supplementary/music_lookup.csv"
NEW_PATH="$ROOT_DIR/data/music_lookup/hashtags_music_lookup.csv"

echo "[$(date)] Waiting for PID $WAIT_PID to finish..."
while kill -0 $WAIT_PID 2>/dev/null; do
    sleep 60
done

echo "[$(date)] PID $WAIT_PID finished. Moving output to new path..."
cp "$OLD_PATH" "$NEW_PATH"
echo "[$(date)] Copied $OLD_PATH → $NEW_PATH"

echo "[$(date)] Starting second run..."
source "$ROOT_DIR/../venv/bin/activate"
python "$SCRIPT_DIR/fetch_music_lookup.py" --input "$ROOT_DIR/data/enriched/hashtags_enriched.csv"
echo "[$(date)] Second run complete."
