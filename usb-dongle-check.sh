#!/bin/bash
LOG="/var/log/usb-dongle.log"
BASE="/mnt/dongles"
FILE="conf.json"
CONF="/home/conf.json"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

# log "Starting USB dongle config check"
for mp in "$BASE"/*; do
    [ -d "$mp" ] || continue
    INC="$mp/$FILE"
    [ -f "$INC" ] || continue

    # 1. First check if different (sorted keys)
    if ! diff -q <(jq -S . "$INC" 2>/dev/null) <(jq -S . "$CONF" 2>/dev/null) >/dev/null 2>&1; then
        # 2. Then validate incoming JSON
        if jq empty "$INC" >/dev/null 2>&1; then
            log "✅ New configuration found on dongle: $(basename "$mp")"
            cp "$INC" "$CONF"
            log "   → Updated $CONF from $(basename "$mp")"
        else
            log "❌ Invalid JSON on dongle: $(basename "$mp")"
        fi
    fi
done
