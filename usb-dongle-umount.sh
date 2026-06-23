#!/bin/sh
DEV="$1"
BASE="/mnt/dongles"
LOG="/var/log/usb-dongle.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> ${LOG}; }
log "USB removed: /dev/$DEV"

# Find and unmount any mountpoint that was using this device
for mp in "$BASE"/*; do
    [ -d "$mp" ] || continue
    
    if mountpoint -q "$mp" && grep -q "/dev/$DEV" /proc/mounts 2>/dev/null; then
        log "Unmounting: $mp"
        umount -l "$mp" 2>/dev/null && log "Successfully unmounted $mp"
        # Optional: remove empty directory
        rmdir "$mp" 2>/dev/null || true
    fi
done
