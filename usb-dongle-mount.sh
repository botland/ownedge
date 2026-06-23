#!/bin/bash
DEV="$1"
BASE="/mnt/dongles"
LOG="/var/log/usb-dongle.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> ${LOG}; }
[ -b "/dev/$DEV" ] || exit 1

# Get filesystem label or use device name
#LABEL=$(blkid -o value -s LABEL "/dev/$DEV" 2>/dev/null || echo "usb")
MOUNTPOINT="$BASE/${DEV}"
log "dev=${DEV}"
mkdir -p "$MOUNTPOINT"
if mountpoint -q "$MOUNTPOINT"; then
    log "Already mounted: $DEV → $MOUNTPOINT"
    exit 0
fi

if mount -o rw,uid=1000,gid=1000 "/dev/$DEV" "$MOUNTPOINT" 2>/dev/null; then
    log "Mounted: $DEV → $MOUNTPOINT"
else
    log "Failed to mount $DEV"
fi
