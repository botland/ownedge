#!/bin/bash
sleep 10   # Give USB devices time to appear

for dev in /dev/sd?[0-9] /dev/sd??[0-9] 2>/dev/null; do
    [ -b "$dev" ] || continue
    if udevadm info -q property "$dev" 2>/dev/null | grep -q "ID_BUS=usb"; then
        /usr/local/bin/usb-dongle-mount.sh "$(basename "$dev")"
    fi
done
