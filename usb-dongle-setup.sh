#!/bin/bash
# Setup script for USB dongles on Devuan
echo "=== Setting up USB dongles ==="

# Create /etc/rc.local if it doesn't exist
if [ ! -f /etc/rc.local ]; then
    cat >/etc/rc.local <<EOF
#!/bin/sh -e
# rc.local - run at boot

# Mount USB dongles at boot
/usr/local/bin/usb-dongle-mount-at-boot.sh

exit 0
EOF
    chmod +x /etc/rc.local
fi

# udev rule for hot-plug
cat >/etc/udev/rules.d/99-usb-dongle.rules <<EOF
ACTION=="add", SUBSYSTEM=="block", SUBSYSTEMS=="usb", KERNEL=="sd*[0-9]", ENV{ID_FS_USAGE}=="filesystem", RUN+="/usr/local/bin/usb-dongle-mount.sh %k"
ACTION=="remove", SUBSYSTEM=="block", SUBSYSTEMS=="usb", KERNEL=="sd*[0-9]", RUN+="/usr/local/bin/usb-dongle-umount.sh %k"
EOF
udevadm control --reload-rules
udevadm trigger

# Cron job
cat >/etc/cron.d/usb-dongle-check <<EOF
* * * * * root /usr/local/bin/usb-dongle-check.sh
EOF

# End
echo "✅ Setup completed!"
