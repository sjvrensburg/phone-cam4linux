#!/bin/sh
# Installs the boot-time v4l2loopback configuration (needs root) so /dev/video10
# exists from boot and squigl-cli never has to prompt for a password.
set -eu
cd "$(dirname "$0")"
install -Dm644 modprobe.d/squigl.conf /etc/modprobe.d/squigl.conf
install -Dm644 modules-load.d/squigl.conf /etc/modules-load.d/squigl.conf
echo "Installed. Either reboot, or load it now with:"
echo "  modprobe -r v4l2loopback 2>/dev/null; modprobe v4l2loopback"
echo "(the -r fails harmlessly if the module is in use; the device appears after the next reboot)"
