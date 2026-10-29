#!/bin/sh
# USB camera diagnosis on the ZCU104.
#   ./camcheck.sh      test /dev/video0
#   ./camcheck.sh 2    test /dev/video2
# Good result: a full row of '<' in the stream test and no new -71 errors.
DEV=${1:-0}

echo "--- USB devices ---"
ls /sys/bus/usb/devices/ | grep -v ":"
dmesg | grep -iE "see3cam|uvc" | tail -3

echo "--- video devices ---"
for d in /sys/class/video4linux/video*; do
    [ -e "$d" ] && echo "/dev/$(basename "$d"): $(cat "$d/name")"
done

echo "--- stream test /dev/video$DEV (10 s) ---"
timeout 10 v4l2-ctl -d "/dev/video$DEV" \
    --set-fmt-video=width=640,height=480,pixelformat=UYVY \
    --stream-mmap --stream-count=30
echo
echo "USB -71 errors so far: $(dmesg | grep -c -- '-71')"
