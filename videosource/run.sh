docker run -d -v /run/udev:/run/udev --privileged \
  -v /usr/bin/convert:/usr/bin/convert \
  -v /var/tmp:/var/tmp
  --device=/dev/vchiq:/dev/vchiq \
  benchpilot/raspbian-picamera2:latest