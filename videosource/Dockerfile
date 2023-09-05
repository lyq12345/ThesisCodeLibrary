FROM benchpilot/raspbian-picamera2:latest
COPY capture.py .
COPY capture_config.yml .
RUN apt update && apt-get -y upgrade
RUN apt-get install -y imagemagick
RUN chmod a+rwx /usr/bin/convert
RUN pip install pyyaml
CMD [ "python3", "capture.py" ]