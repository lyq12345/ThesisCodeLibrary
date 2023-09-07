FROM benchpilot/raspbian-picamera2:latest
COPY app2.py .
RUN apt update && apt-get -y upgrade
RUN apt-get install -y imagemagick python3-opencv opencv-data
RUN chmod a+rwx /usr/bin/convert
RUN pip install pyyaml requests redis
CMD [ "python3", "app2.py" ]