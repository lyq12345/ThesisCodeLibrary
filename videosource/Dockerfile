FROM benchpilot/raspbian-picamera2:latest
COPY capture.py .
COPY capture_config.yml .
RUN pip install pyyaml
CMD [ "python3", "capture.py" ]