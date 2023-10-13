FROM python:3.8.17

COPY . .

RUN pip install --upgrade pip && pip install numpy opencv-python-headless Flask scikit-image imutils

CMD ["python", "app.py"]
