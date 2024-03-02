FROM python:3.8.17

COPY . .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip && pip install numpy opencv-python-headless

CMD ["python", "app.py"]
