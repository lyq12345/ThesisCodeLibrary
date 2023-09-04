FROM python:3.8.17

RUN mkdir models && wget github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/yolov3.pt/ -O ./models/yolov3.pt && pip install --upgrade pip && pip install paho-mqtt cython pillow numpy opencv-python-headless torch --extra-index-url https://download.pytorch.org/whl/cu102 torchvision --extra-index-url https://download.pytorch.org/whl/cu102 pytest tqdm scipy matplotlib mock Flask && pip install --upgrade imageai

COPY img_process.py .
COPY app.py .

CMD ["python", "app.py"]
