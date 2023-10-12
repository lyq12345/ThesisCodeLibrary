FROM python:3.8.17

COPY . .

RUN sh ./download_model.sh

RUN pip install --upgrade pip && pip install paho-mqtt cython pillow numpy opencv-python-headless torch --extra-index-url https://download.pytorch.org/whl/cu102 torchvision --extra-index-url https://download.pytorch.org/whl/cu102 pytest tqdm scipy matplotlib mock Flask && pip install --upgrade imageai

CMD ["python", "app.py"]
