FROM python:3.8.17

COPY . .

RUN sh ./download_model.sh

RUN pip install --upgrade pip && pip install cython pillow numpy opencv-python-headless torch --extra-index-url https://download.pytorch.org/whl/cu102 torchvision --extra-index-url https://download.pytorch.org/whl/cu102 pytest tqdm scipy matplotlib mock Flask && pip install imageai==3.0.3

CMD ["python", "app.py"]
