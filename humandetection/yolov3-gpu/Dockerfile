FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

COPY . .

RUN sh ./download_model.sh

RUN apt-get -y update && apt-get -y install python3-tk

RUN pip3 install --upgrade pip && pip3 install paho-mqtt cython pillow numpy pytest tqdm scipy matplotlib mock flask imageai==3.0.3

CMD ["python3", "app.py"]