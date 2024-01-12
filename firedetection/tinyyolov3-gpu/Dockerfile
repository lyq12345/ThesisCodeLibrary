FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

COPY . .

RUN sh ./download_model.sh

RUN apt-key adv --fetch-keys https://apt.kitware.com/keys/kitware-archive-latest.asc

RUN apt -y update && apt -y install libopencv-dev && apt -y install python3-opencv python3-tk

RUN pip3 install --upgrade pip && pip3 install paho-mqtt cython pillow numpy pytest tqdm scipy matplotlib mock flask imageai==3.0.3

CMD ["python3", "app.py"]