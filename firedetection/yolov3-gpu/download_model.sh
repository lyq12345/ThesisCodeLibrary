#!/bin/bash

DIRECTORY="models"

# check if "models" dir exists
if [ ! -d "$DIRECTORY" ]; then
    mkdir "$DIRECTORY"
fi

# 进入 "models" 目录
cd "$DIRECTORY"

# 下载文件
wget -P . https://github.com/lyq12345/ThesisCodeLibrary/releases/download/v1.0/yolov3_fire-dataset_last.pt
wget -P . https://github.com/lyq12345/ThesisCodeLibrary/releases/download/v1.0/fire-dataset_yolov3_detection_config.json
