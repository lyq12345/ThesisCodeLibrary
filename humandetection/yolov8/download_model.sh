#!/bin/bash

DIRECTORY="models"

# check if "models" dir exists
if [ ! -d "$DIRECTORY" ]; then
    mkdir "$DIRECTORY"
fi

#  go to "models" dir
cd "$DIRECTORY"

# download retrained model
wget -P . https://github.com/lyq12345/ThesisCodeLibrary/releases/download/v1.0/yolov3.pt
