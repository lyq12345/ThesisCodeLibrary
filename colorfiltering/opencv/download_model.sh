#!/bin/bash

DIRECTORY="models"

# check if "models" dir exists
if [ ! -d "$DIRECTORY" ]; then
    mkdir "$DIRECTORY"
fi

#  go to "models" dir
cd "$DIRECTORY"

# download retrained model
wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
