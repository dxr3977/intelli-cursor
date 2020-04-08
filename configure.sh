#!/bin/bash

sudo apt update
sudo apt install python3.6
sudo apt install python3-pip
pip3 install pyautogui
pip3 install numpy
pip3 install scipy
pip3 install time
pip3 install opencv-python
pip3 install matplotlib
pip3 install pillow
pip3 install sklearn

sh install_dlib.sh

mkdir ~/intelli-cursor

mkdir ~/intelli-cursor/src
mkdir ~/intelli-cursor/src/data

cp src/lines.csv ~/intelli-cursor/src/data/
cp src/shape_predictor_68_face_landmarks.dat ~/intelli-cursor/src/data/
cp src/data_fast_images_face_50000.csv ~/intelli-cursor/src/data/

cp src/main.py ~/intelli-cursor/src/
cp src/cursor_training.py ~/intelli-cursor/src/
cp src/cursor_data_acquisitioner.py ~/intelli-cursor/src/

echo "alias intelli-cursor=\"python3 ~/intelli-cursor/src/main.py\"" >> ~/.bashrc
source ~/.bash_profile