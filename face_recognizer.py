#!/usr/bin/python




import cv2
import os 
import numpy as np
import pandas as pd
import sklearn as sk
from cv2 import CascadeClassifier


os.chdir('C://Users//agricadmin//Documents//deep learning')
path='C://Users//agricadmin//Documents//deep learning'


''' Load the pre trained cascade from openCV: first for face / second for eye'''  
face_cascade=cv2.CascadeClassifier('C:\\Users\\agricadmin\\Anaconda2\\pkgs\\opencv3-3.1.0-py27_0\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
fc=face_cascade.load('C:\\Users\\agricadmin\\Anaconda2\\pkgs\\opencv3-3.1.0-py27_0\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
print(fc)
eye_cascade=cv2.CascadeClassifier('C:\\Users\\agricadmin\\Anaconda2\\pkgs\\opencv3-3.1.0-py27_0\\Library\\etc\\haarcascades\\haarcascade_eye.xml')
ec=eye_cascade.load('C:\\Users\\agricadmin\\Anaconda2\\pkgs\\opencv3-3.1.0-py27_0\\Library\\etc\\haarcascades\\haarcascade_eye.xml')
print(ec)

''' Load image and convert to grayscale '''
img=cv2.imread('spotlight.jpg')
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


''' draw rectangle around the area where the faces are'''
faces=face_cascade.detectMultiScale(gray,1.5,4)
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y),(x+w,y+h), (255,0,0),2)
    roi_gray= gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


''' show the final results '''    
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


