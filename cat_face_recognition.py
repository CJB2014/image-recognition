# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:20:33 2016

@author: agricadmin
"""

import cv2
import os
import numpy as np 
import sklearn as sk 
from cv2 import CascadeClassifier 

os.chdir('C://Users//agricadmin//Documents//deep learning')


''' cat fae recognition ''' 
face_cascade=cv2.CascadeClassifier('C:\\Users\\agricadmin\\Anaconda2\\pkgs\\opencv3-3.1.0-py27_0\\Library\\etc\\haarcascades\\haarcascade_frontalcatface.xml')
fc=face_cascade.load('C:\\Users\\agricadmin\\Anaconda2\\pkgs\\opencv3-3.1.0-py27_0\\Library\\etc\\haarcascades\\haarcascade_frontalcatface.xml')
print(fc)


''' Image of a cat ''' 
img=cv2.imread('Bengal_50.jpg')
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(gray,1.3,5)
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y),(x+w,y+h), (255,0,0),2)
    roi_gray= gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
   
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



