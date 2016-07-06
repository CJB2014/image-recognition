# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 14:57:37 2016

@author: agricadmin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 08:05:42 2016

@author: agricadmin
"""
import cv2
import os 
import numpy as np
import pandas as pd
import sklearn as sk
from cv2 import CascadeClassifier

os.chdir('C://Users//agricadmin//Documents//deep learning')
sift=cv2.xfeatures2d.SIFT_create() ## extractor 
surf=cv2.xfeatures2d.SURF_create()



''' load the image extraction of keypoint and descriptor ''' 
store2=cv2.imread("top cap.jpg")
kp_store2=sift.detect(store2)
#kp_store2=surf.detect(store2)
store2_kp=cv2.drawKeypoints(store2,kp_store2,cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("image", store2_kp)
cv2.waitKey(0)
kp_store2, des_store2=sift.detectAndCompute(store2_kp,None)
#kp_store2, des_store2=surf.detectAndCompute(store2_kp,None)


''' load image and extraction of keypoint and descriptor foro second picture''' 
store3=cv2.imread("cap.jpg")
kp_store3=sift.detect(store3)
#kp_store3=surf.detect(store3)
store3_kp=cv2.drawKeypoints(store3,kp_store3,cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("image", store3_kp)
cv2.waitKey(0)
kp_store3, des_store3=sift.detectAndCompute(store3_kp,None)
#kp_store3, des_store3=surf.detectAndCompute(store3_kp,None)



''' brute force matcher ''' 
bf=cv2.BFMatcher(cv2.NORM_L2)

matches=bf.knnMatch(des_store2,des_store2,k=2, compactResult=True)

good=[]
for m, n in matches:
    if m.distance<0.75*n.distance:
        good.append([m])

img3=cv2.drawMatchesKnn(store2,kp_store2,store3,kp_store3,matches[:20],cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
from matplotlib import pyplot as plt
plt.imshow(img3)
N = 2
params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )
,plt.show()



