import mediapipe as mp
import cv2
import time
import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

data=pd.read_csv("dataset3.csv",index_col=0)

data['target'] = [1,0,1,1,1,1,0,1,1,1,0,0,0,0,0,0]
# data['target'] = [1,0]
X,Y = data.iloc[:,:132],data['target']
model = SVC(kernel = 'poly')
model.fit(X,Y)
model = SVC(kernel = 'poly')
model.fit(X,Y)

path1 = "/Users/karthikraja/Documents/Yoga/DATASET/TRAIN/plank/00000240.jpg"
temp1=[]
img5 = cv2.imread(path1)
img6 = cv2.cvtColor(img5, cv2.COLOR_BGR2RGB)
result = pose.process(img6)
if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark
        for j in landmarks:
                temp1 = temp1 + [j.x, j.y, j.z, j.visibility]
        y = model.predict([temp1])
        if y == 1:
            asan = "plank"
        else:
            asan = "Not plank"
        print(asan)
        # cv2.putText(img, asan, (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),3)
        # cv2.imshow("image",img)