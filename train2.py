import mediapipe as mp
import cv2
import time
import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils # For drawing keypoints
points = mpPose.PoseLandmark # Landmarks
path = "/Users/karthikraja/Documents/Yoga/DATASET/TRAIN/plank" # enter dataset path
data = []
for p in points:
        x = str(p)[13:]
        data.append(x + "_x")
        data.append(x + "_y")   
        data.append(x + "_z")
        data.append(x + "_vis")
data = pd.DataFrame(columns = data) # Empty dataset
count = 0
img3=[]
for img2 in os.listdir(path):
        img3.append(img2)
# print(img3)
img3 = sorted(img3, key=lambda x: int(x.split('.')[0]))

index=0

for i in range(len(img3)):
        temp = []
        # print("img2",i)
        # print("img3",img3)
        # print(img3[index])
        # print("index",index)


        img = cv2.imread(path + '/' + img3[index])

        imageWidth, imageHeight = img.shape[:2]

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        blackie = np.zeros(img.shape) # Blank image

        results = pose.process(imgRGB)
        
        # print("index1",index)

        if results.pose_landmarks:

                # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS) #draw landmarks on image
                # print("index",index)
                # print(img3[index])
                mpDraw.draw_landmarks(blackie, results.pose_landmarks, mpPose.POSE_CONNECTIONS) # draw landmarks on blackie

                landmarks = results.pose_landmarks.landmark

                for i,j in zip(points,landmarks):

                        temp = temp + [j.x, j.y, j.z, j.visibility]

                data.loc[count] = temp
                count +=1
        index+=1


        # cv2.imshow("Image", img)

        # cv2.imshow("blackie",blackie)

        # cv2.waitKey(100)

data.to_csv("dataset3.csv") # save the data as a csv file
