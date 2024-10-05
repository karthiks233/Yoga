import mediapipe as mp
import cv2
import time
import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
count=0
var=[]
var1=[]
# coor=[]
cap= cv2.VideoCapture(0)
angle_B_degrees=0
counter=0
stage=None

with mp_pose.Pose(
    min_detection_confidence= 0.5,      
    min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image=cap.read()
        if not success:
            print("No video in camera frame")
            break


        image.flags.writeable = False
        image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        results=pose.process(image)

        image.flags.writeable=True
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        # Specify the landmarks for the right shoulder, right elbow, and right wrist
        right_shoulder_landmark = mp_pose.PoseLandmark.RIGHT_SHOULDER
        right_elbow_landmark = mp_pose.PoseLandmark.RIGHT_ELBOW
        right_wrist_landmark = mp_pose.PoseLandmark.RIGHT_WRIST
        nose=mp_pose.PoseLandmark.NOSE
        right_ear=mp_pose.PoseLandmark.RIGHT_EAR
        right_heel=mp_pose.PoseLandmark.RIGHT_HEEL
        right_hip=mp_pose.PoseLandmark.RIGHT_HIP
        right_knee=mp_pose.PoseLandmark.RIGHT_KNEE
        right_ankle=mp_pose.PoseLandmark.RIGHT_ANKLE
        
        
        
        # Check if the landmarks are detected
        # if results.pose_landmarks and right_shoulder_landmark and right_elbow_landmark:
        if results.pose_landmarks :

                visibility_threshold = 0.5

                landmarks = [
                    (right_shoulder_landmark, 'shoulder'),
                    (right_elbow_landmark, 'elbow'),
                    (right_wrist_landmark, 'wrist'),
                    (nose, 'nose'),
                    (right_hip, 'hip'),
                    (right_knee, 'knee'),
                    (right_ankle, 'ankle'),
                ]

        keypoints = {}

        for idx, name in landmarks:
            landmark = results.pose_landmarks.landmark[idx]
            if landmark.visibility > visibility_threshold:
               # Extract the coordinates of the right shoulder and right elbow
                shoulder_x = results.pose_landmarks.landmark[right_shoulder_landmark].x
                shoulder_y = results.pose_landmarks.landmark[right_shoulder_landmark].y
                shoulder_z = results.pose_landmarks.landmark[right_shoulder_landmark].z

                elbow_x = results.pose_landmarks.landmark[right_elbow_landmark].x
                elbow_y = results.pose_landmarks.landmark[right_elbow_landmark].y
                elbow_z = results.pose_landmarks.landmark[right_elbow_landmark].z

                wrist_x = results.pose_landmarks.landmark[right_wrist_landmark].x
                wrist_y = results.pose_landmarks.landmark[right_wrist_landmark].y
                wrist_z = results.pose_landmarks.landmark[right_wrist_landmark].z

                nose_x=results.pose_landmarks.landmark[nose].x
                nose_y=results.pose_landmarks.landmark[nose].y
                nose_z=results.pose_landmarks.landmark[nose].z

                right_hip_x=results.pose_landmarks.landmark[right_hip].x
                right_hip_y=results.pose_landmarks.landmark[right_hip].y
                right_hip_z=results.pose_landmarks.landmark[right_hip].z

                hip = np.array([right_hip_x, right_hip_y, right_hip_z])


                right_knee_x=results.pose_landmarks.landmark[right_knee].x
                right_knee_y=results.pose_landmarks.landmark[right_knee].y
                right_knee_z=results.pose_landmarks.landmark[right_knee].z

                knee = np.array([right_knee_x, right_knee_y, right_knee_z])

                right_ankle_x=results.pose_landmarks.landmark[right_ankle].x
                right_ankle_y=results.pose_landmarks.landmark[right_ankle].y
                right_ankle_z=results.pose_landmarks.landmark[right_ankle].z

                ankle = np.array([right_ankle_x, right_ankle_y, right_ankle_z])

                # ------------------------------------------------------------------ HANDS PARALLEL TO GROUND ------------------------------------------------------------------


            
                shoulder = np.array([shoulder_x,shoulder_y,shoulder_z])
                elbow = np.array([elbow_x,elbow_y,elbow_z])
                wrist = np.array([wrist_x,shoulder_y,shoulder_z])

                SE = elbow - shoulder  
                EW = wrist - elbow
                

                # Compute the cross product
                cross_product = np.cross(SE, EW)

                # Check if the cross product is [0, 0, 0]
                if np.all(cross_product == 0):
                    print("The hand is straight.")
                else:
                    print("The hand is not straight.")
                
                # Compute the dot product
                dot_product = np.dot(SE, EW)

                # Compute the magnitudes of the vectors
                magnitude_SE = np.linalg.norm(SE)
                magnitude_EW = np.linalg.norm(EW)

                # Calculate the cosine of the angle
                cos_theta = dot_product / (magnitude_SE * magnitude_EW)

                # Calculate the angle in radians
                angle_radians = np.arccos(cos_theta)

                # Convert the angle to degrees
                angle_degrees = np.degrees(angle_radians)
                print("angle_degrees",angle_degrees)

                msg = str(angle_degrees)
                cv2.putText(image,msg, 
                                (101,400), 
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 2, cv2.LINE_AA
                                        )
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

            
                
                cv2.imshow('MediaPipe pose estimation',image)

                if cv2.waitKey(5) & 0xFF==27:
                    break
                
            else:

                msg = 'Points not found'
                cv2.putText(image,msg, 
                                (101,400), 
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 2, cv2.LINE_AA
                                        )
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

            
                
        cv2.imshow('MediaPipe pose estimation',image)

        if cv2.waitKey(5) & 0xFF==27:
            break
                
           
cap.release()









# ----------------------------------------------------------------------------------------------------------------------------------------------
