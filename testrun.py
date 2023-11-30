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
path = "/Users/karthikraja/Documents/Yoga/DATASET/TEST/plank1" # enter dataset path
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


        img = cv2.imread(path + "/" + img3[index])

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

###################################################  Deploying the model ###################################################

 
# data = pd.read_csv("dataset3.csv")
# data['target'] = [1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,0,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0,0,1,1,0,1,1,1,1,0,1,0,0,1,0,1,1,0,0,0,1,1,1,0,1,1,0,1,0,1,0,0,1,1,1,1,1,0,0,0,1,0,1,1,0,0,1,0,0,1,1,1,0,1,0,0,1,1,0,0,1,0,0,1,0,1,1,1,1,0,0,0,0,0,1,0]
# X,Y = data.iloc[:,:127],data['target']
# print(data.shape)


data['target'] = [1,0,1,1,1,1,0,1,1,1,0,0,0,0,0,0]
X,Y = data.iloc[:,1:],data['target']
model = SVC(kernel = 'poly')
model.fit(X,Y)

path1 = "/Users/karthikraja/Documents/Yoga/DATASET/TEST/plank1/1.jpg"
temp1=[]
img5 = cv2.imread(path1)
img6 = cv2.cvtColor(img5, cv2.COLOR_BGR2RGB)
result = pose.process(img6)
print(result.pose_landmarks)
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
        cv2.putText(img, asan, (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),3)
        cv2.imshow("image",img)













########################################################## LIVE VIDEO COUNTER #######################################################################################






# import mediapipe as mp
# import cv2
# import time
# import numpy as np
# import pandas as pd
# import os
# from sklearn.svm import SVC
# import math
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles=mp.solutions.drawing_styles
# mp_pose = mp.solutions.pose
# count=0
# var=[]
# var1=[]
# coor=[]
# cap= cv2.VideoCapture(0)
# angle_B_degrees=0
# counter=0
# stage=None

# with mp_pose.Pose(
#     min_detection_confidence= 0.5,      
#     min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         success, image=cap.read()
#         if not success:
#             print("No video in camera frame")
#             break


#         image.flags.writeable = False
#         image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#         results=pose.process(image)

#         image.flags.writeable=True
#         image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

#         # Specify the landmarks for the right shoulder, right elbow, and right wrist
#         right_shoulder_landmark = mp_pose.PoseLandmark.RIGHT_SHOULDER
#         right_elbow_landmark = mp_pose.PoseLandmark.RIGHT_ELBOW
#         right_wrist_landmark = mp_pose.PoseLandmark.RIGHT_WRIST
#         nose=mp_pose.PoseLandmark.NOSE
#         right_ear=mp_pose.PoseLandmark.RIGHT_EAR
#         right_heel=mp_pose.PoseLandmark.RIGHT_HEEL

        
        
#         # Check if the landmarks are detected
#         # if results.pose_landmarks and right_shoulder_landmark and right_elbow_landmark:
#         if results.pose_landmarks :

#                # Extract the coordinates of the right shoulder and right elbow
#                 shoulder_x = results.pose_landmarks.landmark[right_shoulder_landmark].x
#                 shoulder_y = results.pose_landmarks.landmark[right_shoulder_landmark].y
#                 shoulder_z = results.pose_landmarks.landmark[right_shoulder_landmark].z

#                 elbow_x = results.pose_landmarks.landmark[right_elbow_landmark].x
#                 elbow_y = results.pose_landmarks.landmark[right_elbow_landmark].y
#                 elbow_z = results.pose_landmarks.landmark[right_elbow_landmark].z

#                 wrist_x = results.pose_landmarks.landmark[right_wrist_landmark].x
#                 wrist_y = results.pose_landmarks.landmark[right_wrist_landmark].y
#                 wrist_z = results.pose_landmarks.landmark[right_wrist_landmark].z

#                 nose_x=results.pose_landmarks.landmark[nose].x
#                 nose_y=results.pose_landmarks.landmark[nose].y
#                 nose_z=results.pose_landmarks.landmark[nose].z

#                 right_heel_x=results.pose_landmarks.landmark[right_heel].x
#                 right_heel_y=results.pose_landmarks.landmark[right_heel].y
#                 right_heel_z=results.pose_landmarks.landmark[right_heel].z

                              
                
               

#                 AB=(shoulder_x-elbow_x ,shoulder_y-elbow_y)
#                 BC=(elbow_x-wrist_x,elbow_y-wrist_y)

#                 # Calculate the dot product of AB and BC
#                 dot_product = AB[0] * BC[0] + AB[1] * BC[1] 

#                 # Calculate the magnitudes of AB and BC
#                 magnitude_AB = math.sqrt(AB[0] ** 2 + AB[1] ** 2 )
#                 magnitude_BC = math.sqrt(BC[0] ** 2 + BC[1] ** 2 )

#                 # Calculate the angle B using the dot product formula
#                 angle_B = math.acos(dot_product / (magnitude_AB * magnitude_BC))

#                 # Convert the angle from radians to degrees
#                 angle_B_degrees = math.degrees(angle_B)

#                 # Print the angle B
#                 # print("Angle B between AB and BC:", angle_B_degrees)
#                 var1.append(angle_B_degrees)
                


                # Calculate the angle between the shoulder and elbow
                # angle_radians = math.atan2(shoulder_y-elbow_y,shoulder_x-elbow_x)-math.atan2(wrist_y-elbow_y,wrist_x-elbow_x)
                # angle_degrees = math.degrees(angle_radians)
                # if angle_degrees > 180.0:
                #   angle_degrees = 360-angle_degrees
                # var.append(angle_degrees)
                
                
                # print(f"The angle between the shoulder and elbow is {angle_degrees} degrees for the {count} frame.")
        
#         count+=1
      
#         cv2.putText(image,str(angle_B_degrees), 
#                            (50,200), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 2, cv2.LINE_AA
#                                 )
#         mp_drawing.draw_landmarks(
#             image,
#             results.pose_landmarks,
#             mp_pose.POSE_CONNECTIONS,
#             landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#         )

#         #COUNTER

#         if angle_B_degrees < 30 :
#              stage="down"
#         elif angle_B_degrees >70 and stage=="down":
#              stage="up"
#              counter+=1
#              print(counter)
            

#         cv2.rectangle(image,(0,0), (255,73),(245,117,16),-1)

#         cv2.putText(image,'REPS',(15,12),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0),1,cv2.LINE_AA)
#         cv2.putText(image,str(counter),(10,60),cv2.FONT_HERSHEY_SIMPLEX,2, (255,255,255),2,cv2.LINE_AA)


#         cv2.imshow('MediaPipe pose estimation',image)

#         if cv2.waitKey(5) & 0xFF==27:
#             break
# #     print(count)


# # print(var1)   
# cap.release()
# plt.figure(1)
# plt.plot(var1)
# plt.title("1")

# plt.figure(2)
# plt.plot(var)
# plt.title("2")
# plt.show()





# ------------------------------------------------------------------ 3D GEO ------------------------------------------------------------------



# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the first point
# ax.scatter(elbow_x, elbow_y, elbow_z, c='r', marker='o', label='Point 1')

# # Plot the second point
# ax.scatter(wrist_x, wrist_y, wrist_z, c='b', marker='^', label='Point 2')

# # Set axis labels
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# # Add a legend
# ax.legend()

# # Show the plot
# plt.show()


# ------------------------------------------------------------------ 3D GEO ------------------------------------------------------------------










# ----------------------------------------------------------------------------------------------------------------------------------------------




# NOSE
# LEFT_EYE_INNER
# LEFT_EYE
# LEFT_EYE_OUTER
# RIGHT_EYE_INNER
# RIGHT_EYE
# RIGHT_EYE_OUTER
# LEFT_EAR
# RIGHT_EAR
# MOUTH_LEFT
# MOUTH_RIGHT
# LEFT_SHOULDER
# RIGHT_SHOULDER
# LEFT_ELBOW
# RIGHT_ELBOW
# LEFT_WRIST
# RIGHT_WRIST
# LEFT_PINKY
# RIGHT_PINKY
# LEFT_INDEX
# RIGHT_INDEX
# LEFT_THUMB
# RIGHT_THUMB
# LEFT_HIP
# RIGHT_HIP
# LEFT_KNEE
# RIGHT_KNEE
# LEFT_ANKLE
# RIGHT_ANKLE
# LEFT_HEEL
# RIGHT_HEEL
# LEFT_FOOT_INDEX