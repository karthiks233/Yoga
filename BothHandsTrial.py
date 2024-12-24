import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

straight_tolerance = 0.045  # Adjust as necessary
ground_tolerance = 0.08    # Adjust as necessary

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    while cap.isOpened():
        success, image = cap.read()

        # Convert the image to grayscale to compute brightness
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate the average brightness
        brightness = np.mean(gray_image)
        
        # Define a threshold for low brightness
        brightness_threshold = 50  # Adjust as needed (0-255 scale)

        if not success:
            print("No video in camera frame")
            break

        # Check if brightness is below threshold
        if brightness < brightness_threshold:
            bright_msg = "Brightness too low"
            cv2.putText(image, bright_msg, (50, 300), 
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
            print("Brightness too low")
        
        # Proceed with pose detection if brightness is sufficient
        else:
            # Prepare the image for pose processing
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Revert image changes for display
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                visibility_threshold = 0.5
                
                # Define landmarks to check for right and left hands
                right_landmarks = [
                    (mp_pose.PoseLandmark.RIGHT_SHOULDER, 'shoulder'),
                    (mp_pose.PoseLandmark.RIGHT_ELBOW, 'elbow'),
                    (mp_pose.PoseLandmark.RIGHT_WRIST, 'wrist')
                ]
                left_landmarks = [
                    (mp_pose.PoseLandmark.LEFT_SHOULDER, 'shoulder'),
                    (mp_pose.PoseLandmark.LEFT_ELBOW, 'elbow'),
                    (mp_pose.PoseLandmark.LEFT_WRIST, 'wrist')
                ]

                for side, landmarks in zip(['Right', 'Left'], [right_landmarks, left_landmarks]):
                    valid_landmarks = {}
                    for idx, name in landmarks:
                        landmark = results.pose_landmarks.landmark[idx]
                        if landmark.visibility > visibility_threshold:
                            valid_landmarks[name] = (landmark.x, landmark.y, landmark.z)

                    if 'shoulder' in valid_landmarks and 'elbow' in valid_landmarks and 'wrist' in valid_landmarks:
                        shoulder = np.array(valid_landmarks['shoulder'])
                        elbow = np.array(valid_landmarks['elbow'])
                        wrist = np.array(valid_landmarks['wrist'])

                        # Calculate frame dimensions
                        frame_height, frame_width, _ = image.shape

                        # Calculate x,z,y-position and guiding lines
                        shoulder_x = int(valid_landmarks['shoulder'][0] * frame_height)
                        shoulder_z = int(valid_landmarks['shoulder'][2] * frame_height)

                        elbow_x = int(valid_landmarks['elbow'][0] * frame_height)
                        elbow_y = int(valid_landmarks['elbow'][1] * frame_height)
                        elbow_z = int(valid_landmarks['elbow'][2] * frame_height)

                        wrist_x = int(valid_landmarks['wrist'][0] * frame_height)
                        wrist_y = int(valid_landmarks['wrist'][1] * frame_height)
                        wrist_z = int(valid_landmarks['wrist'][2] * frame_height)
                        
                        # Calculate shoulder y-position and guiding lines
                        shoulder_y = int(valid_landmarks['shoulder'][1] * frame_height)
                        top_line_y = int(shoulder_y - 0.1 * frame_height)  # 10% above shoulder
                        bottom_line_y = int(shoulder_y + 0.1 * frame_height)  # 10% below shoulder

                        # Ensure lines stay within frame boundaries
                        top_line_y = max(0, top_line_y)
                        bottom_line_y = min(frame_height - 1, bottom_line_y)

                        # Draw the guiding lines
                        cv2.line(image, (0, top_line_y), (frame_width, top_line_y), (0, 0, 0), 2)  # Top guiding line
                        cv2.line(image, (0, bottom_line_y), (frame_width, bottom_line_y), (0, 0, 0), 2)  # Bottom guiding line

                        # Get y-coordinates of landmarks
                        elbow_y = int(valid_landmarks['elbow'][1] * frame_height)
                        wrist_y = int(valid_landmarks['wrist'][1] * frame_height)

                        # Calculate z-coordinates
                        shoulder_z = int(valid_landmarks['shoulder'][2] * frame_height)
                        elbow_z = int(valid_landmarks['elbow'][2] * frame_height)
                        wrist_z = int(valid_landmarks['wrist'][2] * frame_height)

                        # Threshold for "too far"
                        z_threshold = 0


                        # Check if all landmarks are within guiding lines
                        if not (top_line_y <= shoulder_y <= bottom_line_y and
                                top_line_y <= elbow_y <= bottom_line_y and
                                top_line_y <= wrist_y <= bottom_line_y ):
                            # Display warning message if points are outside the guiding lines
                            warning_msg = f"{side} arm landmarks outside guiding lines"
                            cv2.putText(image, warning_msg, (10, 150 if side == 'Right' else 200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                        if shoulder_z > z_threshold or elbow_z > z_threshold or wrist_z > z_threshold:
                           
                            too_far_msg = f"{side} arm is too far"
                            cv2.putText(image, too_far_msg, (50, 400 if side == 'Right' else 450),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                    
                
                        else:
                            # Calculate vectors and check straightness
                            SE = elbow - shoulder
                            EW = wrist - elbow

                            cross_product = np.cross(SE, EW)
                            is_straight = np.linalg.norm(cross_product) < straight_tolerance

                            # Check parallelism with the ground
                            is_parallel_to_ground = abs(SE[1]) < ground_tolerance and abs(EW[1]) < ground_tolerance

                            # Determine the message
                            if is_straight and is_parallel_to_ground:
                                msg = f"{side} hand is STRAIGHT AND PARALLEL"
                            elif is_straight:
                                msg = f"{side} hand is STRAIGHT BUT NOT PARALLEL"
                            else:
                                msg = f"{side} hand is NOT STRAIGHT"

                            # Display message
                            cv2.putText(image, msg, (50, 50 if side == 'Right' else 100), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255,0), 3)

                            # Draw landmarks and connections
                            mp_drawing.draw_landmarks(
                                image,
                                results.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                            )
        
        # Display the frame
        cv2.imshow('MediaPipe Pose Estimation', image)

        # Break on 'Esc' key
        if cv2.waitKey(5) & 0xFF == 27:
            break           

cap.release()
cv2.destroyAllWindows()
