import cv2
import mediapipe as mp
import json
import numpy as np
import joblib
import pandas as pd


def is_slouching_hardcoded(left_shoulder, right_shoulder, left_eye, right_eye):
    # Extract necessary landmarks
    '''
    11 left sh
    12 right sh
    2 left eye
    5 right eye
    pose_landmarks
    A list of pose landmarks. Each landmark consists of the following:

    x and y: Landmark coordinates normalized to [0.0, 1.0] by the image width and height respectively.
    z: Represents the landmark depth with the depth at the midpoint of hips being the origin, and the smaller the value the closer the landmark is to the camera. 
    The magnitude of z uses roughly the same scale as x.
    visibility: A value in [0.0, 1.0] indicating the likelihood of the landmark being visible (present and not occluded) in the image.
    pose_world_landmarks
    '''
    
    # Calculate the average shoulder height
    avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    # Calculate the average hip height
    avg_eye_y = (left_eye.y + right_eye.y) / 2
    
    # Calculate the difference
    diff = avg_shoulder_y - avg_eye_y
    return diff < 0.5 


def collect_posture_data(reference_poses={'normal': False, 'incorrect': True, 'incorrect_hands': True}, duration=10):
    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Start video capture
    cap = cv2.VideoCapture(0)
    start_time = cv2.getTickCount()
    frame_rate = cv2.getTickFrequency()
    num_features = 14
    data = {pose_name: [] for pose_name in reference_poses.keys()}
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for pose_name in data:
            collecting_data = False
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                frame.flags.writeable = False
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                frame.flags.writeable = True

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    landmarks = results.pose_landmarks.landmark[:num_features+1] #up to elbow 14 
                    
                    if collecting_data:
                        cv2.putText(frame, f'Getting data for {pose_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        features = [
                            {'x': landmark.x, 'y': landmark.y, 'z': landmark.z, 'visibility': landmark.visibility}
                            for landmark in landmarks
                        ]
                        data[pose_name].append({'landmarks': features, 'slouch': reference_poses[pose_name]})
                        
                        elapsed_time = (cv2.getTickCount() - start_time) / frame_rate
                        if elapsed_time > duration:
                            collecting_data = False
                            break
                    else:
                        cv2.putText(frame, f'Press s to start collecting data for {pose_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.imshow('Posture Check', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    collecting_data = True
                    start_time = cv2.getTickCount()  # Reset start time when starting data collection
                elif key == ord('q') or cv2.getWindowProperty('Posture Check', cv2.WND_PROP_VISIBLE) < 1:
                    print(f"Skipping {pose_name} data collection")
                    collecting_data = False
                    break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Save collected data to a JSON file
    with open('data/posture_data.json', 'w') as f:
        json.dump(data, f, indent=4)

    print("Data collection complete.")

    

 

def check_posture_simple():
     # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # Initialize MediaPipe drawing utils
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    #add confidence thresholds https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md
    with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")  
                continue

            #should improve performance to temporarily disable writeability
            frame.flags.writeable = False
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process the image and detect the pose
            results = pose.process(rgb_frame)

            frame.flags.writeable = True
            # Draw the pose annotation on the image
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                landmarks = results.pose_landmarks.landmark
            
                if is_slouching_hardcoded(landmarks[11],landmarks[12],landmarks[2],landmarks[5]):
                    cv2.putText(frame, 'Slouching Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, 'Good Posture', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the frame
            cv2.imshow('Posture Check', frame)

            if cv2.waitKey(10) & 0xFF == ord('q') or cv2.getWindowProperty('Posture Check', cv2.WND_PROP_VISIBLE) < 1:
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    

def check_posture():
    #setup model
    model = joblib.load('data/model.pkl')
    num_features = 14

    #columns = [f'{feature}{i}' for i in range(num_features+1) for feature in ['x', 'y', 'z', 'visibility']]

    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # Initialize MediaPipe drawing utils
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    #add confidence thresholds https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md

    with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")  
                continue

            #should improve performance to temporarily disable writeability
            frame.flags.writeable = False
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process the image and detect the pose
            results = pose.process(rgb_frame)

            frame.flags.writeable = True
            # Draw the pose annotation on the image
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                #TODO make this a dataframe to get rid of warning
                landmarks = results.pose_landmarks.landmark[:num_features+1] #exclusive slice so +1
                flattened_landmarks = []
                for landmark in landmarks:
                    flattened_landmarks.extend([landmark.x,landmark.y,landmark.z,landmark.visibility])

                if model.predict([flattened_landmarks]):
                    cv2.putText(frame, 'Slouching Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, 'Good Posture', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the frame
            cv2.imshow('Posture Check', frame)

            if cv2.waitKey(10) & 0xFF == ord('q') or cv2.getWindowProperty('Posture Check', cv2.WND_PROP_VISIBLE) < 1:
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    #collect_posture_data()
    check_posture_simple()

