import cv2
import mediapipe as mp

# Initialize MediaPipe Pose model
#add confidence thresholds https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize MediaPipe drawing utils
mp_drawing = mp.solutions.drawing_utils

def is_slouching(landmarks):
    # Extract necessary landmarks
    '''
    pose_landmarks
    A list of pose landmarks. Each landmark consists of the following:

    x and y: Landmark coordinates normalized to [0.0, 1.0] by the image width and height respectively.
    z: Represents the landmark depth with the depth at the midpoint of hips being the origin, and the smaller the value the closer the landmark is to the camera. The magnitude of z uses roughly the same scale as x.
    visibility: A value in [0.0, 1.0] indicating the likelihood of the landmark being visible (present and not occluded) in the image.
    pose_world_landmarks
    '''
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
    
    # Calculate the average shoulder height
    avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    # Calculate the average hip height
    avg_eye_y = (left_eye.y + right_eye.y) / 2
    
    # Calculate the difference
    diff = avg_shoulder_y - avg_eye_y
    
    # Determine slouching based on the height difference threshold
    #TODO make an adaptable version prefereably based on learned info 
    return diff < 0.5  


# Start video capture
cap = cv2.VideoCapture(0)
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
            
            # Check if the user is slouching
            if is_slouching(results.pose_landmarks.landmark):
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
pose.close()
