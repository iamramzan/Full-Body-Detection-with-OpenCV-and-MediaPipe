# P7: Full Body Detection with OpenCV and MediaPipe

#import libraries
import mediapipe as mp
import cv2

# Initialize Mediapipe drawing utilities and holistic model components
mp_drawing = mp.solutions.drawing_utils  # Utility for drawing landmarks
mp_holistic = mp.solutions.holistic  # Mediapipe holistic model (pose, face, hand detection)

# Define pose connections for visualization
mp_holistic.POSE_CONNECTIONS

# Customize drawing specifications (e.g., color, thickness, radius)
mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)

# Start capturing video from the webcam
cap = cv2.VideoCapture(1)

# Initialize the holistic model with minimum confidence thresholds
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Loop to process each frame from the webcam
    while cap.isOpened():
        ret, frame = cap.read()  # Read the current frame from the webcam
        
        # Recolor the frame from BGR (OpenCV format) to RGB (Mediapipe format)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the RGB image using the holistic model to detect landmarks
        results = holistic.process(image)
        # `results` now contains face, pose, and hand landmarks
        
        # Convert the image back to BGR format for OpenCV rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(
            image, 
            results.face_landmarks, 
            mp_holistic.FACEMESH_CONTOURS,  # Draw the facial mesh contours
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),  # Landmarks color/style
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)  # Connections color/style
        )
        
        # 2. Draw right hand landmarks
        mp_drawing.draw_landmarks(
            image, 
            results.right_hand_landmarks, 
            mp_holistic.HAND_CONNECTIONS,  # Draw the hand connections
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),  # Landmarks color/style
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)  # Connections color/style
        )

        # 3. Draw left hand landmarks
        mp_drawing.draw_landmarks(
            image, 
            results.left_hand_landmarks, 
            mp_holistic.HAND_CONNECTIONS,  # Draw the hand connections
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),  # Landmarks color/style
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)  # Connections color/style
        )

        # 4. Draw pose landmarks
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_holistic.POSE_CONNECTIONS,  # Draw the body pose connections
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),  # Landmarks color/style
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)  # Connections color/style
        )
                        
        # Display the processed frame in a window
        cv2.imshow('Raw Webcam Feed', image)

        # Stop the loop when 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
