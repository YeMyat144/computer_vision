import cv2
import numpy as np

# Initialize the ORB detector algorithm
orb = cv2.ORB_create()

# Path to your video file
video_path = "test_countryroad.mp4"

# Start video capture (replace with video file path)
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print(f"Error: Unable to open video file {video_path}")
    exit()

# Create a BFMatcher object for matching features
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Store previous frame features (initialization)
previous_kp = None
previous_des = None

while cap.isOpened():
    # Read a new frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale (ORB works on grayscale images)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints and descriptors in the current frame
    keypoints, descriptors = orb.detectAndCompute(gray_frame, None)

    if previous_kp is not None and previous_des is not None:
        # Match descriptors between previous and current frame using BFMatcher
        matches = bf.match(previous_des, descriptors)

        # Sort matches by distance (lower distance is better)
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw the matches on the frame
        matched_frame = cv2.drawMatches(previous_frame, previous_kp, frame, keypoints, matches[:50], None, flags=2)

        # Show the matched frame
        cv2.imshow('Monocular SLAM - ORB Feature Matching', matched_frame)

    # Store current frame keypoints and descriptors for the next iteration
    previous_kp = keypoints
    previous_des = descriptors
    previous_frame = frame

    # Press 'q' to quit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
