import cv2
import numpy as np

# Camera Intrinsics (Example values, should be calibrated for your camera)
K = np.array([[700, 0, 320],
              [0, 700, 240],
              [0, 0, 1]])

# Initialize ORB detector
orb = cv2.ORB_create()

# Use FLANN for feature matching
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

def process_video_input(video_path, output_video_path):
    # Open the video capture
    cap = cv2.VideoCapture(video_path)

    # Get the width, height, and frame rate of the video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Set up the video writer to save the output video
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Capture two frames for SLAM processing
    ret1, frame1 = cap.read()  # First frame
    ret2, frame2 = cap.read()  # Second frame

    if not ret1 or not ret2:
        print("Error reading video")
        return

    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Match descriptors using FLANN
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test to find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Draw the matches on the output frame
    matched_frame = cv2.drawMatches(frame1, kp1, frame2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Save the frame to the output video
    out.write(matched_frame)

    # Show the matched frame in a window (optional)
    cv2.imshow("Feature Matching", matched_frame)
    cv2.waitKey(0)  # Display window until a key is pressed

    # Extract keypoints from matches for pose estimation
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate the essential matrix using RANSAC
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # Recover pose (rotation and translation) from the essential matrix
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    print("Estimated Rotation Matrix (R):")
    print(R)
    print("\nEstimated Translation Vector (t):")
    print(t)

    # Release the video writer, video capture, and close the window
    cap.release()
    out.release()  # Don't forget to release the writer
    cv2.destroyAllWindows()

# Example Usage:
process_video_input(r"C:\Users\Ye Myat Moe\Documents\sp\computer_vision\quiz4_6511233\input\items_desk.mp4", 
                    r'C:\Users\Ye Myat Moe\Documents\sp\computer_vision\quiz4_6511233\output\output_video.mp4')