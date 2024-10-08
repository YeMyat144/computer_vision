import cv2
from display import Display
from extractor import Frame, denormalize, IRt, match_frames, add_ones
import numpy as np
from pointmap import Map, Point

def triangulate(pose1, pose2, pts1, pts2):
    # Initialize the result array to store the homogeneous coordinates of the 3D points
    ret = np.zeros((pts1.shape[0], 4))
 
    # Invert the camera poses to get the projection matrices
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
 
    # Loop through each pair of corresponding points
    for i, p in enumerate(zip(add_ones(pts1), add_ones(pts2))):
        # Initialize the matrix A to hold the linear equations
        A = np.zeros((4, 4))
 
        # Populate the matrix A with the equations derived from the projection matrices and the points
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
 
        # Perform SVD on A
        _, _, vt = np.linalg.svd(A)
 
        # The solution is the last row of V transposed (V^T), corresponding to the smallest singular value
        ret[i] = vt[3]
 
    # Return the 3D points in homogeneous coordinates
    return ret

### Camera intrinsics
# define principal point offset or optical center coordinates
W, H = 1920//2, 1080//2
 
# define focus length
F = 270
 
# define intrinsic matrix and inverse of that
K = np.array(([F, 0, W//2], [0,F,H//2], [0, 0, 1]))
Kinv = np.linalg.inv(K)
 
# image display initialization
display = Display(W, H)
 
# initialize a map
mapp = Map()
mapp.create_viewer()

def process_frame(img):
    img = cv2.resize(img, (W, H))
    frame = Frame(mapp, img, K)
    if frame.id == 0:
        return
 
    # previous frame f2 to the current frame f1.
    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]
 
    idx1, idx2, Rt = match_frames(f1, f2)
     
    # X_f1 = E * X_f2, f2 is in world coordinate frame, multiplying that with
    # Rt transforms the f2 pose wrt the f1 coordinate frame
    f1.pose = np.dot(Rt, f2.pose)
 
 
    # The output is a matrix where each row is a 3D point in homogeneous coordinates [𝑋, 𝑌, 𝑍, 𝑊]
    # returns an array of size (n, 4), n = feature points
    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
 
 
    # The homogeneous coordinates [𝑋, 𝑌, 𝑍, 𝑊] are converted to Euclidean coordinates
    pts4d /= pts4d[:, 3:]
 
 
    # Reject points without enough "Parallax" and points behind the camera
    # returns, A boolean array indicating which points satisfy both criteria.
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)
 
    for i, p in enumerate(pts4d):
        # If the point is not good (i.e., good_pts4d[i] is False), 
        # the loop skips the current iteration and moves to the next point.
        if not good_pts4d[i]:
            continue
        pt = Point(mapp, p)
        pt.add_observation(f1, i)
        pt.add_observation(f2, i)
 
    for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)
 
        cv2.circle(img, (u1,v1), 3, (0,255,0))
        cv2.line(img, (u1,v1), (u2, v2), (255,0,0))
 
    # 2-D display
    display.paint(img)
 
    # 3-D display
    mapp.display()
 
if __name__== "__main__":
    cap = cv2.VideoCapture("test_countryroad.mp4")
 
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break