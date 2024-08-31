import numpy as np
import cv2 as cv

# Open the video file
cap = cv.VideoCapture(cv.samples.findFile("car.mp4"))

# Get the width and height of the frames
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 files
output_path = 'dense_output.mp4'
out = cv.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

ret, frame1 = cap.read()
if not ret:
    print('Error: No frames grabbed!')
    cap.release()
    out.release()
    cv.destroyAllWindows()
    exit()

prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while True:
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    
    # Display the result
    cv.imshow('frame2', bgr)
    
    # Write the frame to the output video file
    out.write(bgr)
    
    k = cv.waitKey(30) & 0xff
    if k == 27:  # Esc key to exit
        break
    elif k == ord('s'):  # Save current frame
        cv.imwrite('opticalfb.png', frame2)
        cv.imwrite('opticalhsv.png', bgr)

    prvs = next

# Release resources
cap.release()
out.release()
cv.destroyAllWindows()
