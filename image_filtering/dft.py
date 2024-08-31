import numpy as np  
import cv2 as cv  
from matplotlib import pyplot as plt  

# Load image in grayscale  
img = cv.imread('test.jpg', cv.IMREAD_GRAYSCALE)  
assert img is not None, "File could not be read, check with os.path.exists()"  

# Perform DFT  
dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)  
dft_shift = np.fft.fftshift(dft)  

# Calculate magnitude spectrum  
magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))  

# Display and save first set of images  
plt.subplot(121), plt.imshow(img, cmap='gray')  
plt.title('Input Image'), plt.xticks([]), plt.yticks([])  
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')  
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])  
plt.savefig('magnitude_spectrum.png')  # Save magnitude spectrum image  
plt.show()  

# Create mask for high pass filter  
rows, cols = img.shape  
crow, ccol = rows // 2, cols // 2  

# Create a mask: center square is 1, remaining all zeros  
mask = np.zeros((rows, cols, 2), np.uint8)  
mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1  

# Apply mask and inverse DFT  
fshift = dft_shift * mask  
f_ishift = np.fft.ifftshift(fshift)  
img_back = cv.idft(f_ishift)  
img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])  

# Display and save the second set of images  
plt.subplot(121), plt.imshow(img, cmap='gray')  
plt.title('Input Image'), plt.xticks([]), plt.yticks([])  
plt.subplot(122), plt.imshow(img_back, cmap='gray')  
plt.title('Image after High Pass Filter'), plt.xticks([]), plt.yticks([])  
plt.savefig('high_pass_filtered.png')  # Save high pass filtered image  
plt.show()


Take your own color photo with the camera provided in the exam room. Convert the obtained color image to a grayscale image. Perform the following tasks with imported Python libraries using the grayscale image:
1. Design an asymmetric smoothing 3×3 filter mask using the last three digits of your personal student admission number(mine is 6511233). Obtain the result of the correlation between the grayscale image and the filter mask and save it as a BMP image.
2. Obtain the result of the correlation between the grayscale image and the filter mask and save it as a BMP image. Obtain the result of the convolution between the grayscale image and the aforesaid filter mask and save it as a BMP image. Compare the obtained images. 
3. Perform direct Fourier transform onto the grayscale image. Reduce by 75% the magnitude of the frequency component at the origin of the frequency spectrum.
4. Perform inverse Fourier transform onto the modified spectrum and compare the original and modified images.
