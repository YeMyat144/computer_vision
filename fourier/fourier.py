import cv2 as cv  
import numpy as np  
from matplotlib import pyplot as plt  

# Load image in grayscale  
img = cv.imread('messi5.jpg', cv.IMREAD_GRAYSCALE)  
assert img is not None, "file could not be read, check with os.path.exists()"  
f = np.fft.fft2(img)  
fshift = np.fft.fftshift(f)  
magnitude_spectrum = 20 * np.log(np.abs(fshift))  

# Compute DFT with OpenCV  
dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)  
dft_shift = np.fft.fftshift(dft)  
magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))  

# Display and save first set of images  
plt.subplot(121), plt.imshow(img, cmap='gray')  
plt.title('Input Image'), plt.xticks([]), plt.yticks([])  
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')  
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])  
plt.savefig('magnitude_spectrum.png')  # Save magnitude spectrum image  
plt.show()  

# Apply high-pass filter  
rows, cols = img.shape  
crow, ccol = rows // 2, cols // 2  
fshift[crow - 30:crow + 31, ccol - 30:ccol + 31] = 0  
f_ishift = np.fft.ifftshift(fshift)  
img_back = np.fft.ifft2(f_ishift)  
img_back = np.real(img_back)  

# Display and save second set of images  
plt.subplot(131), plt.imshow(img, cmap='gray')  
plt.title('Input Image'), plt.xticks([]), plt.yticks([])  
plt.subplot(132), plt.imshow(img_back, cmap='gray')  
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])  
plt.subplot(133), plt.imshow(img_back, cmap='jet')  
plt.title('Result in JET'), plt.xticks([]), plt.yticks([])  
plt.savefig('high_pass_filtered.png')  # Save high-pass filtered image  
plt.show()  

# Create a mask for low-pass filter  
rows, cols = img.shape  
crow, ccol = rows // 2, cols // 2  

mask = np.zeros((rows, cols, 2), np.uint8)  
mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1  

# Apply mask and inverse DFT  
fshift = dft_shift * mask  
f_ishift = np.fft.ifftshift(fshift)  
img_back = cv.idft(f_ishift)  
img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])  

# Display and save third set of images  
plt.subplot(121), plt.imshow(img, cmap='gray')  
plt.title('Input Image'), plt.xticks([]), plt.yticks([])  
plt.subplot(122), plt.imshow(img_back, cmap='gray')  
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])  
plt.savefig('low_pass_filtered.png')  # Save low-pass filtered image  
plt.show()