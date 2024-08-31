import numpy as np
import cv2
import matplotlib.pyplot as plt

color_image = cv2.imread(r'C:\Users\Ye Myat Moe\Documents\sp\computer_vision\quiz2\software_6511233\itadori.jpg')
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

f_transform = np.fft.fft2(gray_image)
f_shift = np.fft.fftshift(f_transform)

rows, cols = gray_image.shape
crow, ccol = rows // 2 , cols // 2
radius = 30  

mask = np.zeros((rows, cols), np.uint8)
cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)

f_shift_filtered = f_shift * mask

f_ishift = np.fft.ifftshift(f_shift_filtered)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

output_path = r'C:\Users\Ye Myat Moe\Documents\sp\computer_vision\quiz2\software_6511233'  

cv2.imwrite(f'{output_path}\\original_color_image.jpg', color_image)  
cv2.imwrite(f'{output_path}\\original_grayscale_image.jpg', gray_image)  
cv2.imwrite(f'{output_path}\\modified_image.jpg', img_back.astype(np.uint8))

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title('Original Color Image')
plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title('Original Grayscale Image')
plt.imshow(gray_image, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Modified Image after Low-pass Filter')
plt.imshow(img_back, cmap='gray')

plt.show()
