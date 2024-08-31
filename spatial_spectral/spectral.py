import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy.fft import fft2, ifft2, fftshift, ifftshift

image_path = r'C:\Users\Ye Myat Moe\Documents\SP\Computer_Vision\quiz\img.jpg'


image = Image.open(image_path).convert('L')
image_array = np.array(image)

f_transform = fft2(image_array)
f_transform_shifted = fftshift(f_transform)

center = (f_transform_shifted.shape[0] // 2, f_transform_shifted.shape[1] // 2)
f_transform_shifted[center] *= 0.25

f_transform_modified = ifftshift(f_transform_shifted)
image_modified = ifft2(f_transform_modified)
image_modified = np.abs(image_modified)

image_modified_pil = Image.fromarray(np.uint8(image_modified))
image_modified_pil.save('modified_image.bmp')

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image_array, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Modified Image')
plt.imshow(image_modified, cmap='gray')
plt.axis('off')

plt.show()

difference = np.abs(image_array - image_modified)
print("Mean difference between original and modified images:", np.mean(difference))
