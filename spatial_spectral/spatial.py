from PIL import Image
import numpy as np
import scipy.ndimage

image_path = 'C:/Users/Ye Myat Moe/Documents/SP/Computer_Vision/quiz/img.jpg'

try:
    # Attempt to open and convert the image
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)
    
    filter_mask = np.array([
        [2, 3, 3],
        [3, 2, 3],
        [3, 3, 2]
    ])

    # Perform correlation
    correlation_result = scipy.ndimage.correlate(image_array, filter_mask)
    correlation_image = Image.fromarray(correlation_result)
    correlation_image.save('correlation_result.bmp')

    # Perform convolution
    convolution_result = scipy.ndimage.convolve(image_array, filter_mask)
    convolution_image = Image.fromarray(convolution_result)
    convolution_image.save('convolution_result.bmp')

    # Show images
    image.show()
    correlation_image.show()
    convolution_image.show()

    # Calculate differences
    correlation_diff = np.abs(image_array - correlation_result)
    convolution_diff = np.abs(image_array - convolution_result)

    print("Correlation vs Original Image Difference:")
    print(np.mean(correlation_diff))

    print("Convolution vs Original Image Difference:")
    print(np.mean(convolution_diff))

except Exception as e:
    print(f"Error loading or processing image: {e}")
