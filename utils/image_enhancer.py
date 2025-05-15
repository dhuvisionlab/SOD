
from PIL import Image, ImageEnhance
import numpy as np

def calculate_brightness(image):
    # Convert the PIL image to a NumPy array
    image_np = np.array(image)

    # Convert the image to grayscale
    grayscale_image = np.dot(image_np[..., :3], [0.2989, 0.5870, 0.1140])

    # Calculate the average pixel value (brightness)
    avg_brightness = np.mean(grayscale_image)

    return avg_brightness
    
def enhance_dark_images(image, brightness_factor=0, threshold=0):
    # Convert the PIL image to a NumPy array
    image_np = np.array(image)

    # Calculate brightness of the image
    brightness = np.dot(image_np[..., :3], [0.2989, 0.5870, 0.1140]).mean()

    # Check if the image is dark (brightness below the threshold)
    if brightness < threshold:
        # Enhance brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)

    return image

