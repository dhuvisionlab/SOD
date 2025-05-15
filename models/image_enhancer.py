# image_enhancer.py
from PIL import Image, ImageEnhance

def calculate_brightness(image):
    # Convert the image to grayscale
    grayscale_image = image.convert('L')

    # Calculate the average pixel value (brightness)
    pixels = list(grayscale_image.getdata())
    avg_brightness = sum(pixels) / len(pixels)

    return avg_brightness

def enhance_dark_images(image, brightness_factor=1.5, threshold=100):
    # Calculate brightness of the image
    brightness = calculate_brightness(image)

    # Check if the image is dark (brightness below the threshold)
    if brightness < threshold:
        # Enhance brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)

    return image

