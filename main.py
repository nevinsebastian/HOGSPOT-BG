import cv2
import numpy as np
from PIL import Image

def remove_background_and_keep_signature(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to identify light areas (the background)
    _, binary_image = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Invert the binary image to make signature part white and background black
    binary_image_inv = cv2.bitwise_not(binary_image)

    # Create a mask where background (white areas) become transparent
    mask = binary_image_inv

    # Convert image to RGBA (add alpha channel)
    b, g, r = cv2.split(image)

    # Merge the RGB channels with the mask as the alpha channel
    rgba_image = cv2.merge([b, g, r, mask])

    # Convert result to PIL format and save as PNG with transparency
    result_pil = Image.fromarray(rgba_image)
    result_pil.save(output_path, format="PNG")

    print(f"Processed image saved as {output_path}")

# Example usage
input_image_path = 'sign1.jpg'  # Replace with your image file path
output_image_path = '1.png'  # Replace with desired output file path

remove_background_and_keep_signature(input_image_path, output_image_path)
