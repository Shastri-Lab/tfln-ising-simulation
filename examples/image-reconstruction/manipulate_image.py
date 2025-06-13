# import the image in images/black-cat.jpg into a numpy array
import numpy as np
from PIL import Image

def load_image(image_path):
    """
    Load an image from the specified path and convert it to a numpy array.
    
    Args:
        image_path (str): The path to the image file.
        
    Returns:
        np.ndarray: The image as a numpy array.
    """
    img = Image.open(image_path)
    img_array = np.array(img)
    return img_array


image = load_image("images/black-cat.jpg")

# now display the image
import matplotlib.pyplot as plt
def display_image(image_array):
    """
    Display an image from a numpy array.
    
    Args:
        image_array (np.ndarray): The image as a numpy array.
    """
    plt.imshow(image_array)
    plt.axis('off')  # Hide the axes
    plt.show()

# turn the image into black and white
def convert_to_black_and_white(image_array, threshold=128):
    """
    Convert an image to black and white based on a threshold.
    
    Args:
        image_array (np.ndarray): The image as a numpy array.
        threshold (int): The threshold value for converting to black and white.
        
    Returns:
        np.ndarray: The black and white image as a numpy array.
    """
    # Convert to grayscale if the image is RGB
    if len(image_array.shape) == 3:
        gray_image = np.mean(image_array, axis=2)
    else:
        gray_image = image_array
    
    # Apply the threshold
    bw_image = (gray_image > threshold).astype(np.uint8) * 255
    return bw_image

bw_image = convert_to_black_and_white(image)

# scale image to 20x20

def scale_image(image_array, new_size=(20, 20)):
    """
    Scale an image to a new size.
    
    Args:
        image_array (np.ndarray): The image as a numpy array.
        new_size (tuple): The new size as (width, height).
        
    Returns:
        np.ndarray: The scaled image as a numpy array.
    """
    img = Image.fromarray(image_array)
    img_resized = img.resize(new_size, Image.NEAREST)
    return np.array(img_resized)

scaled_image = scale_image(bw_image)

# save the scaled image to images/scaled_black_cat.png
def save_image(image_array, save_path):
    """
    Save a numpy array as an image file.
    
    Args:
        image_array (np.ndarray): The image as a numpy array.
        save_path (str): The path where the image will be saved.
    """
    img = Image.fromarray(image_array)
    img.save(save_path)
save_image(scaled_image, "images/scaled_black_cat.png")
# display the scaled image
display_image(scaled_image)