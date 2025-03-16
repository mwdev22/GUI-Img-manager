import numpy as np
from PIL import Image

def load_image(path):
    image = Image.open(path).convert("L")
    return np.array(image)

def compute_histogram(image):
    histogram = [0] * 256
    for row in image:
        for pixel in row:
            histogram[pixel] += 1
    return histogram

def stretch_histogram(image):
    I_min = np.min(image)
    I_max = np.max(image)
    if I_max == I_min:
        return image 
    stretched = ((image - I_min) / (I_max - I_min) * 255).astype(np.uint8)
    return stretched

def equalize_histogram(image):
    histogram = compute_histogram(image)
    cdf = np.cumsum(histogram)
    cdf_min = cdf[np.nonzero(cdf)[0][0]]  
    cdf_normalized = (cdf - cdf_min) / (cdf[-1] - cdf_min) * 255
    equalized = np.interp(image.flatten(), range(256), cdf_normalized).reshape(image.shape)
    return equalized.astype(np.uint8)

def save_image(image, filename):
    Image.fromarray(image).save(filename)
