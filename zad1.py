import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image(path, grayscale=False):
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(path, mode)
    if image is None:
        raise ValueError("Nie udało się wczytać obrazu.")
    return image

def compute_histogram(image):
    histogram = np.zeros(256, dtype=int)
    for pixel in image.flatten():
        histogram[pixel] += 1
    return histogram

def plot_histogram(histogram):
    plt.figure(figsize=(10, 5))
    plt.bar(range(256), histogram, color='black')
    plt.xlabel("Wartość piksela")
    plt.ylabel("Liczba pikseli")
    plt.title("Histogram obrazu")
    plt.show()

def save_histogram(histogram, filename="histogram.txt"):
    with open(filename, "w") as f:
        for i, value in enumerate(histogram):
            f.write(f"{i}: {value}\n")

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def split_channels(image):
    b, g, r = cv2.split(image)
    return r, g, b

def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def convert_to_lab(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

