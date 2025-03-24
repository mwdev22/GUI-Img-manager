import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageTk

class ImageProcessor:

    @staticmethod
    def load_image(path, grayscale=False):
        mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        image = cv2.imread(path, mode)
        if image is None:
            raise ValueError("Failed to load image")
        return image

    @staticmethod
    def convert_to_tkimage(cv2_image):
        image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        return ImageTk.PhotoImage(img_pil)

    # check if img is 2D
    @staticmethod
    def is_grayscale(image):
        return image.ndim == 2
    
    @staticmethod
    def is_rgb(image):
        # check if the image is 3D and has 3 channels
        return image.ndim == 3 and image.shape[2] == 3

    # conversions   
    @staticmethod
    def to_grayscale(cv2_image):
        return cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    

    @staticmethod
    def to_hsv(cv2_image):
        return cv2.cvtColor(cv2_image, cv2.COLOR_BGR2HSV)
    
    @staticmethod
    def to_lab(cv2_image):
        return cv2.cvtColor(cv2_image, cv2.COLOR_BGR2LAB)

    @staticmethod
    def split_rgb_channels(cv2_image):
        if not ImageProcessor.is_rgb(cv2_image):
            raise ValueError("Image must be in RGB format")
        return cv2.split(cv2_image)

    
    
    # histograms            
    @staticmethod
    def grayscale_histogram(image_data):
        histogram = [0] * 256  

        # all pixels incrementing its value
        for pixel in image_data.ravel():
            histogram[pixel] += 1  

        plt.bar(range(256), histogram, color='black', alpha=0.75)
        plt.title("Histogram (Grayscale)")
        plt.xlabel("Intensywność")
        plt.ylabel("Częstotliwość")
        plt.show()
        
        return histogram
    
    @staticmethod
    def rgb_histogram(cv2_image):
        colors = ('b', 'g', 'r') 
        histograms = []
        for i, color in enumerate(colors):
            channel_histogram, _ = np.histogram(cv2_image[:, :, i].ravel(), bins=256, range=(0, 256))
            histograms.append(channel_histogram)
            plt.hist(cv2_image[:, :, i].ravel(), bins=256, range=(0, 256), color=color, alpha=0.5)
        plt.title("Histogram (RGB)")
        plt.xlabel("Intensywność")
        plt.ylabel("Częstotliwość")
        plt.show()

        return histograms
    
    