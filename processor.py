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
    
    
    @staticmethod
    def equalize_histogram(image):
        if ImageProcessor.is_grayscale(image):
            return cv2.equalizeHist(image)
        else:
            # equalize each channel separately for color images
            channels = cv2.split(image)
            equalized_channels = []
            for ch in channels:
                equalized_channels.append(cv2.equalizeHist(ch))
            return cv2.merge(equalized_channels)

    @staticmethod
    def stretch_histogram(image):
        def process_channel(channel):
            min_val = np.min(channel)
            max_val = np.max(channel)
            if min_val == max_val: 
                return channel
            stretched = ((channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            return stretched
        
        if ImageProcessor.is_grayscale(image):
            process_channel(image)
        else:
            # for color images, process each channel separately
            channels = cv2.split(image)
            stretched_channels = []
            for ch in channels:
                stretched_ch = process_channel(ch)
                stretched_channels.append(stretched_ch)
            return cv2.merge(stretched_channels)

    @staticmethod
    def compare_histograms(original, processed):
        plt.figure(figsize=(12, 6))
        
        if original.ndim == 2:  # grayscale
            plt.subplot(1, 2, 1)
            plt.hist(original.ravel(), 256, [0, 256], color='black')
            plt.title('Original Histogram')
            
            plt.subplot(1, 2, 2)
            plt.hist(processed.ravel(), 256, [0, 256], color='black')
            plt.title('Processed Histogram')
        else:  # Color
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                plt.subplot(2, 3, i+1)
                plt.hist(original[:, :, i].ravel(), 256, [0, 256], color=color)
                plt.title(f'Original {color.upper()} Channel')
                
                plt.subplot(2, 3, i+4)
                plt.hist(processed[:, :, i].ravel(), 256, [0, 256], color=color)
                plt.title(f'Processed {color.upper()} Channel')
        
        plt.tight_layout()
        plt.show()