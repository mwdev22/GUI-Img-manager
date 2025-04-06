import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageTk

LMAX = 255
LMIN = 0

class ImageProcessor:
    

    # ---------- IMAGE CONVERSION -------------------
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
    
#  ------------ HISTOGRAMS ---------------------------
    
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
    def stretch_histogram(image):
        image = Image.fromarray(image)
        pixels = list(image.getdata())

        min_pixel = min(pixels)
        max_pixel = max(pixels)

        # normalizacja
        normalize = lambda value: int((value - min_pixel) / (max_pixel - min_pixel) * 255)

        # skalowanie pixeli
        new_pixels = [normalize(p) for p in pixels]
        stretched = image.copy()
        stretched.putdata(new_pixels)

        return np.array(stretched)  # <-- konwersja z powrotem


    @staticmethod
    def equalize_histogram(image):
        # konwersja do pil image z np.ndarray
        image = Image.fromarray(image)
        width, height = image.size
        pixels = list(image.getdata())

        # konwersja i pobranie pixeli
        histogram = [0] * 256
        for p in pixels:
            histogram[p] += 1

        # utworzenie histogramu
        cdf = [0] * 256
        cdf[0] = histogram[0]
        for i in range(1, 256):
            cdf[i] = cdf[i-1] + histogram[i]

        # skumulowany histogram
        cdf_min = next(v for v in cdf if v > 0)
        total_pixels = width * height
        
        # pierwszy niezerowy element 
        cdf_normalized = [
            round((cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255)
            for i in range(256)
        ]

        equalized_pixels = [cdf_normalized[p] for p in pixels]
        equalized = image.copy()
        equalized.putdata(equalized_pixels)

        return np.array(equalized)  # <-- konwersja z powrotem


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
        
        
# ----------------- POINT 1-ARG OPERATIONS ----------------

    @staticmethod
    def negate_image(cv2_image):
        # change img value to lmax - p
        return LMAX - cv2_image

    @staticmethod
    def stretch_range(cv2_image, p1, p2, q3=0, q4=LMAX):
        stretched_image = ((cv2_image - p1) / (p2 - p1)) * (q4 - q3) + q3
        return np.clip(stretched_image, 0, 255).astype(np.uint8)  # make sure that values are in  [0, 255] range

    @staticmethod
    def find_min_max(cv2_image):
        """
        Znajdowanie minimalnej i maksymalnej wartości piksela w obrazie
        """
        min_pixel = np.min(cv2_image)
        max_pixel = np.max(cv2_image)
        return min_pixel, max_pixel