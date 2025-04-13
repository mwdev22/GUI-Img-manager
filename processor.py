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
            raise ValueError("Nie udało się pobrać obrazu")
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
            raise ValueError("obraz musi być w formacie RGB")
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
    def save_histogram(path, histogram):
        with open(path, 'w') as file:
            if isinstance(histogram[0], (list, np.ndarray)):  # ensures that the histogram is RGB
                file.write("Czerowny kanał:\n")
                file.write("\n".join(map(str, histogram[0])) + "\n")
                file.write("\nZielony kanał:\n")
                file.write("\n".join(map(str, histogram[1])) + "\n")
                file.write("\nNiebieski kanał:\n")
                file.write("\n".join(map(str, histogram[2])) + "\n")
            else:                                             # otherwise, it is grayscale
                file.write("Histogram szaroodcieniowy:\n")
                file.write("\n".join(map(str, histogram)) + "\n") 
    
    @staticmethod
    def stretch_histogram(image):
        image = Image.fromarray(image)
        pixels = list(image.getdata())

        min_pixel = min(pixels)
        max_pixel = max(pixels)

        if min_pixel == max_pixel:
            return np.array(image)  # unikamy dzielenia przez 0

        # poprawiona normalizacja do [0, 255]
        normalize = lambda value: int((value - min_pixel) * 255 / (max_pixel - min_pixel))

        new_pixels = [normalize(p) for p in pixels]
        stretched = image.copy()
        stretched.putdata(new_pixels)

        return np.array(stretched)



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
            plt.title('Bazowy Histogram')
            
            plt.subplot(1, 2, 2)
            plt.hist(processed.ravel(), 256, [0, 256], color='black')
            plt.title('Histogram po operacji')
        else:  # Color
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                plt.subplot(2, 3, i+1)
                plt.hist(original[:, :, i].ravel(), 256, [0, 256], color=color)
                plt.title(f'Bazowy {color.upper()} kanał')
                
                plt.subplot(2, 3, i+4)
                plt.hist(processed[:, :, i].ravel(), 256, [0, 256], color=color)
                plt.title(f'{color.upper()} kanał po operacji')
        
        plt.tight_layout()
        plt.show()
        
        
# ----------------- POINT 1-ARG OPERATIONS ----------------

    @staticmethod
    def find_min_max(cv2_image):

        min_pixel = np.min(cv2_image)
        max_pixel = np.max(cv2_image)
        return min_pixel, max_pixel

    @staticmethod
    def negate_image(cv2_image):
        # change img value to lmax - p
        return LMAX - cv2_image

    @staticmethod
    def stretch_range(cv2_image, p1, p2, q3=0, q4=LMAX):
        stretched_image = ((cv2_image - p1) / (p2 - p1)) * (q4 - q3) + q3
        return np.clip(stretched_image, 0, 255).astype(np.uint8)  # make sure that values are in  [0, 255] range

    def posterize_image(self, image: np.ndarray, num_levels=4):

        
        if num_levels < 2 or num_levels > 256:
           raise ValueError("Poziomy szarości muszą być w zakresie 2-256")

        if not self.is_grayscale(image):
            raise ValueError("Obraz musi być szaroodcieniowy")

        height, width = image.shape
        step = 256 // num_levels
        posterized = np.zeros_like(image)

        for y in range(height):
            for x in range(width):
                pixel = image[y, x]
                posterized[y, x] = (pixel // step) * step

        return posterized

        
# --------------- NEIGHBOURHOOD OPERATIONS -------------------
    @staticmethod
    def blur(image):
        return cv2.blur(image)
    
    @staticmethod
    def gaussian_blur(image):
        return cv2.GaussianBlur(image)
    
    @staticmethod
    def sobel(image):
        return cv2.Sobel(image)
    
    @staticmethod
    def laplacian(image):
        return cv2.Laplacian(image)
    
    @staticmethod
    def canny(image):
        return cv2.Canny(image)
    
    @staticmethod
    def sharpen_linear(image, mask):
        return cv2.filter2D(image, kernel=mask)
    
    def sharpen_linear_laplacian(self, image, masks):
        return [self.sharpen_linear(image, mask) for mask in masks]
    
    def direct_edge_detection(image):
        ...

    