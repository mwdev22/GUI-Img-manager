import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageTk
from typing import Union

LMAX = 255
LMIN = 0
BORDER_TYPES = {
    "reflect": cv2.BORDER_REFLECT,
    "replicate": cv2.BORDER_REPLICATE,
    # "constant": cv2.BORDER_CONSTANT,
    # "wrap": cv2.BORDER_WRAP,
    "default": cv2.BORDER_DEFAULT
}

LAPLACIAN_MASKS = [
    np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),
    np.array([[1, -2, 1], [-2, 5, -2], [1, -2, 1]])
]

PREWITT_KERNELS = {
    "N":  np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),
    "NE": np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]]),
    "E":  np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
    "SE": np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]),
    "S":  np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
    "SW": np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]]),
    "W":  np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),
    "NW": np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]])
}


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
        # convert to 8-bit if necessary
        if cv2_image.dtype != np.uint8:
            cv2_image = cv2.normalize(cv2_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # convert color space if needed
        if len(cv2_image.shape) == 2:  
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_GRAY2RGB)
        else:  
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        
        image_pil = Image.fromarray(cv2_image)
        return ImageTk.PhotoImage(image_pil)
    
    def to_np_arr(self, img):
        
        if isinstance(img, Image.Image):
            return np.array(img)
        elif isinstance(img, ImageTk.PhotoImage):
            return self.convert_tkimage_to_opencv(img)
        elif isinstance(img, np.ndarray):
            return img
        else:
            raise TypeError("Unsupported image type. The image must be PIL, Tkinter.PhotoImage, or a numpy array.")

    @staticmethod
    def convert_to_opencv_format(image):
        if isinstance(image, ImageTk.PhotoImage):
            image = ImageProcessor.convert_tkimage_to_opencv(image)
        
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:  
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif len(image.shape) == 2:  
                pass
        
        else:
            raise TypeError("Unsupported image type. The image must be PIL, Tkinter.PhotoImage, or a numpy array.")
        
        return image

    @staticmethod
    def convert_tkimage_to_opencv(tk_image: ImageTk.PhotoImage):
        """
        Convert Tkinter PhotoImage to OpenCV compatible BGR format.
        """
        # Convert Tkinter PhotoImage to RGB format
        
        image = np.array(tk_image)
        # Convert RGB to BGR for OpenCV compatibility
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # check if img is 2D
    @staticmethod
    def is_grayscale(image):
        return image.ndim == 2
    
    @staticmethod
    def is_rgb(image):
        # check if the image is 3D and has 3 channels
        return image.ndim == 3 and image.shape[2] == 3
    
    @staticmethod
    def is_binary(image):
        return image.ndim == 2 and np.array_equal(np.unique(image), [0, 255])

    # conversions   
    def to_grayscale(self, cv2_image):
        return cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY) if self.is_rgb(cv2_image) else cv2_image
    

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
            print(max_pixel, min_pixel)
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

# --------------- POINT 2-ARG OPERATIONS ----------------

    @staticmethod
    def add_images(image1, image2):
        return cv2.add(image1, image2)
    
    @staticmethod
    def subtract_images(image1, image2):
        return cv2.subtract(image1, image2)
    
    @staticmethod  
    def blend(image1, image2, alpha=0.5):
        return cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)
    
    @staticmethod    
    def XOR(image1, image2):
        return cv2.bitwise_xor(image1, image2)
    
    @staticmethod
    def AND(image1, image2):
        return cv2.bitwise_and(image1, image2)
    
    @staticmethod
    def OR(image1, image2):
        return cv2.bitwise_or(image1, image2)
    
    @staticmethod
    def NOT(image1):
        return cv2.bitwise_not(image1)
        
    
    
        
# --------------- NEIGHBOURHOOD OPERATIONS -------------------
    @staticmethod
    def blur(image, ksize=(5,5)):
        return cv2.blur(image, ksize)

    @staticmethod
    def gaussian_blur(image, ksize=(5,5), sigmaX=0):
        return cv2.GaussianBlur(image, ksize, sigmaX)
    
    def sobel(self, image, border_type="default"):
        if not self.is_grayscale(image):
            image = self.to_grayscale(image)
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, borderType=BORDER_TYPES[border_type])
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, borderType=BORDER_TYPES[border_type])
        return cv2.magnitude(sobel_x, sobel_y)

    def laplacian(self, image, border_type="default"):
        if not self.is_grayscale(image):
            image = self.to_grayscale(image)
        laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3, borderType=BORDER_TYPES[border_type])
        return np.absolute(laplacian)
    
    def canny(self, image, threshold1=100, threshold2=200, apertureSize=3, L2gradient=False):
        if not self.is_grayscale(image):
            image = self.to_grayscale(image)
        edges = cv2.Canny(image, threshold1, threshold2, 
                        apertureSize=apertureSize, 
                        L2gradient=L2gradient,
                        )
        return edges
    
    @staticmethod
    def custom_mask(image, mask, border_type="default"):
        return cv2.filter2D(image, -1, kernel=mask, borderType=BORDER_TYPES[border_type])
    
    def sharpen_linear_laplacian(self, image, border_type='default', masks=LAPLACIAN_MASKS):
        return [self.custom_mask(image, mask, border_type=border_type) for mask in masks]
    
    def direct_edge_detection(self, image, border_type="default"):
        results = {}
        border_code = BORDER_TYPES.get(border_type, cv2.BORDER_REFLECT)
        
        if not self.is_grayscale(image):
            image = self.to_grayscale(image)
        
        for direction, kernel in PREWITT_KERNELS.items():
            filtered = cv2.filter2D(image, cv2.CV_64F, kernel=kernel, borderType=border_code)
            filtered = np.absolute(filtered)
            filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            results[direction] = filtered
        
        return results
    
    @staticmethod
    def median_filter(image, kernel_size=3, border_type="reflect"):
 
        if kernel_size not in [3, 5, 7]:
            raise ValueError("Rozmiar jądra musi być 3, 5 lub 7")
        
        
        return cv2.medianBlur(image, ksize=kernel_size)
    
    
# --------------- GEOMETRIC TRANSFORMATIONS -------------------
    @staticmethod
    def two_stage_filter(image, smooth_kernel, sharpen_kernel, border_type=cv2.BORDER_DEFAULT):
        
        smoothed = cv2.filter2D(image, -1, smooth_kernel, borderType=BORDER_TYPES[border_type])
        
        sharpened = cv2.filter2D(smoothed, -1, sharpen_kernel, borderType=BORDER_TYPES[border_type])
        
        return smoothed, sharpened


    @staticmethod
    def combine_kernels(kernel1, kernel2):
        
        combined = np.zeros((5, 5), dtype=np.float32)
        for i in range(3):  # kernel1 rows
            for j in range(3):  # kernel1 cols
                combined[i:i+3, j:j+3] += kernel1[i,j] * kernel2
        
        # normalize while preserving sign
        sum_abs = np.sum(np.abs(combined))
        return combined / sum_abs if sum_abs != 0 else combined
    
    @staticmethod
    def img_diff(image1, image2):
        return cv2.absdiff(image1, image2)
    
    
# ----------------- MORPHOLOGY OPERATIONS -------------------

    @staticmethod
    def binarize(image, threshold=127):
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return binary

    @staticmethod
    def get_rhombus_kernel(size=3):
        if size % 2 == 0:
            raise ValueError("Rozmiar kernela musi być nieparzysty")
        
        radius = size // 2
        kernel = np.zeros((size, size), np.uint8)
        
        for i in range(size):
            for j in range(size):
                if abs(i - radius) + abs(j - radius) <= radius:
                    kernel[i, j] = 1
        return kernel

    @staticmethod
    def get_square_kernel(size=3):
        return np.ones((size, size), np.uint8)
    
    def morph_operation(self, image, operation='erosion', kernel_type='rhombus', kernel_size=3, 
        border_type=cv2.BORDER_CONSTANT, border_value=0):
        
        if not self.is_binary(image):
            raise ValueError("Obraz musi być zbinarnyzowany.")
        
        if kernel_type == 'rhombus':
            kernel = self.get_rhombus_kernel(kernel_size)
        elif kernel_type == 'square':
            kernel = self.get_square_kernel(kernel_size)
        else:
            raise ValueError("Zły typ kernela. Wybierz 'rhombus' lub 'square'")
        
        res = getattr(self, operation)(image, kernel, border_type=border_type, border_value=border_value)
        if res is None:
            raise ValueError(f"Zła operacja: {operation}. Wybierz 'erode', 'dilate', 'open', lub 'close'")
        return res
    
    @staticmethod
    def dilate(image, kernel, border_type=cv2.BORDER_CONSTANT, border_value=0):
        return cv2.dilate(image, kernel, iterations=1, borderType=border_type, borderValue=border_value)

    @staticmethod
    def erode(image, kernel, border_type=cv2.BORDER_CONSTANT, border_value=0):
        return cv2.erode(image, kernel, iterations=1, borderType=border_type, borderValue=border_value)
    
    @staticmethod
    def open(image, kernel,border_type=cv2.BORDER_CONSTANT, border_value=0):
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, borderType=border_type, borderValue=border_value)

    @staticmethod
    def close(image, kernel,border_type=cv2.BORDER_CONSTANT, border_value=0):
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, borderType=border_type, borderValue=border_value)
    
    @staticmethod
    def skeletonize(image, border_type=cv2.BORDER_CONSTANT, max_iterations=100):

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
            
        if np.mean(image) > 127:
            image = 255 - image
            
        skeleton = image.copy()
        prev_skeleton = np.zeros_like(skeleton)
        
        kernel1 = np.array([[0, 0, 0],
                           [-1, 1, -1],
                           [1, 1, 1]], dtype=np.int8)
        
        kernel2 = np.array([[-1, 0, 0],
                           [1, 1, 0],
                           [-1, 1, -1]], dtype=np.int8)
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            prev_skeleton[:] = skeleton
            
            hitmiss1 = cv2.morphologyEx(skeleton, cv2.MORPH_HITMISS, kernel1, 
                                      borderType=border_type)
            skeleton = skeleton - hitmiss1
            
            hitmiss2 = cv2.morphologyEx(skeleton, cv2.MORPH_HITMISS, kernel2, 
                                      borderType=border_type)
            skeleton = skeleton - hitmiss2
            
            if np.all(skeleton == prev_skeleton):
                break
                
        skeleton = 255 - skeleton
        
        return skeleton