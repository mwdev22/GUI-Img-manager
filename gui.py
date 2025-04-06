from tkinter import Tk, Menu, filedialog, Label, messagebox, Toplevel, Frame, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import image_required
from processor import ImageProcessor


class GUI:
    def __init__(self):
        # base window config
        self.root = Tk()
        self.root.title("Image Processor")
        self.root.geometry("800x600")
        self.root.option_add('*tearOff', False)
        
        # Set maximum window dimensions
        self.max_width = 1200
        self.max_height = 900
        
        # image variables for storing image data
        self.og_image = None
        self.current_image = None
        self.tk_image = None
        
        # processor for image operations
        self.processor = ImageProcessor()

        self.mount_menu()

        # frame to hold the image label
        self.image_frame = Frame(self.root)
        self.image_frame.pack(fill="both", expand=True)

        # image label for displaying
        self.image_label = Label(self.image_frame)
        self.image_label.pack(fill="both", expand=True)
        
        # resize event
        self.root.bind("<Configure>", self.on_window_resize)

    def mount_menu(self):
        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = self.create_file_menu(menubar)
        menubar.add_cascade(label="File", menu=file_menu)

        process_menu = self.create_process_menu(menubar)
        menubar.add_cascade(label="Process", menu=process_menu)
    
    
    def create_file_menu(self, menubar):
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Image", command=self.open_file_dialog)
        file_menu.add_command(label="Reset Image", command=self.reset_image)
        file_menu.add_command(label="Exit", command=self.root.quit)
        return file_menu
        
    def create_process_menu(self, menubar):
        process_menu = Menu(menubar, tearoff=0)
        process_menu.add_command(label="Grayscale", command=self.apply_grayscale)
        process_menu.add_command(label="Histogram", command=self.show_histogram)
        process_menu.add_command(label="HSV", command=self.apply_hsv)
        process_menu.add_command(label="LAB", command=self.apply_lab)
        process_menu.add_command(label="Split into RGB channels", command=self.split_channels)
        
        histogram_menu = Menu(process_menu, tearoff=0)
        histogram_menu.add_command(label="Stretch Histogram", command=self.apply_stretch_histogram)
        histogram_menu.add_command(label="Equalize Histogram", command=self.apply_equalize_histogram)
        # histogram_menu.add_command(label="Compare Histograms", command=self.compare_histograms)
        process_menu.add_cascade(label="Histogram Operations", menu=histogram_menu)
        
        point_op_menu = Menu(process_menu, tearoff=0)
        point_op_menu.add_command(label="Negation", command=self.apply_negation)
        point_op_menu.add_command(label="Stretch Range", command=self.apply_stretch_range)
        process_menu.add_cascade(label="Point Operations",menu=point_op_menu)
        
        return process_menu
    
    def resize_image_to_fit(self, image, width, height):
        h, w = image.shape[:2]
        
        aspect_ratio = w / h
        target_ratio = width / height
        
        if aspect_ratio > target_ratio:
            # image is wider than the target area
            new_width = width
            new_height = int(width / aspect_ratio)
        else:
            # image is taller than the target area
            new_width = int(height * aspect_ratio)
            new_height = height
            
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized
    
    def on_window_resize(self, event):
        if hasattr(self, 'current_image') and self.current_image is not None:
            window_width = self.root.winfo_width()
            window_height = self.root.winfo_height() - 30  # approximate menu height
            
            self.display_current_image(window_width, window_height)

    def open_file_dialog(self):
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image Files", ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.gif"))]
        )

        if file_paths:
            self.load_image(file_paths[0])

    #   info messages for user
    def show_message(self, title, message):
        messagebox.showinfo(title, message)
    
    def show_error(self, title, message):
        messagebox.showerror(title, message)
        
    def load_image(self, path):
        self.current_image = self.processor.load_image(path)
        self.og_image = self.current_image.copy()
        
        self.adjust_window_size()
        
        self.display_current_image()


    # window and image management
    def adjust_window_size(self):
        if self.current_image is None:
            return
            
        h, w = self.current_image.shape[:2]
        
        margin_w = 50
        margin_h = 80
        
        new_width = min(w + margin_w, self.max_width)
        new_height = min(h + margin_h, self.max_height)
        
        self.root.geometry(f"{new_width}x{new_height}")
        
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - new_width) // 2
        y = (screen_height - new_height) // 2
        self.root.geometry(f"+{x}+{y}")

    def display_current_image(self, width=None, height=None):
        if self.current_image is None:
            return
            
        if width is None or height is None:
            width = self.root.winfo_width()
            height = self.root.winfo_height() - 30  
        
        resized_image = self.resize_image_to_fit(self.current_image, width, height)
        
        self.tk_image = self.processor.convert_to_tkimage(resized_image)
        self.image_label.config(image=self.tk_image)
        self.image_label.image = self.tk_image

    def reset_image(self):
        self.current_image = self.og_image.copy()
        self.display_current_image()

    # image processsing
    @image_required
    def show_separated_image(self, title, image):
        new_window = Toplevel(self.root)
        new_window.title(title)
        
        h, w = image._PhotoImage__size
        
        max_width = min(w, 600)
        max_height = min(h, 500)
        new_window.geometry(f"{max_width}x{max_height}")
        
        label = Label(new_window, image=image)
        label.pack(fill="both", expand=True)
        label.image = image

    @image_required
    def apply_grayscale(self):
        gray_image = self.processor.to_grayscale(self.current_image)
        self.current_image = gray_image
        self.display_current_image()
        
    @image_required
    def apply_hsv(self):
        if not self.processor.is_rgb(self.current_image):
            self.show_error("Invalid Image", "HSV conversion requires RGB image")
            return
        hsv_image = self.processor.to_hsv(self.current_image)
        self.current_image = hsv_image
        self.display_current_image()
        
    @image_required
    def apply_lab(self):
        if not self.processor.is_rgb(self.current_image):
            self.show_error("Invalid Image", "LAB conversion requires RGB image")
            return
        lab_image = self.processor.to_lab(self.current_image)
        self.current_image = lab_image
        self.display_current_image()

    @image_required
    def show_histogram(self):
        if self.processor.is_grayscale(self.current_image):
            histogram = self.processor.grayscale_histogram(self.current_image)
        else:
            histogram = self.processor.rgb_histogram(self.current_image)
        
        save = messagebox.askyesno("Save Histogram", "Do you want to save the histogram?")
        if save:
            file_path = filedialog.asksaveasfilename(
                title="Save Histogram",
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt")]
            )

            if file_path:  
                self.processor.save_histogram(file_path, histogram)
                messagebox.showinfo("Success", f"Histogram saved to {file_path}")
        

    @image_required
    def split_channels(self):
        if not self.processor.is_rgb(self.current_image):
            self.show_error("Invalid Image", "Channel splitting requires RGB image")
            return
        
        channels = self.processor.split_rgb_channels(self.current_image)
        channel_names = ["Red Channel", "Green Channel", "Blue Channel"]
        
        for i, channel in enumerate(channels):
            channel_image = cv2.merge([channel, channel, channel])  
            tk_channel_image = self.processor.convert_to_tkimage(channel_image)
            self.show_separated_image(channel_names[i], tk_channel_image) 

    
    @image_required
    def apply_equalize_histogram(self):
        if not self.processor.is_grayscale(self.current_image):
            self.show_error("Error", "This only works for grayscale images.")
            return

        processed = self.processor.equalize_histogram(self.current_image)
        self.processor.compare_histograms(self.current_image, processed)
        self.current_image = processed
        self.display_current_image()
        


    @image_required
    def apply_stretch_histogram(self):
        if not self.processor.is_grayscale(self.current_image):
            self.show_error("Error", "This only works for grayscale images.")
            return

        processed = self.processor.stretch_histogram(self.current_image)
        self.processor.compare_histograms(self.current_image, processed)
        self.current_image = processed
        self.display_current_image()

    @image_required
    def apply_negation(self):
        negated_image = self.processor.negate_image(self.current_image)
        self.current_image = negated_image
        self.display_current_image()

    @image_required
    def apply_stretch_range(self):
        p1, p2 = self.processor.find_min_max(self.current_image)
        stretched_image = self.processor.stretch_range(self.current_image, p1, p2)
        self.current_image = stretched_image
        self.display_current_image()

    def run(self):
        self.root.mainloop()
        

if __name__ == "__main__":
    app = GUI()
    app.run()