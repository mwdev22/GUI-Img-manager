from tkinter import Tk, Menu, filedialog, Label, messagebox
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
        

        # image variables for storing image data
        self.og_image = None
        self.current_image = None
        self.tk_image = None
        
        # processor for image operations
        self.processor = ImageProcessor()

        self.mount_menu()

        # image label for displaying
        self.image_label = Label(self.root)
        self.image_label.pack(fill="both", expand=True)

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
        return process_menu
    
    

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
        # save the cv2 image for processing
        self.current_image = self.processor.load_image(path)
        # og image for possible reset
        self.og_image = self.current_image.copy()
        # tk image for displaying
        self.tk_image = self.processor.convert_to_tkimage(self.current_image)
        self.image_label.config(image=self.tk_image)
        self.image_label.image = self.tk_image  
        
    def reset_image(self):
        self.current_image = self.og_image.copy()
        self.tk_image = self.processor.convert_to_tkimage(self.current_image)
        self.image_label.config(image=self.tk_image)
        self.image_label.image = self.tk_image

    # image operations
    @image_required
    def apply_grayscale(self):
        gray_image = self.processor.to_grayscale(self.current_image)
        self.tk_image = self.processor.convert_to_tkimage(gray_image)
        self.current_image = gray_image
        self.image_label.config(image=self.tk_image)
        self.image_label.image = self.tk_image
        
    @image_required
    def apply_hsv(self):
        if not self.processor.is_rgb(self.current_image):
            self.show_error("Invalid Image", "HSV conversion requires RGB image")
            return
        hsv_image = self.processor.to_hsv(self.current_image)
        self.tk_image = self.processor.convert_to_tkimage(hsv_image)
        self.current_image = hsv_image
        self.image_label.config(image=self.tk_image)
        self.image_label.image = self.tk_image
        
    @image_required
    def apply_lab(self):
        if not self.processor.is_rgb(self.current_image):
            self.show_error("Invalid Image", "LAB conversion requires RGB image")
            return
        lab_image = self.processor.to_lab(self.current_image)
        self.tk_image = self.processor.convert_to_tkimage(lab_image)
        self.current_image = lab_image
        self.image_label.config(image=self.tk_image)
        self.image_label.image = self.tk_image

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
            

    def run(self):
        self.root.mainloop()
        



if __name__ == "__main__":
    app = GUI()
    app.run()