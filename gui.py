from tkinter import Tk, Menu, filedialog, Label, messagebox, Toplevel, Frame, Button, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from utils import image_required
from processor import ImageProcessor


class GUI:
    def __init__(self):
        # base window config
        self.root = Tk()
        self.root.title("Image Processor")
        self.root.geometry("800x600")
        self.root.option_add('*tearOff', False)
        
        # processor for image operations
        self.processor = ImageProcessor()

        self.mount_menu()
        
        # frame to hold the image label
        frame = Frame(self.root)
        frame.pack(fill="both", expand=True)

        # image label for displaying
        image_label = Label(frame)
        image_label.pack(fill="both", expand=True)
        
        self.windows = {
            self.root : WindowContext(self.root, image_label, None, frame=frame, processor=self.processor)
        }
        
        self.context = self.windows[self.root]
        # resize event
        

    def mount_menu(self, root=None):
        if root is None:
            root = self.root
        menubar = Menu(root)
        self.root.config(menu=menubar)

        file_menu = self.create_file_menu(menubar)
        menubar.add_cascade(label="Plik", menu=file_menu)

        process_menu = self.create_process_menu(menubar)
        menubar.add_cascade(label="Przetwarzanie", menu=process_menu)

    def create_file_menu(self, menubar):
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Wczytaj obraz", command=self.open_file_dialog)
        file_menu.add_command(label="Resetuj obraz", command=self.reset_image)
        file_menu.add_command(label="Zamknij", command=self.root.quit)
        return file_menu

    def create_process_menu(self, menubar):
        process_menu = Menu(menubar, tearoff=0)
        process_menu.add_command(label="Skaluj do odcieni szarości", command=self.apply_grayscale)
        process_menu.add_command(label="Histogram", command=self.show_histogram)
        process_menu.add_command(label="Konwersja HSV", command=self.apply_hsv)
        process_menu.add_command(label="Konwersja LAB", command=self.apply_lab)
        process_menu.add_command(label="Rozdziel kanały RGB", command=self.split_channels)

        histogram_menu = Menu(process_menu, tearoff=0)
        histogram_menu.add_command(label="Rozciągnij histogram", command=self.apply_stretch_histogram)
        histogram_menu.add_command(label="Wyrównaj histogram", command=self.apply_equalize_histogram)
        process_menu.add_cascade(label="Operacje na histogramie", menu=histogram_menu)

        point_op_menu = Menu(process_menu, tearoff=0)
        point_op_menu.add_command(label="Negacja", command=self.apply_negation)
        point_op_menu.add_command(label="Rozciąganie zakresu", command=self.apply_stretch_range)
        point_op_menu.add_command(label="Posteryzacja", command=self.apply_posterization)
        process_menu.add_cascade(label="Operacje punktowe", menu=point_op_menu)

        return process_menu
    

    def open_file_dialog(self):
        file_paths = filedialog.askopenfilenames(
            title="Wybierz obraz",
            filetypes=[("Pliki graficzne", ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.gif"))]
        )

        if file_paths:
            self.load_image(file_paths[0])


    #   info messages for user
    def show_message(self, title, message):
        messagebox.showinfo(title, message)
    
    def show_error(self, title, message):
        messagebox.showerror(title, message)
    
    ask_for_input_int = lambda self, title, prompt: simpledialog.askinteger(title, prompt, minvalue=2, maxvalue=256)

    
        
    def load_image(self, path):
        ctx = self.windows.get(self.root)
        ctx.image = self.processor.load_image(path)
        ctx.og_image = ctx.image.copy()
        ctx.label.bind("<Button-1>", lambda event, root=ctx.root, img=ctx.image: self.set_current_image(root, img))
        self.context.adjust_window_size()
        self.context.display_current_image()

    # image processsing
    @image_required
    def show_new_window(self, title, image, channel_image: np.ndarray):
        new_window = Toplevel(self.root)
        new_window.title(title)
        
        frame = Frame(new_window)
        frame.pack(fill="both", expand=True)

        label = Label(new_window, image=image)
        label.pack(fill="both", expand=True)
        label.image = image
        
        context = self.windows[new_window] = WindowContext(new_window, label, channel_image, frame, self.processor)
        self.context.adjust_window_size()
        self.context.display_current_image()
        new_window.bind("<Configure>", lambda e, ctx=context: ctx.on_window_resize(e))
        label.bind("<Button-1>", lambda event, root=new_window, img=channel_image: self.set_current_image(root, img))
        
    
    def set_current_image(self, root, img: np.ndarray):
        context = self.windows.get(root)
        if context:
            context.image = img
            self.context = context
            context.display_current_image()

    def reset_image(self):
        self.context.image = self.context.og_image.copy()
        self.context.display_current_image()


    @image_required
    def apply_grayscale(self):
        gray_image = self.processor.to_grayscale(self.context.image)
        self.context.image = gray_image
        self.context.display_current_image()
        
    @image_required
    def apply_hsv(self):
        if not self.processor.is_rgb(self.context.image):
            self.show_error("Nieprawidłowy obraz", "Konwersja do HSV wymaga obrazu RGB.")
            return
        hsv_image = self.processor.to_hsv(self.context.image)
        self.context.image = hsv_image
        self.context.display_current_image()

    @image_required
    def apply_lab(self):
        if not self.processor.is_rgb(self.context.image):
            self.show_error("Nieprawidłowy obraz", "Konwersja do LAB wymaga obrazu RGB.")
            return
        lab_image = self.processor.to_lab(self.context.image)
        self.context.image = lab_image
        self.context.display_current_image()

    def show_histogram(self):
        if self.processor.is_grayscale(self.context.image):
            histogram = self.processor.grayscale_histogram(self.context.image)
        else:
            histogram = self.processor.rgb_histogram(self.context.image)

        save = messagebox.askyesno("Zapisz histogram", "Czy chcesz zapisać histogram?")
        if save:
            file_path = filedialog.asksaveasfilename(
                title="Zapisz histogram",
                defaultextension=".txt",
                filetypes=[("Pliki tekstowe", "*.txt")]
            )

            if file_path:
                self.processor.save_histogram(file_path, histogram)
                messagebox.showinfo("Sukces", f"Histogram zapisany w {file_path}")
        

    @image_required
    def split_channels(self):
        if not self.processor.is_rgb(self.context.image):
            self.show_error("Invalid Image", "Channel splitting requires RGB image")
            return
        
        channels = self.processor.split_rgb_channels(self.context.image)
        channel_names = ["Red Channel", "Green Channel", "Blue Channel"]
        
        for i, channel in enumerate(channels):
            channel_image = cv2.merge([channel, channel, channel])  
            tk_channel_image = self.processor.convert_to_tkimage(channel_image)
            self.show_new_window(channel_names[i], tk_channel_image, channel_image) 

    
    @image_required
    def apply_equalize_histogram(self):
        if not self.processor.is_grayscale(self.context.image):
            self.show_error("Błąd", "Wyrównywanie histogramu działa tylko na obrazach w skali szarości.")
            return
        processed = self.processor.equalize_histogram(self.context.image)
        self.context.image = processed
        self.context.display_current_image()
        self.processor.compare_histograms(self.context.image, processed)

    @image_required
    def apply_stretch_histogram(self):
        if not self.processor.is_grayscale(self.context.image):
            self.show_error("Błąd", "Rozciąganie histogramu działa tylko na obrazach w skali szarości.")
            return
        processed = self.processor.stretch_histogram(self.context.image)
        self.context.image = processed
        self.context.display_current_image()
        self.processor.compare_histograms(self.context.image, processed)

    @image_required
    def apply_negation(self):
        negated_image = self.processor.negate_image(self.context.image)
        self.context.image = negated_image
        self.context.display_current_image()

    @image_required
    def apply_stretch_range(self):
        p1, p2 = self.processor.find_min_max(self.context.image)
        stretched_image = self.processor.stretch_range(self.context.image, p1, p2)
        self.context.image = stretched_image
        self.context.display_current_image()
        
    @image_required
    def apply_posterization(self):
        if not self.processor.is_grayscale(self.context.image):
            self.show_error("Błąd", "Posteryzacja działa tylko na obrazach w odcieniach szarości.")
            return
        try:
            levels = int(self.ask_for_input_int("Poziomy posteryzacji", "Podaj liczbę poziomów szarości (2-256):"))
            processed = self.processor.posterize_image(self.context.image, num_levels=levels)
            self.context.image = processed
            self.context.display_current_image()
        except ValueError:
            self.show_error("Nieprawidłowe dane", "Liczba poziomów szarości musi być z zakresu 2-256.")


    def run(self):
        self.root.mainloop()
        
        
class WindowContext:
    def __init__(self, root, label, image=None, frame=None, processor: ImageProcessor = None):
        self.root: Union[Tk, Toplevel] = root
        self.label: Label = label
        self.image: Union[np.ndarray, None] = image
        self.og_image: Union[np.ndarray, None] = image.copy() if image is not None else None
        self.img_frame: Frame = frame
        self.tk_image: Union[ImageTk.PhotoImage, None]  = None
        self.processor = processor
        self.max_width = 1200
        self.max_height = 900
        self.root.geometry("800x600")  
    
    def on_window_resize(self, event):
            
        # get current window size (accounting for decorations)
        width = max(self.root.winfo_width(), 1)
        height = max(self.root.winfo_height(), 1)
            
        self.display_current_image(width, height)
    
    def display_current_image(self, width=None, height=None):
        if self.image is None:
            return
            
        # get current dimensions if not provided
        if width is None:
            width = max(self.root.winfo_width(), 1)
        if height is None:
            height = max(self.root.winfo_height(), 1)
            if isinstance(self.root, Tk):
                height = max(height - 30, 1)

        resized = self.resize_image_to_fit(self.image, width, height)
        tk_image = self.processor.convert_to_tkimage(resized)

        self.tk_image = tk_image
        self.label.config(image=tk_image)
        self.label.image = tk_image
    
    def resize_image_to_fit(self, image, width, height):
        # ensure minimum dimensions
        width = max(width, 1)
        height = max(height, 1)
        
        h, w = image.shape[:2]
        
        # calculate aspect ratios
        img_ratio = w / h
        win_ratio = width / height
        
        if img_ratio > win_ratio:
            # image is wider relative to window
            new_width = width
            new_height = int(width / img_ratio)
        else:
            new_height = height
            new_width = int(height * img_ratio)
            
        new_width = max(new_width, 1)
        new_height = max(new_height, 1)
        
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized

    def adjust_window_size(self):
        if self.image is None:
            return
            
        h, w = self.image.shape[:2]
        margin_w = 50
        margin_h = 80
        new_width = min(w + margin_w, self.max_width)
        new_height = min(h + margin_h, self.max_height)
        
        self.root.geometry(f"{new_width}x{new_height}")

if __name__ == "__main__":
    app = GUI()
    app.run()