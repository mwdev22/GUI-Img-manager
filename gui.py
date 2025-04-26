import tkinter as tk
from tkinter import (
    Tk, Menu, filedialog, Label, messagebox, Toplevel, Frame, 
    Button, Entry, simpledialog, LEFT, StringVar, 
    Radiobutton, W, IntVar, BOTH, E, END, EW, Scale, HORIZONTAL, X
)
from PIL import ImageTk, Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from functools import wraps
from processor import ImageProcessor, BORDER_TYPES
import os


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
        
        frame = Frame(self.root)
        frame.pack(fill="both", expand=True)

        # image label for displaying
        image_label = Label(frame)
        image_label.pack(fill="both", expand=True)

        
        self.windows = {
            self.root : WindowContext(self.root, image_label, None, frame=frame, processor=self.processor)
        }
        
        
        self.context = self.windows[self.root]
        
        self.profile_points = []
        self.profile_line_active = False
        
        self.root.bind("<Button-1>", lambda event, root=self.root: self.dispatch_click_handler(event, root))
        
        
    # helper decorator for operations
    def image_required(func):
        @wraps(func)    
        def wrapper(self, *args, **kwargs):
            if self.context.image is None:
                
                messagebox.showinfo("Brak Obrazu", "Najpierw wczytaj obraz, aby przeprowadzić operację.")
                self.open_file_dialog()
                    
                if self.context.image is not None:
                    return func(self, *args, **kwargs)  
                else:
                    messagebox.showwarning("Nie wybrano obrzu", "Obraz nie został wczytany.")
            else:
                return func(self, *args, **kwargs)  
        return wrapper
    
    #  ------------- MENU -------------------
    def mount_menu(self, root=None):
        if root is None:
            root = self.root
        menubar = Menu(root)
        self.root.config(menu=menubar)

        file_menu = self.create_file_menu(menubar)
        menubar.add_cascade(label="Plik", menu=file_menu)

        self.create_process_menus(menubar)
        

    def create_file_menu(self, menubar):
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Wczytaj obraz", command=self.open_file_dialog)
        file_menu.add_command(label="Resetuj obraz", command=self.reset_image)
        file_menu.add_command(label="Kompresja RLE", command=self.apply_rle_compression)
        file_menu.add_command(label="Zamknij", command=self.root.quit)
        return file_menu

    def create_process_menus(self, menubar: Menu):
        
        process_menu = Menu(menubar, tearoff=0)
        process_menu.add_command(label="Skaluj do odcieni szarości", command=self.apply_grayscale)
        process_menu.add_command(label="Binaryzacja", command=self.apply_binarization)
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
        
        neighboorhood_op_menu = Menu(menubar, tearoff=0)
        neighboorhood_op_menu.add_command(label="Bliur", command=lambda: self.dispatch_neighborhood('blur'))
        neighboorhood_op_menu.add_command(label="Gaussian Blur", command=lambda: self.dispatch_neighborhood('gaussian_blur'))
        neighboorhood_op_menu.add_command(label="Laplacian", command=lambda: self.dispatch_neighborhood('laplacian', ask_border=True))
        neighboorhood_op_menu.add_command(label="Sobel", command=lambda: self.dispatch_neighborhood('sobel', ask_border=True))
        neighboorhood_op_menu.add_command(label="Canny", command=lambda: self.dispatch_neighborhood('canny'))
        neighboorhood_op_menu.add_command(label="Prewitt", command=self.apply_prewitt)
        neighboorhood_op_menu.add_command(label="Własna maska", command=self.apply_custom_mask)
        neighboorhood_op_menu.add_command(label="Wyostrzanie liniowe (Laplacian)", command=self.apply_sharpen_laplacian)
        neighboorhood_op_menu.add_command(label="Filtracja Medianowa", command=self.apply_median_filter)
        
        process_menu.add_command(label="Operacje Dwuargumentowe", command=self.perform_two_arg_operation)
        
        process_menu.add_command(label="Filtracja 2 oraz 1 etapowa (porownanie)", command=self.compare_filtering_methods)
        
        
        segmentation_menu = Menu(menubar, tearoff=0)
        segmentation_menu.add_command(label="Ręczne progowanie",
                                    command=self.manual_threshold)
        segmentation_menu.add_command(label="Progowanie adaptacyjne",
                                    command=self.adaptive_threshold)
        segmentation_menu.add_command(label="Otsu",
                                    command=self.apply_otsu_threshold)
        segmentation_menu.add_command(label="GrabCut", command=self.apply_grabcut)
        segmentation_menu.add_command(label="Watershed", command=self.apply_watershed)
        segmentation_menu.add_command(label="Inpainting", command=self.apply_inpainting)

        advanced_menu = Menu(menubar, tearoff=0)
        
        morphology_menu = Menu(advanced_menu, tearoff=0)
        morphology_menu.add_command(label="Erozja", command=lambda: self.dispatch_morphology('erode'))
        morphology_menu.add_command(label="Dylatacja", command=lambda: self.dispatch_morphology('dilate'))
        morphology_menu.add_command(label="Otwarcie", command=lambda: self.dispatch_morphology('open'))
        morphology_menu.add_command(label="Zamknięcie", command=lambda: self.dispatch_morphology('close'))
        
        advanced_menu.add_cascade(label="Operacje Morfologiczne", menu=morphology_menu)
        advanced_menu.add_command(label="Szkieletyzacja", command=self.apply_skeletonization)
        advanced_menu.add_command(label="Detekcja krawędzi (Hough)", command=self.detect_lines_hough)
        advanced_menu.add_command(label="Linia Profilu", command=self.start_profile_line)
        advanced_menu.add_command(label="Piramida obrazów", command=self.show_image_pyramid)

        menubar.add_cascade(label="Przetwarzanie", menu=process_menu)
        menubar.add_cascade(label="Operacje sąsiedztwa", menu=neighboorhood_op_menu)
        menubar.add_cascade(label="Operacje zaawansowane", menu=advanced_menu)
        menubar.add_cascade(label="Segmentacja", menu=segmentation_menu)

    # --------- END OF MENU -------------------

    # --------- USER INTERACTIONS -------------
    
    def get_window_names(self):
        return [window.title() for window in self.windows.keys() if isinstance(window, Toplevel) or isinstance(window, Tk)]
    
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
    ask_for_input_str = lambda self, title, prompt: simpledialog.askstring(title, prompt)
    
    def _create_centered_dialog(self, title, width=None, height=None):
        dialog = Toplevel()
        dialog.title(title)
        dialog.resizable(False, False)
        return dialog

    def _center_dialog(self, dialog: Toplevel):
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'+{x}+{y}')

    def _add_ok_cancel_buttons(self, dialog: Toplevel, on_ok):
        btn_frame = Frame(dialog)
        btn_frame.pack(pady=5)
        
        Button(btn_frame, text="OK", command=on_ok).pack(side=LEFT, padx=5)
        Button(btn_frame, text="Anuluj", command=dialog.destroy).pack(side=LEFT, padx=5)
        return btn_frame
    
    def find_window_by_title(self, title):
        for window in self.windows:
            if window.title() == title:
                return self.windows[window]
        return 
    
    

    

    def hough_params_dialog(self):
        
        dialog = self._create_centered_dialog("Parametry Hougha")
        main_frame = Frame(dialog, padx=10, pady=10)
        main_frame.pack(fill=BOTH, expand=True)
        
        main_frame.grid_columnconfigure(1, weight=1)
        
        params = [
            {'name': 'rho', 'label': "Rozdzielczość odległości (piksele):", 
            'default': 1, 'type': 'int', 'min': 1},
            {'name': 'theta', 'label': "Rozdzielczość kąta (stopnie):", 
            'default': 1.0, 'type': 'float', 'min': 0.1},
            {'name': 'threshold', 'label': "Próg akumulatora:", 
            'default': 100, 'type': 'int', 'min': 1},
            {'name': 'min_line_length', 'label': "Min. długość linii (piksele):", 
            'default': 50, 'type': 'int', 'min': 1},
            {'name': 'max_line_gap', 'label': "Maks. przerwa w linii (piksele):", 
            'default': 10, 'type': 'int', 'min': 0}
        ]
        
        entries = {}
        
        for row, param in enumerate(params):
            Label(main_frame, text=param['label']).grid(
                row=row, column=0, padx=5, pady=5, sticky=E)
            
            entry = Entry(main_frame)
            entry.grid(row=row, column=1, padx=5, pady=5, sticky=EW)
            entry.insert(0, str(param['default']))
            entries[param['name']] = entry
        
        border_var = StringVar(value='default')
        border_row = len(params)
        
        Label(main_frame, text="Metoda obsługi brzegów:").grid(
            row=border_row, column=0, padx=5, pady=5, sticky=E)
        
        border_frame = Frame(main_frame)
        border_frame.grid(row=border_row, column=1, sticky=W)
        
        for btype in BORDER_TYPES:
            Radiobutton(
                border_frame, 
                text=btype, 
                variable=border_var, 
                value=btype
            ).pack(side=LEFT, padx=2)
        
        button_frame = Frame(dialog)
        button_frame.pack(pady=(0, 10))
        
        result = None
        
        def validate_and_accept():
            nonlocal result
            try:
                params_dict = {}
                for param in params:
                    value = entries[param['name']].get()
                    
                    if param['type'] == 'int':
                        value = int(value)
                    elif param['type'] == 'float':
                        value = float(value)
                    
                    if value < param['min']:
                        raise ValueError(f"{param['label']} musi być ≥ {param['min']}")
                    
                    params_dict[param['name']] = value
                
                params_dict['theta'] *= np.pi/180
                
                params_dict['border_type'] = BORDER_TYPES[border_var.get()]
                result = params_dict
                dialog.destroy()
                
            except ValueError as e:
                self.show_error("Błąd parametrów", str(e))
                return
        
        Button(button_frame, text="OK", command=validate_and_accept, width=10).pack(
            side=LEFT, padx=5)
        Button(button_frame, text="Anuluj", command=dialog.destroy, width=10).pack(
            side=LEFT, padx=5)
        
        self._center_dialog(dialog)
        dialog.grab_set()
        dialog.wait_window()
        
        return result
    
    def morphology_dialog(self):
        dialog = self._create_centered_dialog("Parametry morfologii")
        
        
        kernel_var = StringVar(value='rhombus')
        kernel_map = {
            'rhombus': 'Romb',
            'square': 'Kwadrat'
        }
        Label(dialog, text="Element strukturalny:").pack(pady=5)
        for kernel, name in kernel_map.items():
            Radiobutton(dialog, text=name, variable=kernel_var, value=kernel).pack(anchor=W)
        
        Label(dialog, text="Rozmiar elementu (nieparzysty):").pack(pady=5)
        size_entry = Entry(dialog)
        size_entry.pack()
        size_entry.insert(0, "3")
        
        border_var = StringVar(value='constant')
        Label(dialog, text="Obsługa brzegu:").pack(pady=5)
        borders = {
            'constant': 'Stała wartość (0)',
            'replicate': 'Replikacja',
            'reflect': 'Odbicie',
            'reflect_101': 'Odbicie (bez krawędzi)',
            'wrap': 'Zawinięcie'
        }
        for code, name in borders.items():
            Radiobutton(dialog, text=name, variable=border_var, value=code).pack(anchor=W)
        
        result = None
        
        def on_ok():
            nonlocal result
            try:
                size = int(size_entry.get())
                if size <= 0 or size % 2 == 0:
                    raise ValueError("Rozmiar musi być dodatnią liczbą nieparzystą")
                
                result = {
                    'kernel_type': kernel_var.get(),
                    'kernel_size': size,
                    'border_type': getattr(cv2, f'BORDER_{border_var.get().upper()}'),
                    'border_value': 0
                }
                dialog.destroy()
            except ValueError as e:
                messagebox.showerror("Błąd", str(e))
        
        self._add_ok_cancel_buttons(dialog, on_ok)
        self._center_dialog(dialog)
        dialog.grab_set()
        dialog.wait_window()
        
        return result
    
    def select_second_image_dialog(self):
        if len(self.windows) < 2:
            messagebox.showinfo("Brak okien", "Nie ma innych otwartych okien z obrazami.")
            return None
        
        dialog = self._create_centered_dialog("Wybierz drugi obraz")
        
        Label(dialog, text="Wybierz obraz do operacji:").pack(pady=10)
        
        window_titles = [w.title() for w in self.windows.keys() 
                if w != self.context.root and self.windows[w].image is not None]
        
        if not window_titles:
            messagebox.showinfo("Brak obrazów", "Nie ma innych obrazów do wyboru.")
            dialog.destroy()
            return None
        
        selected_window = StringVar(value=window_titles[0])
        
        for title in window_titles:
            Radiobutton(
                dialog,
                text=title,
                variable=title,
                value=title
            ).pack(anchor=W, padx=10)
        
        operation_var = StringVar(value="add")
        Label(dialog, text="Wybierz operację:").pack(pady=5)
        
        operations = [
            ("Dodawanie", "add_images"),
            ("Odejmowanie", "subtract_images"),
            ("Mieszanie (blend)", "blend"),
            ("Bitowe AND", "AND"),
            ("Bitowe NOT", "NOT"),
            ("Bitowe OR", "OR"),
            ("Bitowe XOR", "XOR")
        ]
        
        for text, mode in operations:
            Radiobutton(
                dialog,
                text=text,
                variable=operation_var,
                value=mode
            ).pack(anchor=W, padx=10)
        
        # For blend operation we need alpha parameter
        alpha_frame = Frame(dialog)
        alpha_frame.pack(pady=5)
        Label(alpha_frame, text="Alpha <dla operacji blend> (0-1):").pack(side=LEFT)
        alpha_entry = Entry(alpha_frame, width=5)
        alpha_entry.pack(side=LEFT)
        alpha_entry.insert(0, "0.5")
        
        result = None
        
        def on_ok():
            nonlocal result
            try:
                alpha = float(alpha_entry.get())
                if not 0 <= alpha <= 1:
                    raise ValueError("Alpha musi być w zakresie 0-1")
                
                result = (selected_window.get(), operation_var.get(), alpha)
                dialog.destroy()
            except ValueError as e:
                messagebox.showerror("Błąd", str(e))
       
        self._center_dialog(dialog)
        self._add_ok_cancel_buttons(dialog, on_ok)
        dialog.grab_set()
        dialog.wait_window()
        
        return result

    def ask_border_dialog(self, borders=BORDER_TYPES):
        dialog = self._create_centered_dialog("Wybierz typ obramowania")
        
        selected_border = StringVar(value="default")
        Label(dialog, text="Wybor obramowania").pack(pady=5)
        
        for name in borders:
            Radiobutton(
                dialog,
                text=name,
                variable=selected_border,
                value=name
            ).pack(anchor=W, padx=10)
        
        confirmed = False
        
        def on_ok():
            nonlocal confirmed
            confirmed = True
            dialog.destroy()
        
        self._add_ok_cancel_buttons(dialog, on_ok)
        self._center_dialog(dialog)
        dialog.grab_set()
        dialog.wait_window()
        
        return selected_border.get() if confirmed else None

    def custom_mask_dialog(self, title = "Własna maska", default_values=None, mask_size=3):
        dialog = self._create_centered_dialog(title=title)
        
        default_values = default_values or [[0]*mask_size for _ in range(mask_size)]
        entries = []
        
        for i in range(mask_size):
            for j in range(mask_size):
                entry = Entry(dialog, width=5, justify='center')
                entry.grid(row=i, column=j, padx=2, pady=2)
                entry.insert(0, str(default_values[i][j]))
                entries.append(entry)
        
        result = None
        
        def on_ok():
            nonlocal result
            try:
                mask = []
                for i in range(mask_size):
                    row = []
                    for j in range(mask_size):
                        value = entries[i*mask_size + j].get()
                        if not value:
                            raise ValueError(f"Pusta wartość wiersz:{i+1}, kol:{j+1}")
                        row.append(float(value))
                    mask.append(row)
                
                result = np.array(mask, dtype=np.float32)
                dialog.destroy()
            except ValueError as e:
                messagebox.showerror("Błąd", f"Nieprawidłowa wartość: {str(e)}")
        
        btn_frame = Frame(dialog)
        btn_frame.grid(row=mask_size, column=0, columnspan=mask_size, pady=5)
        Button(btn_frame, text="OK", command=on_ok).pack(side=LEFT, padx=5)
        Button(btn_frame, text="Anuluj", command=dialog.destroy).pack(side=LEFT, padx=5)
        
        self._center_dialog(dialog)
        dialog.grab_set()
        dialog.wait_window()
        
        return result
    
    def median_filter_dialog(self):
        dialog = self._create_centered_dialog("Filtracja medianowa")
        
        # Kernel size selection
        Label(dialog, text="Rozmiar Kernela:").pack(pady=5)
        kernel_size = IntVar(value=3)
        
        sizes_frame = Frame(dialog)
        sizes_frame.pack()
        for size in [3, 5, 7]:
            Radiobutton(
                sizes_frame,
                text=f"{size}x{size}",
                variable=kernel_size,
                value=size
            ).pack(side=LEFT, padx=5)
        
        Label(dialog, text="Border Handling:").pack(pady=5)
        border_type = StringVar(value="reflect")
        
        border_frame = Frame(dialog)
        border_frame.pack()
        for btype in BORDER_TYPES:
            Radiobutton(
                border_frame,
                text=btype,
                variable=border_type,
                value=btype
            ).pack(anchor=W, padx=10)
        
        btn_frame = Frame(dialog)
        btn_frame.pack(pady=10)
        
        result = None
        
        def on_ok():
            nonlocal result
            result = (kernel_size.get(), border_type.get())
            dialog.destroy()
        
        Button(btn_frame, text="OK", command=on_ok).pack(side=LEFT, padx=5)
        Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=LEFT, padx=5)
        
        # Center dialog
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'+{x}+{y}')
        
        dialog.grab_set()
        dialog.wait_window()
        
        return result
    # -------------- END OF USER INTERACTIONS -------------
    
    # -------------- IMAGE MANAGEMENT ---------------------
    def load_image(self, path):
        ctx = self.windows.get(self.root)
        ctx.image = self.processor.load_image(path)
        ctx.og_image = ctx.image.copy()
        ctx.label.bind("<Button-1>", lambda event, root=ctx.root: self.set_current_image(root))
        self.context.adjust_window_size()
        self.context.display_current_image()
        
    def dispatch_click_handler(self, event, root):
        if self.profile_line_active:
            self.handle_profile_line_click(event)
        else:
            self.set_current_image(root)

    def set_current_image(self, root):
        if self.profile_line_active:
            return
        context = self.windows.get(root)
        if context:
            self.context = context
            context.display_current_image()

    def reset_image(self):
        self.context.image = self.context.og_image.copy()
        self.context.display_current_image()
    # -------------- END OF IMAGE MANAGEMENT ---------------------

    # -------------- IMAGE OPERATIONS ---------------------
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
        # new_window.bind("<Configure>", lambda e, ctx=context: ctx.on_window_resize(e))
        new_window.bind("<Button-1>", lambda event, root=new_window: self.dispatch_click_handler(event, root))

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
                self.show_message("Sukces", f"Histogram zapisany w {file_path}")
        

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
        old_img = self.context.image.copy()
        self.context.image = processed
        self.context.display_current_image()
        self.processor.compare_histograms(old_img, processed)

    @image_required
    def apply_stretch_histogram(self):
        if not self.processor.is_grayscale(self.context.image):
            self.show_error("Błąd", "Rozciąganie histogramu działa tylko na obrazach w skali szarości.")
            return
        processed = self.processor.stretch_histogram(self.context.image)
        old_img = self.context.image.copy()
        self.context.image = processed
        self.context.display_current_image()
        self.processor.compare_histograms(old_img, processed)

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

    @image_required
    def dispatch_neighborhood(self, mode='laplacian', ask_border=False):
        
        if not self.processor.is_grayscale(self.context.image):
            self.show_error("Błąd", "Operacje sąsiedztwa działają tylko na obrazach w odcieniach szarości.")
            return
        
        try:
            if ask_border:
                selected_border = self.ask_border_dialog()
                if selected_border is None:
                    return
                processed_img = getattr(self.processor, mode)(
                    self.context.image,
                    border_type=selected_border
                )
            else:
                processed_img = getattr(self.processor, mode)(self.context.image)
            
            self.context.image = processed_img
            self.context.display_current_image()
            
        except Exception as e:
            self.show_error("Błąd", f"Błąd operacji {mode}: {str(e)}")

    @image_required
    def apply_sharpen_laplacian(self):
        try:
            selected_border = self.ask_border_dialog()
            if selected_border is None:
                return
            results = self.processor.sharpen_linear_laplacian(self.context.image, border_type=selected_border)
            if not results:
                self.show_error("Błąd", "Brak rezultatu z processingu.")
                return
                
            for i, img in enumerate(results):
                self.show_new_window(f"Laplacian mask {i+1}", 
                                    self.processor.convert_to_tkimage(img), 
                                    img)
        except Exception as e:
            self.show_error("Błąd", f"Błąd przy wyostrzaniu Laplacianem: {str(e)}")
            
    @image_required
    def apply_prewitt(self):
        results = self.processor.direct_edge_detection(self.context.image)
        for direction, img in results.items():
            self.show_new_window(f"Prewitt - {direction}", self.processor.convert_to_tkimage(img), img)

    @image_required
    def apply_custom_mask(self):
        mask = self.custom_mask_dialog()
        if mask is None:
            return
        result = self.processor.custom_mask(self.context.image, mask)
        self.show_new_window("Własna maska 3x3", self.processor.convert_to_tkimage(result), result)

    @image_required
    def apply_median_filter(self):
        params = self.median_filter_dialog()
        if not params:
            return
        
        kernel_size, border_type = params
        
        try:
            filtered = self.processor.median_filter(
                self.context.image,
                kernel_size=kernel_size,
                border_type=border_type
            )
            
            self.context.image = filtered
            self.context.display_current_image()
            
        except Exception as e:
            self.show_error("Error", f"Median filtering failed: {str(e)}")
    @image_required
    def perform_two_arg_operation(self):
        selection = self.select_second_image_dialog()
        if not selection:
            return
        
        selected_window_ref, operation, alpha = selection
        second_context = self.find_window_by_title(selected_window_ref)
        
        if second_context is None or second_context.image is None:
            self.show_error("Błąd", "Wybrany obraz jest pusty lub okno zostało zamknięte.")
            return
        try:
            image1 = self.context.image
            image2 = second_context.image
            print(image1.shape, image2.shape)
            
            if image1.shape != image2.shape:
                self.show_error("Błąd", "Obrazy muszą mieć te same wymiary.")
                return
            
            args = {'image1': image1, 'image2': image2}
            if operation == "blend":
                args.update({'alpha': alpha})
            elif operation == "NOT":
                args.pop('image2')
            processed_img = getattr(self.processor, operation)(**args)
            
            if processed_img is not None:
                self.context.image = processed_img
                self.context.display_current_image()
                
        except Exception as e:
            self.show_error("Błąd", f"Operacja nie powiodła się: {str(e)}")
            
    @image_required
    def compare_filtering_methods(self):
        if not self.processor.is_grayscale(self.context.image):
            self.show_error("Błąd", "Operacja wymaga obrazu w odcieniach szarości.")
            return

        smooth_kernel = np.array(self.custom_mask_dialog(title="Maska Wygładzania",default_values=[[0, 0, 0], [0, 0, 0], [0, 0, 0]], mask_size=3), dtype=np.float32)
        
        sharpen_kernel = np.array(self.custom_mask_dialog(title="Maska Wyostrzania", default_values=[[0, 0, 0], [0, 1, 0], [0, 0, 0]], mask_size=3), dtype=np.float32)

        combined_kernel = self.processor.combine_kernels(smooth_kernel, sharpen_kernel)

        border_type = self.ask_border_dialog()
        if border_type is None:
            return

        smoothed, two_stage_result = self.processor.two_stage_filter(
            self.context.image, smooth_kernel, sharpen_kernel, border_type)
        
        single_stage_result = self.processor.custom_mask(
            self.context.image, combined_kernel, border_type)

        self.show_new_window("Faza 1: Wygładzanie", 
                           self.processor.convert_to_tkimage(smoothed), 
                           smoothed)
        
        self.show_new_window("Faza 2: Wyostrzanie (dwuetapowy)", 
                           self.processor.convert_to_tkimage(two_stage_result), 
                           two_stage_result)
        
        self.show_new_window("Filtracja jednoetapowa (5x5)", 
                           self.processor.convert_to_tkimage(single_stage_result), 
                           single_stage_result)

        difference = self.processor.img_diff(single_stage_result, two_stage_result)
        self.show_new_window("Różnica między metodami", 
                           self.processor.convert_to_tkimage(difference), 
                           difference)

        diff_mean = np.mean(difference)
        diff_max = np.max(difference)
        diff_std = np.std(difference)
        
        stats_msg = (
            f"Statystyki różnic:\n"
            f"Średnia: {diff_mean:.2f}\n"
            f"Maksimum: {diff_max}\n"
            f"Odchylenie standardowe: {diff_std:.2f}"
        )
        
        self.show_message("Statystyki porównania", stats_msg)
    
    @image_required
    def dispatch_morphology(self, operation):
        params = self.morphology_dialog()
        if not params:
            return
        try:
            if not self.processor.is_binary(self.context.image):
                self.show_error("Błąd", "Operacje morfologiczne działają tylko na obrazach binarnych.")
                return
            result = self.processor.morph_operation(
                self.context.image,
                operation=operation,
                kernel_type=params['kernel_type'],
                kernel_size=params['kernel_size'],
                border_type=params['border_type'],
                border_value=params['border_value']
            )
            
            self.context.image = result
            self.context.display_current_image()
            
        except Exception as e:
            self.show_error("Błąd", f"Operacja morfologiczna nie powiodła się: {str(e)}")
            
    @image_required
    def apply_binarization(self):
        image = self.context.image
        if not self.processor.is_grayscale(image):
            image = self.processor.to_grayscale(image)
        threshold = self.ask_for_input_int("Binarizacja", "Podaj próg (0-255):")
        
        if threshold is None:
            return
        binarized_image = self.processor.binarize(image, threshold)
        self.context.image = binarized_image
        self.context.display_current_image()
        
    @image_required
    def apply_skeletonization(self):
        try:
            if not self.processor.is_binary(self.context.image):
                self.show_error("Błąd", "Szkieletyzacja działa tylko na obrazach binarnych.")
                return
            binary_img = self.processor.binarize(self.context.image)
            
            border_type = self.ask_border_dialog()
            if border_type is None:
                return
            
            max_iters = self.ask_for_input_int("Szkieletyzacja", "Podaj maksymalną ilość iteracji):")
                
            skeleton = self.processor.skeletonize(
                binary_img,
                border_type=BORDER_TYPES[border_type],
                max_iterations=max_iters
            )
            
            self.context.image = skeleton
            self.context.display_current_image()
            
        except Exception as e:
            self.show_error("Błąd", f"Szkieletyzacja nie powiodła się: {str(e)}")

    @image_required
    def detect_lines_hough(self):
        try:
            if not self.processor.is_grayscale(self.context.image):
                self.show_error("Błąd", "Detekcja linii wymaga obrazu w odcieniach szarości.")
                return
            
            params = self.hough_params_dialog()
            if not params:
                return
                
            result = self.processor.detect_lines(
                self.context.image,
                rho=params['rho'],
                theta=params['theta'],
                threshold=params['threshold'],
                min_line_length=params['min_line_length'],
                max_line_gap=params['max_line_gap'],
                border_type=params['border_type']
            )
            
            self.show_new_window("Detekcja linii", 
                               self.processor.convert_to_tkimage(result), 
                               result)
            
        except Exception as e:
            self.show_error("Błąd", f"Detekcja linii nie powiodła się: {str(e)}")
            
    @image_required
    def start_profile_line(self):
        self.profile_points = []
        self.profile_line_active = True
        self.show_message("Linia profilu", "Naciśnij dwa punkty na obrazie, aby utworzyć linię profilu.")

    def handle_profile_line_click(self, event):
        if not self.profile_line_active:
            return
        
        x = event.x - self.context.label.winfo_x()
        y = event.y - self.context.label.winfo_y()
        
        
        self.profile_points.append((x, y))
        
        if len(self.profile_points) == 2:
            self.show_profile_line()
            self.profile_line_active = False

    def show_profile_line(self):
        img = self.context.image
        
        x1, y1 = self.profile_points[0]
        x2, y2 = self.profile_points[1]
        
        self.processor.show_profile_line(img,x1, y1, x2, y2)  

    @image_required
    def show_image_pyramid(self):
        self.pyramid, self.pyramid_scales = self.processor.build_image_pyramid(self.context.image, levels_up=4, levels_down=4)
        self.current_pyramid_level = len(self.pyramid_scales) // 2
        self.show_message("Piramida Obrazow", "Użyj kółka myszy do przewijania poziomów piramidy.")
        
        self.pyramid_window = tk.Toplevel(self.root)
        self.pyramid_window.title("Piramida Obrazow")
        
        
        self.pyramid_window.resizable(False, False)
        
        self.pyramid_canvas = tk.Canvas(self.pyramid_window)
        self.pyramid_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.update_pyramid_display()
        
        self.pyramid_canvas.bind("<MouseWheel>", self.on_pyramid_scroll)
        self.pyramid_canvas.bind("<Button-4>", self.on_pyramid_scroll)  
        self.pyramid_canvas.bind("<Button-5>", self.on_pyramid_scroll)  
        
        self.pyramid_canvas.focus_set()

    def on_pyramid_scroll(self, event):
        if event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            if self.current_pyramid_level < len(self.pyramid) - 1:
                self.current_pyramid_level += 1
        elif event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            if self.current_pyramid_level > 0:
                self.current_pyramid_level -= 1
        
        self.update_pyramid_display()

    def update_pyramid_display(self):
        current_img = self.pyramid[self.current_pyramid_level]
        
        tk_img = self.processor.convert_to_tkimage(current_img)
        
        self.pyramid_canvas.delete("all")
        self.pyramid_canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        
        self.pyramid_canvas.config(scrollregion=(0, 0, current_img.shape[1], current_img.shape[0]))
        
        self.pyramid_canvas.tk_img = tk_img
        
        self.pyramid_window.title(f"Rozmiar: {self.pyramid_scales[self.current_pyramid_level]}x")
        self.pyramid_window.geometry(f"{current_img.shape[1]}x{current_img.shape[0]}")

        

    @image_required
    def apply_otsu_threshold(self):
        segmented = self.processor.otsu_threshold(self.context.image)
        self.apply_threshold_result(segmented)
    
    def apply_threshold_result(self, segmented_image):
        self.context.image = segmented_image
        self.context.display_current_image()
    
    @image_required
    def manual_threshold(self):
        dialog = Toplevel(self.root)
        dialog.title("Ręczne progowanie")
        dialog.resizable(False, False)
        self._center_dialog(dialog)
        
        current_threshold = IntVar(value=127)
        
        def update_preview(_=None):
            threshold = current_threshold.get()
            segmented = self.processor.manual_threshold(self.context.image, threshold)
            preview_img = self.processor.convert_to_tkimage(segmented)
            preview_label.config(image=preview_img)
            preview_label.image = preview_img
        
        scale = Scale(dialog, from_=0, to=255, orient=HORIZONTAL,
                     variable=current_threshold, command=update_preview)
        scale.pack(fill=X, padx=10, pady=5)
        
        preview_label = Label(dialog)
        preview_label.pack()
        
        btn_frame = Frame(dialog)
        btn_frame.pack(pady=5)
        
        Button(btn_frame, text="Zastosuj", 
              command=lambda: self.apply_threshold_result(
                  self.processor.manual_threshold(
                      self.context.image, current_threshold.get()))
              ).pack(side=LEFT, padx=5)
        
        Button(btn_frame, text="Anuluj", 
              command=dialog.destroy).pack(side=LEFT, padx=5)
        
        update_preview()

    @image_required
    def adaptive_threshold(self):
        dialog = Toplevel(self.root)
        dialog.title("Progowanie adaptacyjne")
        dialog.resizable(False, False)
        self._center_dialog(dialog)
        
        block_size = IntVar(value=11)
        C = IntVar(value=2)
        
        def update_preview():
            bs = max(3, block_size.get() | 1) 
            segmented = self.processor.adaptive_threshold(
                self.context.image, bs, C.get())
            preview_img = self.processor.convert_to_tkimage(segmented)
            preview_label.config(image=preview_img)
            preview_label.image = preview_img
        
        Frame(dialog).pack(pady=5)
        
        Label(dialog, text="Rozmiar bloku:").pack()
        Scale(dialog, from_=3, to=51, orient=HORIZONTAL, 
             variable=block_size, command=lambda _: update_preview()).pack()
        
        Label(dialog, text="Stała C:").pack()
        Scale(dialog, from_=0, to=20, orient=HORIZONTAL,
             variable=C, command=lambda _: update_preview()).pack()
        
        preview_label = Label(dialog)
        preview_label.pack(pady=10)
        
        btn_frame = Frame(dialog)
        btn_frame.pack(pady=5)
        
        Button(btn_frame, text="Zastosuj",
              command=lambda: self.apply_threshold_result(
                  self.processor.adaptive_threshold(
                      self.context.image, 
                      max(3, block_size.get() | 1), 
                      C.get()))
              ).pack(side=LEFT, padx=5)
        
        Button(btn_frame, text="Anuluj", 
              command=dialog.destroy).pack(side=LEFT, padx=5)
        
        update_preview()
        
    @image_required
    def apply_grabcut(self):
        self.context.label.bind("<ButtonPress-1>", self.on_grabcut_press)
        self.context.label.bind("<B1-Motion>", self.on_grabcut_motion)
        self.context.label.bind("<ButtonRelease-1>", self.on_grabcut_release)
        
        self.context.grabcut_start = None
        self.context.grabcut_end = None

    def on_grabcut_press(self, event):
        self.context.grabcut_start = (event.x, event.y)
        self.context.grabcut_end = (event.x, event.y)

    def on_grabcut_motion(self, event):
        self.context.grabcut_end = (event.x, event.y)
        self.update_label_with_rectangle()

    def on_grabcut_release(self, event):

        self.context.grabcut_end = (event.x, event.y)

        x0, y0 = self.context.grabcut_start
        x1, y1 = self.context.grabcut_end
        x, y = min(x0, x1), min(y0, y1)
        w, h = abs(x1 - x0), abs(y1 - y0)
        rect = (x, y, w, h)

        # unmount events
        self.context.label.unbind("<ButtonPress-1>")
        self.context.label.unbind("<B1-Motion>")
        self.context.label.unbind("<ButtonRelease-1>")

        # process grabcut
        processed = self.processor.grabcut_segmentation(self.context.image, rect=rect)
        self.context.image = processed
        self.context.display_current_image()
        
    def update_label_with_rectangle(self):
        image_copy = self.context.image.copy()

        if self.context.grabcut_start and self.context.grabcut_end:
            x0, y0 = self.context.grabcut_start
            x1, y1 = self.context.grabcut_end
            cv2.rectangle(image_copy, (x0, y0), (x1, y1), (0, 0, 255), 2)  # czerwony prostokąt

        # aktualizacja obrazu 
        image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(image_rgb)
        self.context.tk_image = ImageTk.PhotoImage(im_pil)
        self.context.label.config(image=self.context.tk_image)

    @image_required
    def apply_watershed(self):
        processed = self.processor.watershed_segmentation(self.context.image)
        self.show_new_window("Watershed", 
                       self.processor.convert_to_tkimage(processed),
                       processed)
    
    @image_required
    def apply_inpainting(self):
        try:
            processed = self.processor.inpaint(self.context.image)
        except Exception as e:
            self.show_error("Błąd", f"Operacja nie powiodła się: {str(e)}")
            return
        self.show_new_window("Inpainting",
                       self.processor.convert_to_tkimage(processed),
                       processed)
        
    @image_required
    def apply_rle_compression(self):
        try:
            img = self.context.image
            
            compressed_data = self.processor.rle_compress(img)
            
            # Save to temporary file
            temp_path = "temp_compressed.rle"
            self.processor.save_rle_compressed(compressed_data, temp_path)
            
            # Calculate stats - THIS IS WHERE SK IS CALCULATED
            stats = self.processor.calculate_compression_stats(img, temp_path)
            
            # Show results with compression level (SK)
            result_msg = (
                f"Original Size: {stats['original_size']:,} bytes\n"
                f"Compressed Size: {stats['compressed_size']:,} bytes\n"
                f"Compression Level (SK): {stats['compression_level']:.2f}\n"
                f"Space Saved: {stats['space_saving']:.1%}"
            )
            messagebox.showinfo("RLE Compression Results", result_msg)
            
            # Clean up
            os.remove(temp_path)
            
        except Exception as e:
            messagebox.showerror("Error", f"RLE Compression failed: {str(e)}")
        
    # -------------- END OF IMAGE OPERATIONS ---------------------

    def run(self):
        self.root.mainloop()

        

# helper class to manage opened windows and images
class WindowContext:
    def __init__(self, root, label, image=None, frame=None, processor: ImageProcessor = None):
        self.root: Union[Tk, Toplevel] = root
        self.label: Label = label
        self.image: Union[np.ndarray, None] = image
        self.og_image: Union[np.ndarray, None] = image.copy() if image is not None else None
        self.img_frame: Frame = frame
        self.tk_image: Union[ImageTk.PhotoImage, None] = None
        self.processor = processor
        
        self.root.geometry("400x400")  
        
        self.grabcut_start = None
        self.grabcut_end = None
        
        if image is not None:
            self.adjust_window_size()
            self.display_current_image()
        
    @staticmethod
    def find_window_by_title(gui: GUI, title):
        for window in gui.windows:
            if window.title() == title:
                return window
        return None
    

    
    def display_current_image(self, width=None, height=None):
        if self.image is None:
            return
            
        tk_image = self.processor.convert_to_tkimage(self.image)
        
        self.tk_image = tk_image
        self.label.config(image=tk_image)
        self.label.image = tk_image
        
        self.adjust_window_size()
    
    def adjust_window_size(self):
        if self.image is None:
            return
            
        h, w = self.image.shape[:2]
        
        window_width = w + (self.root.winfo_width() - self.label.winfo_width())
        window_height = h + (self.root.winfo_height() - self.label.winfo_height())
        
        self.root.geometry(f"{window_width}x{window_height}")
        
        self.root.minsize(window_width, window_height)



if __name__ == "__main__":
    app = GUI()
    app.run()