from functools import wraps
from tkinter import messagebox

def image_required(func):
    """Decorator to ensure an image is loaded before calling the function."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.current_image is None:
            messagebox.showinfo("No Image Loaded", "Please load an image first!")
            self.open_file_dialog()
            if self.current_image is not None:
                return func(self, *args, **kwargs)  
            else:
                messagebox.showwarning("No Image Selected", "No image was selected.")
        else:
            return func(self, *args, **kwargs)  
    return wrapper