from functools import wraps
from tkinter import messagebox

def image_required(func):
    """Decorator to ensure an image is loaded before calling the function."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.context.image is None:
            
            messagebox.showinfo("No Image selected or loaded", "Please load an image first!")
            self.open_file_dialog()
                
            if self.context.image is not None:
                return func(self, *args, **kwargs)  
            else:
                messagebox.showwarning("No Image Selected", "No image was selected.")
        else:
            return func(self, *args, **kwargs)  
    return wrapper