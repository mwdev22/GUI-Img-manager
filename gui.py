from tkinter import Tk, Button, filedialog
import cv2
from PIL import Image, ImageTk
import tkinter as tk


class GUI:
    def __init__(self):
        self.root = Tk()
        self.root.title("OBRAZY")
        self.root.geometry("500x400")

        self.mount_button("Load Image", self.open_file_dialog, {"x": 180, "y": 20})

        

        self.image_label = tk.Label(self.root)
        self.image_label.place(x=50, y=60)

    def open_file_dialog(self):
        file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.gif"))]
         )
        
        if file_path:
            print(file_path)
            image = self.load_image(file_path)
            self.display_image(image)

    def load_image(self, path, grayscale=False):
        mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        image = cv2.imread(path, mode)
        if image is None:
            raise ValueError("Failed to load image")
        return image

    def display_image(self, cv2_image):
        
        image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
      
        img_pil = Image.fromarray(image_rgb)
        img_pil = img_pil.resize((300, 200))  
        img_tk = ImageTk.PhotoImage(img_pil)
        
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk

    def mount_button(self, text, command, position):
        button = Button(self.root, text=text, command=command)
        button.place(x=position['x'], y=position['y'])

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = GUI()
    app.run()
