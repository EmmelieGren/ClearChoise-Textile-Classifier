import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
from functions import load_and_preprocess_image, predict_pattern_keras


def open_image():
    filepath = filedialog.askopenfilename(title="Select Image File", filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("All files", "*.*")))
    if filepath:
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        img = Image.fromarray(img)
        img.thumbnail((300, 300)) 
        img = ImageTk.PhotoImage(img)
        label.config(image=img)
        label.image = img
        
        img_preprocessed = load_and_preprocess_image(filepath)
        prediction = predict_pattern_keras(img_preprocessed)
        
        result_label.config(text=f"Predicted pattern: {prediction}")

# Main window
root = tk.Tk()
root.title("Pattern Detection")

# Label to display the image
label = tk.Label(root)
label.pack(padx=10, pady=10)

# Label
result_label = tk.Label(root, text="")
result_label.pack(padx=10, pady=5)

# Button
button = tk.Button(root, text="Open Image", command=open_image)
button.pack(padx=10, pady=5)


root.mainloop()

