import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import torch
import numpy as np
from functions import predict_pattern_for_images
import tensorflow as tf


def open_image():
    filepath = filedialog.askopenfilename(title="Select Image File", filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("All files", "*.*")))
    if filepath:
        try:
            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
            img = Image.fromarray(img)
            img.thumbnail((300, 300)) 
            img = ImageTk.PhotoImage(img)
            label.config(image=img)
            label.image = img
            
            # Perform pattern prediction
            predictions = predict_pattern_for_images([filepath])
            prediction_text = f"Predicted pattern: {predictions[0]}"
            result_label.config(text=prediction_text)
        except Exception as e:
            result_label.config(text=f"Error: {e}")





# Create the main window
root = tk.Tk()
root.title("Pattern Detection")

# Create a label to display the image
label = tk.Label(root)
label.pack(padx=10, pady=10)

# Create a label to display the predicted pattern
result_label = tk.Label(root, text="")
result_label.pack(padx=10, pady=5)

# Create a button to open the image file dialog
button = tk.Button(root, text="Open Image", command=open_image)
button.pack(padx=10, pady=5)

# Run the GUI application
root.mainloop()
