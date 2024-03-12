import cv2
import numpy as np
import torch
import tensorflow as tf

# Yolo problem
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

yolo_path = "C:/Users/emmel/Desktop/yolov5"
model_path = "./Flower_model/flower_model_28-02-24.pt"

yolo_model = torch.hub.load(yolo_path, 'custom', path=model_path, source='local')
yolo_model.conf = 0.30

patterns_modelpath = "./solid_color_models/pattern_model_24-03-06.h5" 
patterns_model = tf.keras.models.load_model(patterns_modelpath)

jeans_modelpath = "./solid_color_models/jeans_model_24-03-06.h5" 
jeans_model = tf.keras.models.load_model(jeans_modelpath)
labels_jeans = ["Not jeans", 
                "Jeans", 
]

geometric_modelpath = "./geometric_model/geometric_24-03-04.h5" 
geometric_model = tf.keras.models.load_model(geometric_modelpath)
labels_geometric = ["check", 
                    "stripe", 
                    "dots"
]

def load_and_preprocess_image(image_path):
    img_dim = (64,64)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_dim)
    img = img[:, :, np.newaxis]
    img = img / 255
    return np.expand_dims(img, axis=0)

def predict_pattern_keras(img_preprocessed):
    pred_pattern = patterns_model.predict(img_preprocessed)
    pred_class_pattern = np.argmax(pred_pattern)

    if pred_class_pattern == 1:  # solid color
        predictions_jeans = jeans_model.predict(img_preprocessed)
        predicted_class_jeans = np.argmax(predictions_jeans)
        return labels_jeans[predicted_class_jeans]
        
    else:
        pred_geometric = geometric_model.predict(img_preprocessed)
        pred_class_geometric = np.argmax(pred_geometric)
        return labels_geometric[pred_class_geometric]

# Function to predict pattern for images
def predict_pattern_for_images(img_paths):
    predictions = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        with torch.no_grad():
            results = yolo_model(img)  
            yolo_preds = results.pandas().xyxy[0]

        if len(yolo_preds) > 0:
            predictions.append("Flowers")
        else:
            img_preprocessed = load_and_preprocess_image(img_path)
            prediction = predict_pattern_keras(img_preprocessed)
            predictions.append(prediction)
    
    return predictions
