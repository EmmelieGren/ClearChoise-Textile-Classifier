import cv2
import numpy as np
from colormath.color_objects import sRGBColor
from colormath.color_diff import delta_e_cie2000

# Define color names and their corresponding RGB values
color_names = {
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'red': (255, 0, 0),
    'green': (0, 128, 0),
    'blue': (0, 0, 255),
    'yellow': (255,255,204), #ljusgul
    'orange': (255,165,0),
    #'purple': (153,50,204)
}

# Function to calculate color difference
def color_difference(rgb1, rgb2):
    rmean = (rgb1[0] + rgb2[0]) / 2
    r = rgb1[0] - rgb2[0]
    g = rgb1[1] - rgb2[1]
    b = rgb1[2] - rgb2[2]
    return np.sqrt((2 + rmean / 256) * r ** 2 + 4 * g ** 2 + (2 + (255 - rmean) / 256) * b ** 2)

# Function to get the closest color name from RGB values
def get_color_name(rgb_tuple):
    closest_color_name = None
    min_distance = float('inf')
    for name, rgb in color_names.items():
        distance = color_difference(rgb_tuple, rgb)
        if distance < min_distance:
            min_distance = distance
            closest_color_name = name
    return closest_color_name

def output_color(img_paths):
    image = cv2.cvtColor(cv2.imread(img_paths), cv2.COLOR_BGR2RGB)
    unique_colors, counts = np.unique(image.reshape(-1, 3), axis=0, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_colors = unique_colors[sorted_indices]
    top_color_names = [get_color_name(tuple(color)) for color in sorted_colors[:1]]
    return top_color_names



image_path = 'b1.jpg'
color = output_color(image_path)
print (color)