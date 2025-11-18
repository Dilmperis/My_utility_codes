# 3 Methods to upload images in a script from a folder:

# Pillow (PIL)
import os
from PIL import Image
import  numpy as np

path_folder = "path/to/folder"
dataset = []

for file in os.listdir(path_folder):
  if file.endswith('.png'):
    image_path = os.path.join(path_folder, file)
    img = Image.open(image_path)
    img_as_array = np.array(img)
    dataset.append(img_as_array)

dataset = np.array(dataset)
print(dataset.shape)
