import os
import os.path
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image


dir_path = r'/Users/ejzhang/Downloads/HealthLabResearch/improved-diffusion/data'
for file in os.listdir(dir_path):
    save_path = f'resized_128/{file}'
    f_img = dir_path + "/" + file
    img = Image.open(f_img)
    resize = transforms.Resize((128, 128)) (img)
    resize.save(save_path)