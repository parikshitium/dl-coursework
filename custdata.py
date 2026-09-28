import torch
from torch import nn
from pathlib import Path
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = Path('data/')
image_path = data_path / 'pizza_steak_sushi'
train_dir = data_path / "train"
test_dir = data_path / "test"

def walk_through(dir_path):
    for dirpath,dirnames,filenames in os.walk(dir_path):
        print(f'There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}')
#random.seed(42)
image_path_list = list(data_path.glob('*/*/*.jpg'))
random_image_path = random.choice(image_path_list)
image_class = random_image_path.parent.stem
image = Image.open(random_image_path)
image_array = np.asarray(image)
'''plt.figure(figsize=(10,7))
plt.imshow(image_array)
plt.title(f'This is a {image_class}, Image dimensions: {image_array.shape}')
plt.axis(False)
plt.show()'''

data_transform = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

def plot_transformed_images(image_paths,transform,n=3,seed=None):
    if seed:
        random.seed(seed)
    random_image_paths = random.sample(image_paths,k=n)

    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(nrows=1,ncols=2)
            ax[0].imshow(f)
            ax[0].set_title(f'Original\nSize: {f.size}')
            ax[0].axis(False)

            transformed_image = transform(f).permute(1,2,0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f'Transformed\nShape: {transformed_image.shape}')
            ax[1].axis(False)

            fig.suptitle(f'Class: {image_path.parent.stem}',fontsize=16)
            plt.show()
'''plot_transformed_images(image_path_list,
                        data_transform,
                        n=3)'''

train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform,
                                  target_transform=None)
test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)
class_names = train_data.classes
class_dict = train_data.class_to_idx
'''img, label = train_data[0][0],train_data[0][1]
img = img.permute(1,2,0)
plt.figure(figsize=(10,7))
plt.imshow(img)
plt.title(class_names[label],fontsize=14)
plt.axis(False)
plt.show()'''

train_dtload = DataLoader(dataset=train_data,
                          shuffle=True,
                          num_workers=0,
                          batch_size=1)
test_dtload = DataLoader(dataset=test_data,
                         shuffle=False,
                         num_workers=os.cpu_count(),
                         batch_size=1)

img, label = next(iter(train_dtload))