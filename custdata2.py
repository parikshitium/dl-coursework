import torch
from torch import nn
from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from PIL import Image
from typing import Tuple, List, Dict
import random
import os
from torch.utils.data import Dataset
import pathlib
from custdata import train_data,test_data,class_names
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

data_path = Path('data/')
image_path = data_path / 'pizza_steak_sushi'
train_dir = data_path / 'train'
test_dir = data_path / 'test'

image_path_list = list(data_path.glob('*/*/*.jpg'))
random_image_path = random.choice(image_path_list)
image_class = random_image_path.parent.stem


def find_classes(directory: str) -> Tuple[List[str], Dict[str,int]]:
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    if not classes:
        raise FileNotFoundError(f'No classes in {directory}')
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx
class ImageFolderCustom(Dataset):
    def __init__(self,
                 target_dir: str,
                 transform: None):
        self.paths = list(pathlib.Path(target_dir).glob('*/*.jpg'))
        self.transforms = transform
        self.classes, self.class_to_idx = find_classes(target_dir)

    def load_image(self,index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)
    def __len__(self) -> int:
        return len(self.paths)
    def __getitem__(self,index: int) -> Tuple[torch.Tensor, int]:
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        if self.transforms:
            return self.transforms(img), class_idx
        else:
            return img, class_idx

train_transforms = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])

train_data_custom = ImageFolderCustom(target_dir=train_dir,
                                      transform=train_transforms)
test_data_custom = ImageFolderCustom(target_dir=test_dir,
                                      transform=test_transforms)


def display_random_images(dataset: torch.utils.data.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    if n > 10:
        n = 10
        display_shape = False
        print('Printing only 10 images')
    if seed:
        random.seed(seed)

    random_samples_idx = random.sample(range(len(dataset)), k=n)

    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]
        targ_image = targ_image.permute(1,2,0)
        plt.subplot(1,n,i+1)
        plt.imshow(targ_image)
        plt.axis(False)
        if classes:
            title = f'Class: {classes[targ_label]}'
            if display_shape:
                title = title + f'\nShape: {targ_image.shape}'
        plt.title(title)
    plt.tight_layout()
    plt.show()

train_dtload_custom = DataLoader(dataset=train_data_custom,
                                 shuffle=True,
                                 num_workers=0,
                                 batch_size=32)
test_dtload_custom = DataLoader(dataset=test_data_custom,
                                shuffle=False,
                                num_workers=os.cpu_count(),
                                batch_size=32)

img_custom, label_custom = next(iter(train_dtload_custom))
print(img_custom.shape)
print(label_custom.shape)