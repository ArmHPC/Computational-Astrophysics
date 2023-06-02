import random

import numpy as np
import torch
from PIL import Image

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as TF


def load_image(path):
    im = Image.open(path)
    im = np.array(im)
    im = (im - im.min())/(im.max() - im.min())
    if im.mean() > 0.5:
        im = 1 - im
    return im


class ZFill:
    def __init__(self, max_height=160, max_width=50):
        self.max_height = max_height
        self.max_width = max_width

    def __call__(self, image):
        s = image.shape

        d_width = (self.max_width - s[2])
        d_height = (self.max_height - s[1])

        d_left = int(d_width / 2)
        d_top = int(d_height / 2)
        d_right = int(d_width - d_left)
        d_bottom = int(d_height - d_top)

        image = TF.pad(img=image, padding=[d_left, d_top, d_right, d_bottom], fill=0)
        return image


class HeightShift:
    def __init__(self, fraction=0.15):
        assert 0 <= fraction <= 1
        self.fraction = fraction

    def __call__(self, image):
        width = image.shape[-1]
        height = image.shape[-2]

        direction = random.choice([-1, 1])
        height_fraction = int(self.fraction * height) + 1
        shift_size = random.choice(list(range(height_fraction)))

        if direction > 0:
            img = TF.crop(img=image, left=0, width=width, top=shift_size, height=height-shift_size)
        else:
            img = TF.crop(img=image, left=0, width=width, top=0, height=height-shift_size)

        return img


class WidthShift:
    def __init__(self, fraction=0.15):
        assert 0 <= fraction <= 1
        self.fraction = fraction

    def __call__(self, image):
        width = image.shape[-1]
        height = image.shape[-2]

        direction = random.choice([-1, 1])
        width_fraction = int(self.fraction * width) + 1
        shift_size = random.choice(list(range(width_fraction)))

        if direction > 0:
            img = TF.crop(img=image, left=0, width=width-shift_size, top=0, height=height)
        else:
            img = TF.crop(img=image, left=shift_size, width=width-shift_size, top=0, height=height)

        return img


def load_images(path, batch_size, domain, rank, world_size, pin_memory=False, num_workers=0):
    if domain == 'train':
        transform = T.Compose([
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            # T.RandomRotation(degrees=2),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.5),
            WidthShift(fraction=0.2),
            HeightShift(fraction=0.2),
            T.Resize(size=(140, 20)),
            ZFill(max_height=160, max_width=50),
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Resize(size=(140, 20)),
            ZFill(max_height=160, max_width=50),
        ])

    dataset = ImageFolder(path, transform=transform, loader=load_image)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler,
                             pin_memory=pin_memory, num_workers=num_workers,
                             drop_last=False, shuffle=False)

    class_indices = [[] for _ in range(len(dataset.classes))]
    for i in range(len(dataset)):
        _, label = dataset[i]
        class_indices[label].append(i)

    class_proportions = []
    for indices in class_indices:
        class_proportions.append(len(indices) / len(dataset))

    print(f'Loading {len(dataset)} images from {path} is over.')

    return data_loader, dataset.class_to_idx, class_proportions
