import random

import numpy as np
import torch
from PIL import Image

from torch.utils.data import DataLoader
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


def apply_patch_gaussian(img, patch_size=3, sigma=5):
    # Convert PIL Image to NumPy array
    img_np = np.array(img)

    # Randomly choose a patch position
    h, w, _ = img_np.shape
    top = np.random.randint(0, h - patch_size)
    left = np.random.randint(0, w - patch_size)
    bottom = top + patch_size
    right = left + patch_size

    # Create the patch
    patch = np.zeros((patch_size, patch_size, 1), dtype=np.uint8)

    # Apply Gaussian noise to the patch
    noise = np.random.normal(scale=sigma, size=(patch_size, patch_size, 3))
    patch = np.clip(patch + noise, 0, 255).astype(np.uint8)

    # Replace the patch in the original image
    img_np[top:bottom, left:right] = patch

    # Convert NumPy array back to PIL Image
    import matplotlib.pyplot as plt
    plt.imshow(img_np)
    plt.gray()
    plt.show()
    patched_img = Image.fromarray(img_np)

    return patched_img


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


def spoil_centered_object(img_tensor, spoil_percentage=5, max_spoil_intensity=0.5, p_sco=0.7):
    if random.random() > 1 - p_sco:
        _, h, w = img_tensor.size()
        spoil_size = int(min(h, w) * spoil_percentage / 100)

        # Calculate the coordinates for the centered patch
        top = (h - spoil_size) // 2
        left = (w - spoil_size) // 2
        bottom = top + spoil_size
        right = left + spoil_size

        # Generate a patch of random intensity to spoil the object
        patch = torch.rand_like(img_tensor[:, top:bottom, left:right]) * max_spoil_intensity

        # Apply the patch to the centered portion of the image
        img_tensor[:, top:bottom, left:right] = torch.clamp(
            img_tensor[:, top:bottom, left:right] - patch, 0, 1
        )

    return img_tensor


def load_images(path, batch_size, domain, _drop_last=True, _shuffle=True):
    if domain == 'train':
        transform = T.Compose([
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.RandomRotation(degrees=2),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.5),
            # T.Lambda(lambda x: spoil_centered_object(x, spoil_percentage=15, max_spoil_intensity=0.5)),
            # WidthShift(fraction=0.2),
            WidthShift(fraction=0.35),
            # HeightShift(fraction=0.2),
            HeightShift(fraction=0.35),
            T.Resize(size=(140, 20), antialias=True),
            ZFill(max_height=160, max_width=50),
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Resize(size=(140, 20), antialias=True),
            ZFill(max_height=160, max_width=50),
        ])

    dataset = ImageFolder(path, transform=transform, loader=load_image)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                             shuffle=_shuffle, pin_memory=True, drop_last=_drop_last)

    if domain == 'inference':
        return data_loader
    else:
        class_indices = [[] for _ in range(len(dataset.classes))]
        for i in range(len(dataset)):
            _, label = dataset[i]
            class_indices[label].append(i)

        class_proportions = []
        for indices in class_indices:
            class_proportions.append(len(indices) / len(dataset))

        print(f'Loading {len(dataset)} images from {path} is over.')

        return data_loader, dataset.class_to_idx, class_proportions

