# contains functions for image processing

import torch
from torch import nn

from torchvision import transforms

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

color_projection = torch.tensor([[0.26, 0.09, 0.02],
                                 [0.27, 0.00, -0.05],
                                 [0.27, -0.09, 0.03]])

color_mean = torch.tensor([0.485, 0.456, 0.406])
color_std = torch.tensor([0.229, 0.224, 0.225])

BandW_intensity = torch.tensor([0.299, 0.587, 0.114])

def load_image(image_path, size=None):
    image = Image.open(image_path).convert('RGB')
    if size is not None:
        image = transforms.Resize(size)(image)
    
    image = np.array(image)/255
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()

    return image

def save_image(image, path):
    image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image = (image*255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(path, quality=95)

def to_pil_image(tensor):
    if isinstance(tensor, torch.Tensor):
        image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image = (image*255).astype(np.uint8)
        image = Image.fromarray(image)
    else:
        image = tensor
    return image

def imshow(image, title=None):
    if isinstance(image, list):
        size = int((np.sqrt(len(image)) * 6))
        fig, ax = plt.subplots(1, len(image), figsize=(size, size), frameon=False)
        if title is not None:
            ax[0].set_title(title)

        for i, im in enumerate(image):
            im = to_pil_image(im)
            ax[i].imshow(im, cmap='gray' if im.mode == 'L' else None)
            ax[i].axis('off')
    else:
        image = to_pil_image(image)
        
        plt.figure(frameon=False)
        plt.axis('off')

        plt.imshow(image, cmap='gray' if image.mode == 'L' else None)

        if title is not None:
            plt.title(title)

    plt.show()

def BlackAndWhite(image):
    if image.shape[1] == 1:
        return image
    
    image = image.squeeze(0).permute(1, 2, 0)
    image = image @ BandW_intensity
    image = image.unsqueeze(0).unsqueeze(0)

    return image

def color_transfer(image, style):
    style_mean = style.mean(dim=(2, 3), keepdim=True)
    style_std = style.std(dim=(2, 3), keepdim=True)

    return transforms.Normalize(mean=style_mean, std=style_std)(image)

def preprocess(image):
    return transforms.Normalize(mean=color_mean, std=color_std)(image)

########
# Now we define functions for image processing
########

########
# Kernel functions
########

def gaussian_kernel(n=5, sigma=1):
    kernel = torch.zeros((1, 1, n, n))

    for i in range(n):
        for j in range(n):
            kernel[0, 0, i, j] = -((i-n//2)**2 + (j-n//2)**2)/(2*sigma**2)
    kernel = torch.exp(kernel)
    
    return kernel/kernel.sum()

def edge_kernel():
    kernel = gaussian_kernel(5)
    kernel[0, 0, 2, 2] = 0
    m = kernel.sum()
    kernel = -kernel
    kernel[0, 0, 2, 2] = m
    
    return kernel

def sharpen_kernel():
    kernel = edge_kernel()
    kernel[0, 0, 2, 2] = 1 + kernel[0, 0, 2, 2]
    
    return kernel

def distance_kernel(n = 5):
	kernel = torch.zeros((n, n))
	sqrtn = torch.sqrt(torch.tensor(n))
	for i in range(n):
		for j in range(n):
			kernel[i, j] = ((i - n//2)/sqrtn) ** 2 + ((j - n//2)/sqrtn) **2
	kernel = torch.sqrt(kernel).unsqueeze(0).unsqueeze(0)

	return kernel/kernel.sum()

def distance_inverse_kernel(n = 5, eps = 1):
    kernel = distance_kernel(n)
    
    kernel = 1/(kernel + eps) ** 2

    return kernel/kernel.sum()

########
# Convolution functions
########

def convolve(image, kernel):
    padded_image = transforms.Resize((image.shape[2] + kernel.shape[2] - 1, image.shape[3] + kernel.shape[3] - 1))(image)
    return nn.functional.conv2d(padded_image, kernel, padding=0)

def blur(image, size=5, sigma=1):
    kernel = gaussian_kernel(size, sigma)
    if image.shape[1] == 3:
        kernel = torch.cat((kernel, kernel, kernel), dim=1)
    return convolve(image, kernel)

def edge(image):
    image = blur(image, 7, 1.414)
    image = BlackAndWhite(image)
    image -= image.mean()
    kernel = edge_kernel()
    image = convolve(image, kernel)

    mean, std = image.mean(), image.std()
    image = (image - mean)/std
    image = image.abs()
    image = image.clamp(0, 1)**2

    return image

def sharpen(image):
    image = blur(image, 7, 1.414)
    kernel = sharpen_kernel()
    return convolve(image, kernel)

def distance_blur(image, n = 5, eps = 1):
    kernel = distance_inverse_kernel(n, eps)
    return convolve(image, kernel)

def smooth_edge_detector(image, max_distance = 21):
    image = blur(image, 7, 1.414)
    image = edge(image)
    image = blur(image, max_distance, sigma = max_distance/4)
    image /= image.max()

    return image

########
# MISCELLANEOUS
########

def bw_np_to_torch(image):
    image = torch.tensor(image).float()
    image = image.unsqueeze(0).unsqueeze(0)
    return image

def to_map(image, lambda_):
    image = image.clone()
    mean, std = image.mean(), image.std()
    image[image < mean + lambda_*std] = 0
    image[image > 0] = 1
    return image