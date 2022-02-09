import os
import cv2
import torch
import random
import numpy as np
from PIL import Image

def equal(im, flg_max=1, flg_min=0):
    mi = im.min()
    mx = im.max()
    imax = flg_max * mx + (1 - flg_max) * mi
    imin = flg_min * mx + (1 - flg_min) * mi

    im[im > imax] = imax
    im[im < imin] = imin
    return (im - np.min(im))/(np.max(im) - np.min(im))


def gauss_cal(img, flag='blur', kernel=3):
    im = img.astype(np.float64)
    if flag == 'gaussian_blur':
        blurred = cv2.GaussianBlur(im, (kernel, kernel), 10)
    elif flag == 'blur':
        blurred = cv2.blur(im, (kernel, kernel))
    im = im - blurred
    im[im < 0] = 0
    return im.astype(np.uint16)


def read_tiff_files(root):
    images = []
    for i in sorted(os.listdir(root)):
        img = Image.open(os.path.join(root, i))
        slice = np.array(img)
        images.append(slice)
    return np.array(images)


def random_weaken_contrast(im, mid_points, rad, sigma):
    for mid in mid_points:
        image = im[mid[0] - rad: mid[0] + rad,
                   mid[1] - rad: mid[1] + rad,
                   mid[1] - rad: mid[1] + rad]
        mean = image.mean()
        d = np.meshgrid(range(2 * rad),range(2 * rad), range(2 * rad))

        matrix = ((d[0] - rad) ** 2 + (d[1] - rad) ** 2 + (d[2] - rad) ** 2) ** (1 / 2)
    #     print(matrix.max())
        matrix[matrix > rad] = rad

        matrix = (1 - matrix / rad) * sigma
        res = image + matrix * (mean - image)

        im[mid[0] - rad: mid[0] + rad,
           mid[1] - rad: mid[1] + rad,
           mid[1] - rad: mid[1] + rad] = res
    return im


def contrast_augmentation(volume, label, rad=25, N=5, sigma=0.65):
    # raw = volume.copy()
    center_cube = label[rad: -rad, rad: -rad, rad: -rad]
    zxis, xxis, yxis = np.where(center_cube == 255)
    try:
        rands = random.sample(range(0, len(zxis)), N)
        mid = [[zxis[rand] + rad, xxis[rand] + rad, yxis[rand] + rad] for rand in rands]
    except:
        return volume
    return random_weaken_contrast(volume, mid, rad, sigma)


def torch_fliplr(x):
    return torch.flipud(x.transpose(0,2)).transpose(0,2)


def torch_dilation(tensor, iter):
    tensor_ori = tensor.clone()
    for _ in range(iter):
        p1 = torch.nn.functional.max_pool3d(tensor, (3, 1, 1), 1, (1, 0, 0))
        p2 = torch.nn.functional.max_pool3d(tensor, (1, 3, 1), 1, (0, 1, 0))
        p3 = torch.nn.functional.max_pool3d(tensor, (1, 1, 3), 1, (0, 0, 1))
        tensor = torch.max(torch.max(p1, p2), p3) - tensor_ori
    return tensor
