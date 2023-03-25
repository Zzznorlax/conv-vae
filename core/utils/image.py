import cv2
import math
import rawpy
import numpy as np
from random import random

import torch


def resize(img, size: int = 512, strict: bool = False):

    short = min(img.shape[:2])
    scale = size / short

    if not strict:
        img = cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)), interpolation=cv2.INTER_NEAREST)

    else:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)

    return img


def crop(img, size=512):
    try:
        y, x = random.randint(0, img.shape[0] - size), random.randint(0, img.shape[1] - size)

    except Exception as _:
        y, x = 0, 0

    return img[y:y + size, x:x + size, :]


def to_size(img: np.ndarray, size: int = 512, use_crop: bool = False):

    if use_crop:
        img = crop(img, size)

    else:
        img = resize(img, size=size)

    return img


def read_raw(path: str, bps: int = 16) -> np.ndarray:

    img = None
    with rawpy.imread(path) as raw:
        img = raw.postprocess(output_bps=bps, demosaic_algorithm=None)

    return img


def compress_img(img: np.ndarray, compress_bits: int = 8) -> np.ndarray:
    comp_img = (img / 2**compress_bits).astype(np.uint8)

    return comp_img


def show_bit_depth(img: np.ndarray):
    max_val = np.max(img)
    min_val = np.min(img)

    # bit_depth = 2**math.ceil(math.log2(math.log2(max_val)))
    bit_depth = math.log2(max_val)

    print("image dimensions: {}".format(img.shape))
    print("value range: {} >= val >= {}".format(max_val, min_val))
    print("value bit count: {}".format(bit_depth))

    return bit_depth


def save_tensor_to_img(tensor: torch.Tensor, dest: str, output_depth: int = 8):
    img = tensor.squeeze(0).cpu().detach().numpy()
    bit_depth = show_bit_depth(img)

    dtype = np.uint8
    if output_depth == 16:
        dtype = np.uint16

    remapping_scale = 2**(bit_depth - output_depth)
    # if bit_depth < 0:
    #     dtype = np.float32
    #     remapping_scale = 1

    if abs(bit_depth - output_depth) < 1:
        remapping_scale = 1

    img = (img / remapping_scale).astype(dtype)

    img = np.moveaxis(img, 0, -1)

    cv2.imwrite(dest, img)

    return img


def hist_equalization(img, num_bins: int = 256):

    # get image histogram
    hist, bins = np.histogram(img.flatten(), num_bins, density=True)
    cdf = hist.cumsum()  # cumulative distribution function
    cdf = (num_bins - 1) * cdf / cdf[-1]  # normalize

    eq_img = np.interp(img.flatten(), bins[:-1], cdf)

    return eq_img.reshape(img.shape), cdf


if __name__ == "__main__":

    path = 'inputs/raw.dng'

    img = read_raw(path)

    img = resize(img)

    show_bit_depth(img)
