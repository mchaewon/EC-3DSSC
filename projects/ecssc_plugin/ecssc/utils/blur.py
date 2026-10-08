# ---------------------------------------------
# EC-3DSSC (ECSSC). Built on VoxFormer (https://github.com/NVlabs/VoxFormer),
# NVIDIA Source Code License-NC. See LICENSE.
# ---------------------------------------------
"""Test-time linear motion blur for robustness evaluation.

The blur is applied to the normalized RGB tensor right before the image backbone
(current frame and all temporal frames). Because ImageNet normalization is a per-channel
affine map and the kernel sums to 1 (replicate padding), this is equivalent to blurring
the 0-1 RGB image.

    severity 0: no blur (input returned unchanged)
    severity 1 / 2 / 3: kernel length 7 / 15 / 25 px, uniform weights 1/k
    angle: blur direction in degrees (0 = horizontal)
"""
import math

import torch
import torch.nn.functional as F

BLUR_KERNEL_SIZE = {1: 7, 2: 15, 3: 25}


def motion_kernel(k, angle_deg):
    kern = torch.zeros(k, k)
    c = (k - 1) / 2
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    for i in range(k):
        t = i - c
        x = int(round(c + t * dx)); y = int(round(c + t * dy))
        if 0 <= x < k and 0 <= y < k:
            kern[y, x] = 1.0
    return (kern / kern.sum()).view(1, 1, k, k)


def motion_blur(img, severity=0, angle=0.0):
    """img: (..., C, H, W) tensor. Returns img itself when severity <= 0."""
    severity = int(severity)
    if severity <= 0 or img is None:
        return img
    if severity not in BLUR_KERNEL_SIZE:
        raise ValueError(f'blur severity must be 0-3, got {severity}')
    k = BLUR_KERNEL_SIZE[severity]
    ker = motion_kernel(k, float(angle)).to(img.device, img.dtype)
    shp = img.shape
    x = img.reshape(-1, *shp[-3:])
    w = ker.expand(x.shape[1], 1, k, k)
    x = F.conv2d(F.pad(x, (k // 2,) * 4, mode='replicate'), w, groups=x.shape[1])
    return x.reshape(shp)
