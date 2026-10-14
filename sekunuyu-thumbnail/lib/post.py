"""3D で描いた絵の仕上げ（ぼかし・発光・被写界深度・トーン・周辺減光）。"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def blur(img, sigma):
    return ndi.gaussian_filter(img, sigma=(sigma, sigma, 0), mode="nearest")


def bloom(img, strength=0.55, threshold=0.8):
    small = img[::2, ::2]
    bright = np.clip(small - threshold, 0, None)
    b = blur(bright, 6) * 0.6 + blur(bright, 18) * 0.5 + blur(bright, 45) * 0.45
    b = b.repeat(2, 0).repeat(2, 1)[: img.shape[0], : img.shape[1]]
    return img + b * strength


def depth_of_field(img, depth, near=9.0, far=40.0, sigma=3.0):
    k = smoothstep(near, far, np.where(np.isfinite(depth), depth, far))[..., None]
    return img * (1 - k) + blur(img, sigma) * k


def tonemap(x):
    knee = 0.78
    y = np.where(x < knee, x, knee + (1 - knee) * (1 - np.exp(-(x - knee) / (1 - knee))))
    return np.clip(y, 0, 1)


def grade(img, vignette=0.38, saturation=1.12):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
    vig = 1 - vignette * np.clip(r - 0.35, 0, None) ** 1.6
    out = img * vig[..., None]
    lum = out.mean(axis=2, keepdims=True)
    out = lum + (out - lum) * saturation
    return np.clip(out, 0, 1)
