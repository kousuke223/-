"""チャンネルアイコン（切り抜き済み PNG）をドット絵アバターに変換する。"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance
from scipy import ndimage as ndi

from .pixelart import blank, dilate, ellipse_mask, hexc, line_mask, pad

OUTLINE = hexc("#2A1E2E")


def _kmeans_palette(pixels: np.ndarray, k: int, iters: int = 16, seed: int = 7) -> np.ndarray:
    """小さな k-means（k-means++ 初期化）。pixels: (N, 3) float32"""
    rng = np.random.default_rng(seed)
    centers = [pixels[rng.integers(len(pixels))]]
    for _ in range(1, k):
        d = np.min(((pixels[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        centers.append(pixels[rng.choice(len(pixels), p=d / d.sum())])
    c = np.array(centers, np.float32)
    for _ in range(iters):
        lab = np.argmin(((pixels[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        for i in range(k):
            sel = pixels[lab == i]
            if len(sel):
                c[i] = sel.mean(0)
    return c


def pixelate(cutout: Image.Image, grid_w: int, colors: int = 26) -> np.ndarray:
    rgb = cutout.convert("RGB")
    alpha = np.asarray(cutout.getchannel("A")) > 127
    arr = np.asarray(rgb).astype(np.float32)
    # 透明部分に一番近い不透明ピクセルの色を広げて、縮小時のフチの白っぽさを防ぐ
    idx = ndi.distance_transform_edt(~alpha, return_distances=False, return_indices=True)
    rgb = Image.fromarray(arr[idx[0], idx[1]].astype(np.uint8))
    rgb = ImageEnhance.Color(rgb).enhance(1.3)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.08)
    gw = grid_w
    gh = round(cutout.height * gw / cutout.width)
    small = np.asarray(rgb.resize((gw, gh), Image.BOX)).astype(np.float32)
    am = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8)).resize((gw, gh), Image.BOX)) > 115
    px = small[am]
    pal = _kmeans_palette(px, colors)
    lab = np.argmin(((small[..., None, :] - pal[None, None]) ** 2).sum(-1), axis=-1)
    out = blank(gw, gh)
    out[..., :3] = np.clip(pal[lab] + 0.5, 0, 255).astype(np.uint8)
    out[..., 3] = np.where(am, 255, 0)
    return out


def add_headset(spr: np.ndarray, s: float, off: int, main=hexc("#FFD23F"), shade=hexc("#E08A1E"),
                band=hexc("#34364A"), band_hi=hexc("#6B7090")) -> None:
    """ゲーミングヘッドセットを描き足す。座標は切り抜き画像基準 → s 倍して off ずらす。"""
    h, w = spr.shape[:2]

    def P(x, y):
        return x * s + off, y * s + off

    cx, cy = P(55.5, 82)
    # ヘッドバンド（楕円の上半分のリング）。外周が髪の輪郭にほぼ沿う大きさ
    rx_o, ry_o = 39.5 * s, 71 * s
    th = max(2.0, 5.5 * s)
    outer = ellipse_mask(h, w, cx, cy, rx_o, ry_o)
    inner = ellipse_mask(h, w, cx, cy, rx_o - th, ry_o - th)
    yy = np.arange(h)[:, None]
    ring = outer & ~inner & (yy < cy - 10 * s)
    hi = ring & ~ellipse_mask(h, w, cx, cy, rx_o - 1, ry_o - 1)
    spr[ring] = band
    spr[hi] = band_hi
    # イヤーカップ
    for ex in (15.5, 95.5):
        ecx, ecy = P(ex, 82)
        cup = ellipse_mask(h, w, ecx, ecy, 8.5 * s, 15 * s)
        cup_in = ellipse_mask(h, w, ecx + (1 if ex < 50 else -1), ecy + 1, 8.5 * s - 1.5, 15 * s - 2)
        cushion = ellipse_mask(h, w, ecx + (3.5 * s if ex < 50 else -3.5 * s), ecy, 5 * s, 13 * s)
        spr[cushion] = band
        spr[cup & ~cushion] = shade
        spr[cup_in & ~cushion] = main
        # ハイライト
        hx, hy = P(ex - 3 if ex < 50 else ex - 1, 72)
        spr[int(hy):int(hy) + 2, int(hx):int(hx) + 1] = hexc("#FFF6C2")
    # マイク
    pts = [P(12, 92), P(15, 104), P(24, 112), P(36, 114)]
    boom = line_mask(h, w, pts, width=max(1, round(2.2 * s)))
    spr[boom] = band
    mx, my = P(38, 114)
    tip = ellipse_mask(h, w, mx, my, 3.2 * s, 3.2 * s)
    spr[tip] = main
    spr[tip & ~ellipse_mask(h, w, mx - 0.6, my - 0.6, 3.2 * s - 1, 3.2 * s - 1)] = shade


def make_avatar(cutout_path: str, grid_w: int = 64, headset: bool = True, **headset_colors) -> np.ndarray:
    cut = Image.open(cutout_path).convert("RGBA")
    spr = pixelate(cut, grid_w)
    off = 5
    spr = pad(spr, off)
    if headset:
        add_headset(spr, grid_w / cut.width, off, **headset_colors)
    m = spr[..., 3] > 0
    ring = dilate(m, 1, diag=False) & ~m
    spr[ring] = OUTLINE
    # 余白を詰める
    ys, xs = np.nonzero(spr[..., 3])
    return spr[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()

