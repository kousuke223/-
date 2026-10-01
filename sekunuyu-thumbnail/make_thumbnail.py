#!/usr/bin/env python3
"""マイクラ ハードコア #2（ネザー探検）のサムネイルを作る。

    python3 make_thumbnail.py            # output/thumbnail_02.png と .jpg（1280x720）
    python3 make_thumbnail.py --preview  # 文字なし・低画質ですばやく確認
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import urllib.request

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lib import textures as TX  # noqa: E402
from lib.models import Pose, ghast, player  # noqa: E402
from lib.voxel import Camera, Face, Lights, render  # noqa: E402
from lib.world3d import BRICK, FALL, GLOW, LAVA, MAGMA, QUARTZ, RACK, VoxelWorld  # noqa: E402

W, H = 1280, 720

FONTS = {
    "NotoSansJP-Black.ttf": "Noto+Sans+JP:wght@900",
    "DelaGothicOne-Regular.ttf": "Dela+Gothic+One",
}


def ensure_fonts() -> None:
    fdir = os.path.join(HERE, "fonts")
    os.makedirs(fdir, exist_ok=True)
    for fname, family in FONTS.items():
        path = os.path.join(fdir, fname)
        if os.path.exists(path):
            continue
        print(f"フォントをダウンロード中: {fname}")
        req = urllib.request.Request(f"https://fonts.googleapis.com/css2?family={family}",
                                     headers={"User-Agent": "Wget/1.12"})
        css = urllib.request.urlopen(req).read().decode()
        url = re.search(r"url\((https://[^)]+\.ttf)\)", css).group(1)
        urllib.request.urlretrieve(url, path)


# ---------------------------------------------------------------- 地形


LAVA_TOP = -1.125   # 溶岩の表面の高さ（足元のすぐ下）


def build_world(seed: int = 4) -> VoxelWorld:
    rng = np.random.default_rng(seed)
    w = VoxelWorld(-64, -8, -80, 128, 40, 88)
    # 溶岩の海
    w.box(-64, 64, -3, -2, -80, 8, RACK)
    w.box(-64, 64, -2, -1, -80, 8, LAVA)
    # プレイヤーが走っている岬（右と奥は溶岩）
    for x in range(-20, 6):
        for z in range(-16, 8):
            edge = 2 + int(1.3 * math.sin(z * 0.7) + rng.integers(0, 2))
            back = -5 - int(1.5 * math.sin(x * 0.6)) - (6 if x < -6 else 0)
            if x < edge and z >= back:
                w.box(x, x + 1, -3, 0, z, z + 1, RACK)
    for x, z in [(1, 3), (-3, -4), (-8, -9)]:
        w.box(x, x + 1, -1, 0, z, z + 1, MAGMA)
    for x, y, z in [(2, -1, 4), (-4, -1, -5), (1, -1, -4)]:
        w.box(x, x + 1, y, y + 1, z, z + 1, QUARTZ)
    for x, z in [(-5, -3), (-6, 1), (-9, -6), (-3, 4)]:  # 1段高いでこぼこ
        w.box(x, x + 1, 0, 1, z, z + 1, RACK)
    # 左奥の丘
    for x in range(-44, -9):
        for z in range(-50, -16):
            h = int(1 + 4 * (0.5 + 0.5 * math.sin(x * 0.21 + z * 0.13)) + rng.integers(0, 2) + max(0, -12 - x) * 0.3)
            w.box(x, x + 1, -2, h, z, z + 1, RACK)
    for x, y, z in [(-14, 3, -24), (-20, 5, -32), (-28, 7, -40)]:
        w.box(x, x + 2, y, y + 2, z, z + 2, GLOW)
    # 奥の崖と溶岩の滝
    for x in range(-64, 64):
        top = int(17 + 5 * math.sin(x * 0.17) + 3 * math.sin(x * 0.53) + rng.integers(0, 2))
        w.box(x, x + 1, -2, top, -80, -66, RACK)
    w.box(6, 10, -1, 16, -66, -65, FALL)
    w.box(-26, -23, -1, 12, -66, -65, FALL)
    # 溶岩に立つ岩の柱
    for (x, z, top) in [(11, -14, 3), (24, -30, 7), (34, -12, 2), (-4, -40, 4)]:
        w.box(x, x + 3, -2, top, z, z + 3, RACK)
        w.box(x - 1, x + 4, -2, -1, z - 1, z + 4, MAGMA)
    # ネザー要塞の橋
    w.box(-64, 40, 6, 8, -46, -42, BRICK)
    for x in range(-64, 40, 2):
        w.box(x, x + 1, 8, 9, -46, -45, BRICK)
        w.box(x, x + 1, 8, 9, -43, -42, BRICK)
    for x in range(-60, 40, 10):
        w.box(x, x + 3, -2, 6, -46, -42, BRICK)
    w.box(14, 24, -2, 16, -50, -38, BRICK)
    for y in (9, 12):
        w.box(17, 19, y, y + 2, -38, -37, GLOW)
        w.box(20, 22, y, y + 2, -38, -37, GLOW)
    # 右上に浮いた岩とグロウストーン
    w.box(18, 34, 15, 18, -34, -22, RACK)
    for x, z in [(20, -26), (24, -30), (28, -24), (31, -28)]:
        w.box(x, x + 2, 13, 15, z, z + 2, GLOW)
    return w


# 炎を置く場所（ブロックの上）
FIRES = [(-2, 0, -3), (-6, 0, -2), (-4, 1, -3), (-7, 0, -6), (-11, 0, -8), (12, 3, -14), (-9, 0, 2)]


def fire_faces(tex):
    """十字に組んだ2枚の板で炎を作る（マイクラと同じやり方）。"""
    out = []
    for x, y, z in FIRES:
        for e1 in (np.array([1.0, 0, 1.0]) / math.sqrt(2) * 1.15, np.array([1.0, 0, -1.0]) / math.sqrt(2) * 1.15):
            c = np.array([x + 0.5, y + 1.25, z + 0.5])
            out.append(Face(c - e1 / 2, e1, np.array([0, -1.25, 0]), tex, emis=1.7, lit=False, double=True))
    return out


def block_textures():
    return {
        RACK: [TX.netherrack(s) for s in (1, 2, 3)],
        LAVA: TX.lava(),
        GLOW: TX.glowstone(),
        BRICK: TX.nether_bricks(),
        MAGMA: TX.magma(),
        QUARTZ: TX.quartz_ore(),
        FALL: TX.lavafall(),
    }


# ---------------------------------------------------------------- シーン


class Scene:
    def __init__(self, w: int, h: int):
        self.cam = Camera((1.1, 0.95, 2.3), (-0.45, 1.95, 0.0), 52, w, h)
        pose = Pose(yaw=-25, lean=14, head_yaw=40, head_pitch=6, head_roll=-6,
                    arm_r=(-188, 58), arm_l=(26, 0), leg_r=-40, leg_l=35)
        self.ghast_pos = np.array([1.5, 7.3, -14.0])
        # 先にプレイヤーを置いて頭の位置を知る（ライトはあとで差し込む）
        T, HS = TX.player_textures(), TX.headset_textures()
        self.player_faces, self.pinfo = player(T, HS, None, (0, 0, 0), pose)
        gb, gm, gt = TX.ghast_textures()
        to_p = np.array([0.0, 1.6, 0.0]) - self.ghast_pos
        yaw = math.degrees(math.atan2(to_p[0], to_p[2]))
        pitch = math.degrees(math.atan2(-to_p[1], math.hypot(to_p[0], to_p[2]))) * 0.5
        self.ghast_faces, self.mouth = ghast(gb, gm, gt, self.ghast_pos, yaw=yaw, pitch=pitch)
        target = self.pinfo["head_center"] + np.array([0.25, 0.2, -0.4])
        self.fireball = self.mouth + 0.885 * (target - self.mouth)

        fire = (self.fireball, (1.5, 0.72, 0.2), 1.6)
        # キャラ用：顔がはっきり見えるキーライト＋赤いふちどり
        self.hero = Lights(
            ambient=(0.2, 0.1, 0.1),
            dirs=[((-0.45, 0.55, 1.0), (1.0, 0.86, 0.76)), ((0.9, 0.35, -0.5), (1.0, 0.38, 0.12))],
            points=[fire], lava_y=LAVA_TOP, lava_color=(1.0, 0.36, 0.09), lava_h=2.2)
        # 地形用：溶岩の照り返しと火の玉の光だけ（暗めにして文字を読みやすく）
        self.lights = Lights(
            ambient=(0.11, 0.04, 0.04),
            dirs=[((0.8, 0.25, 0.4), (0.32, 0.11, 0.05))],
            points=[(self.fireball, (2.6, 1.25, 0.35), 3.2)],
            lava_y=LAVA_TOP, lava_color=(1.1, 0.38, 0.09), lava_h=2.0,
            blobs=[(0.0, 0.1, 0.55, 0.6)])
        for f in self.player_faces + self.ghast_faces:
            f.lights = self.hero
        self.world = build_world()
        self.world_faces = self.world.faces(block_textures(), near_z=-7) + fire_faces(TX.fire())

    def render(self):
        faces = self.player_faces + self.ghast_faces + self.world_faces
        return render(faces, self.cam, self.lights)


# ---------------------------------------------------------------- 仕上げ（霧・発光・ぼかし）


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def fog_and_sky(color, depth, wy, cam: Camera):
    h, w = depth.shape
    yy = (np.arange(h)[:, None] + 0.5 - cam.cy) / cam.fpx
    # 空（何もないところ）は、視線の高さで色を変える
    up = -yy + cam.R[2, 1]  # ざっくり：画面上ほど上向き
    sky_k = smoothstep(-0.15, 0.55, up)
    low = np.array([0.95, 0.30, 0.10])
    high = np.array([0.16, 0.03, 0.04])
    sky = low * (1 - sky_k)[..., None] + high * sky_k[..., None]
    sky = np.broadcast_to(sky, (h, w, 3))
    hit = np.isfinite(depth)
    D = np.where(hit, depth, 400.0)
    hk = smoothstep(-4, 14, np.where(hit, wy, 30))[..., None]
    fogc = low * (1 - hk) * 0.9 + high * hk
    k = (1 - np.exp(-np.clip(D - 6, 0, None) * 0.028))[..., None]
    out = np.where(hit[..., None], color * (1 - k) + fogc * k, sky)
    return out.astype(np.float32)


def blur(img, sigma):
    return ndi.gaussian_filter(img, sigma=(sigma, sigma, 0), mode="nearest")


def bloom(img, strength=0.55):
    small = img[::2, ::2]
    bright = np.clip(small - 0.8, 0, None)
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


def grade(img):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
    vig = 1 - 0.38 * np.clip(r - 0.35, 0, None) ** 1.6
    out = img * vig[..., None]
    lum = out.mean(axis=2, keepdims=True)
    out = lum + (out - lum) * 1.12
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------- 火の玉・火の粉


def fireball(img, cam: Camera, pos, src, radius=0.34, seed=3):
    """ドット感のある火の玉と、ガストの口から伸びる炎の尾（加算合成）。"""
    h, w = img.shape[:2]
    fx, fy, fz = cam.project(np.asarray(pos))
    sx, sy, _ = cam.project(np.asarray(src))
    r = radius * cam.fpx / fz
    d = np.array([sx - fx, sy - fy])
    trail_len = max(np.linalg.norm(d), r * 4)
    d = d / (np.linalg.norm(d) + 1e-6)
    cell = max(2, int(r / 6))
    gh, gw = h // cell + 1, w // cell + 1
    gy, gx = np.mgrid[0:gh, 0:gw]
    px = (gx + 0.5) * cell - fx
    py = (gy + 0.5) * cell - fy
    along = px * d[0] + py * d[1]
    across = -px * d[1] + py * d[0]
    rng = np.random.default_rng(seed)
    noise = ndi.zoom(rng.random((gh // 3 + 2, gw // 3 + 2)), 3, order=1)[:gh, :gw]
    # 玉（ふちがゆらゆら）
    ang = np.arctan2(py, px)
    wob = 1 + 0.18 * np.sin(ang * 5 + 1.3) + (noise - 0.5) * 0.3
    dist = np.hypot(px, py) / (r * wob)
    ball = np.clip(1.25 - dist, 0, 1)
    # 尾（ガスト側に、ゆらぎながら細くなる）
    tl = trail_len * 0.95
    t = np.clip(along / tl, 0, 1)
    width = r * 1.05 * (1 - t) ** 0.7 * (1 + 0.35 * np.sin(along / (r * 0.6) + 2))
    tail = np.where((along > -r * 0.2) & (along < tl),
                    np.clip(1 - np.abs(across) / (width + 1e-6) + (noise - 0.5) * 0.7, 0, 1) * (1 - t) ** 1.1, 0)
    v = np.maximum(ball, tail)
    col = np.zeros((gh, gw, 3), np.float32)
    col += ((v > 0.04) * v)[..., None] * np.array([1.3, 0.16, 0.04])
    col += (v > 0.3)[..., None] * np.array([1.1, 0.5, 0.06])
    col += (v > 0.62)[..., None] * np.array([0.9, 0.9, 0.35])
    col += (ball > 0.85)[..., None] * np.array([0.5, 0.5, 0.4])
    up = col.repeat(cell, 0).repeat(cell, 1)[:h, :w]
    img += up + blur(up, r * 0.5) * 0.7
    return (fx, fy, r)


def embers(img, n=170, seed=8, around=None):
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    layer = np.zeros_like(img)
    for i in range(n):
        if around is not None and i < n // 3:
            x = around[0] + rng.normal(0, around[2] * 1.8)
            y = around[1] + rng.normal(0, around[2] * 1.4)
        else:
            x = rng.uniform(0, w)
            y = h - abs(rng.normal(0, h * 0.45))
        s = int(rng.choice([2, 3, 3, 4, 5]) * w / 1280)
        x, y = int(x), int(y)
        if 0 <= x < w - s and 0 <= y < h - s:
            c = np.array([1.6, 0.6, 0.12]) if rng.random() < 0.7 else np.array([1.8, 1.3, 0.5])
            layer[y:y + s, x:x + s] += c * rng.uniform(0.5, 1.2)
    img += layer + blur(layer, 4 * w / 1280) * 1.5


# ---------------------------------------------------------------- 文字入れ

TEXT = {
    "series1": "マイクラ",
    "series2": "ハードコア",
    "catch1": "ネザー、",
    "catch2": "無理。",
    "badge": "#2",
}


def compose(art: Image.Image, S: int) -> Image.Image:
    """S 倍の大きさで文字・ハート・バッジを描く。"""
    from lib import overlay as ov
    fdir = os.path.join(HERE, "fonts")
    black = os.path.join(fdir, "NotoSansJP-Black.ttf")
    dela = os.path.join(fdir, "DelaGothicOne-Regular.ttf")
    c = art.convert("RGBA")
    dark = (34, 6, 8, 255)
    shadow = (6 * S, 8 * S, (0, 0, 0, 170), 4 * S)
    # シリーズ名（#1 と同じ雰囲気）
    f = ov.font(black, 118 * S)
    b1 = ov.fx_text(c, (22 * S, 2 * S), TEXT["series1"], f,
                    [(0, (255, 240, 120)), (0.5, (255, 206, 40)), (1, (255, 138, 18))],
                    strokes=[(16 * S, dark), (9 * S, (214, 28, 28, 255))], shadow=shadow, spacing=-4 * S)
    ov.fx_text(c, (b1[2] + 4 * S, 2 * S), TEXT["series2"], f,
               [(0, (255, 92, 70)), (0.55, (236, 28, 30)), (1, (176, 10, 16))],
               strokes=[(16 * S, dark), (9 * S, (255, 255, 255, 255))], shadow=shadow, spacing=-4 * S)
    # キャッチ
    fc = ov.font(black, 86 * S)
    b2 = ov.fx_text(c, (30 * S, 150 * S), TEXT["catch1"], fc, (255, 255, 255, 255),
                    strokes=[(11 * S, dark)], shadow=(5 * S, 7 * S, (60, 0, 0, 220), 2 * S), spacing=-4 * S)
    fc2 = ov.font(black, 104 * S)
    b3 = ov.fx_text(c, (b2[2] - 10 * S, 136 * S), TEXT["catch2"], fc2,
                    [(0, (255, 250, 170)), (0.45, (255, 214, 60)), (1, (255, 120, 20))],
                    strokes=[(13 * S, dark)], shadow=(5 * S, 7 * S, (60, 0, 0, 220), 2 * S), spacing=-6 * S, rotate=-4)
    ov.badge(c, (b3[2] + 6 * S, 156 * S), TEXT["badge"], ov.font(dela, 64 * S), pad=(20 * S, 6 * S),
             border=8 * S, radius=16 * S, rotate=6)
    # 飛び散る汗（頭の左側）
    for (sx, sy, ang, sc) in [(478, 300, 35, 5), (446, 362, 62, 4), (470, 426, 88, 4)]:
        c.alpha_composite(ov.sweat(sc * S, ang), (sx * S, sy * S))
    # 体力（ハート）
    hx, hy = 26 * S, 652 * S
    kinds = ["full", "half"] + ["empty"] * 8
    for i, k in enumerate(kinds):
        c.alpha_composite(ov.heart(k, 6 * S), (hx + i * 44 * S, hy))
    return c


# ---------------------------------------------------------------- メイン


def render_art(ss: int = 2) -> Image.Image:
    """3D シーンを ss 倍の大きさで描いて仕上げる（縮小はしない）。"""
    sc = Scene(W * ss, H * ss)
    color, depth, wy = sc.render()
    img = fog_and_sky(color, depth, wy, sc.cam)
    fb = fireball(img, sc.cam, sc.fireball, sc.mouth)
    embers(img, around=fb)
    img = depth_of_field(img, depth, sigma=2.2 * ss)
    img = bloom(img)
    img = grade(tonemap(img))
    return Image.fromarray((img * 255 + 0.5).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="等倍ですばやく確認")
    ap.add_argument("--out", default=os.path.join(HERE, "output"))
    args = ap.parse_args()
    ensure_fonts()
    os.makedirs(args.out, exist_ok=True)
    S = 1 if args.preview else 2
    art = render_art(ss=S)
    thumb = compose(art, S).convert("RGB")
    if S != 1:
        thumb = thumb.resize((W, H), Image.LANCZOS)
    name = "preview" if args.preview else "thumbnail_02"
    thumb.save(os.path.join(args.out, f"{name}.png"))
    if not args.preview:
        thumb.save(os.path.join(args.out, f"{name}.jpg"), quality=92, optimize=True)
    print("ok")


if __name__ == "__main__":
    main()
