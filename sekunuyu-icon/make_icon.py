#!/usr/bin/env python3
"""せくぬゆの YouTube チャンネルアイコン（800×800）を作る。

    python3 make_icon.py   # output/icon_A.png 〜 icon_C.png と、見くらべ用の icon_options.png を書き出す

キャラは ../sekunuyu-skin/sekunuyu_skin.png（エンダーパーカー＋黄色いヘッドセット）を、
サムネイルと同じレンダラーで描く。YouTube では丸く切り取られるので、大事なものは真ん中の円に入れる。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "sekunuyu-thumbnail"))
sys.path.insert(0, os.path.join(ROOT, "sekunuyu-skin"))

import make_skin  # noqa: E402
from lib import overlay as ov  # noqa: E402
from lib.fonts import BLACK, ensure_fonts  # noqa: E402
from lib.models import Pose, player  # noqa: E402
from lib.post import tonemap  # noqa: E402
from lib.skin import CHANNEL_SKIN, load_skin, skin_textures  # noqa: E402
from lib.voxel import Camera, Lights, render  # noqa: E402

SIZE = 800
SS = 2                       # 2倍で描いて縮める
YELLOW = [(255, 236, 140), (255, 196, 40)]
PURPLE = [(120, 70, 190), (40, 22, 70)]
SPARK = [(194, 91, 255), (123, 59, 181), (236, 190, 255)]


# ---------------------------------------------------------------- 背景


def radial(size, inner, outer, center=(0.5, 0.5), reach=0.75) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size] / size
    t = np.clip(np.hypot(xx - center[0], yy - center[1]) / reach, 0, 1)[..., None]
    return np.array(inner, float) * (1 - t) + np.array(outer, float) * t


def rays(img: np.ndarray, n=18, center=(0.5, 0.5), strength=0.1) -> np.ndarray:
    """中心から広がる光の筋（明るい扇形と暗い扇形を交互に）。"""
    size = img.shape[0]
    yy, xx = np.mgrid[0:size, 0:size] / size
    ang = np.arctan2(yy - center[1], xx - center[0])
    on = (np.floor((ang + math.pi) / (2 * math.pi) * n * 2) % 2 == 0)[..., None]
    return np.where(on, img * (1 + strength) + 8, img * (1 - strength * 0.4))


def sparkles(im: Image.Image, n: int, seed: int, keep_out=None):
    """エンダーのつぶつぶ（むらさきの小さな四角）を散らす。keep_out=(cx, cy, r) の円の中には置かない。"""
    rng = np.random.default_rng(seed)
    d = ImageDraw.Draw(im)
    W = im.width
    placed = 0
    while placed < n:
        x, y = rng.uniform(0.06, 0.94, 2) * W
        if keep_out and math.hypot(x - keep_out[0] * W, y - keep_out[1] * W) < keep_out[2] * W:
            continue
        s = rng.uniform(0.018, 0.04) * W
        col = SPARK[int(rng.integers(len(SPARK)))]
        d.rectangle((x - s / 2, y - s / 2, x + s / 2, y + s / 2), fill=col + (255,))
        if rng.random() < 0.5:   # 小さいつぶを横にもう1つ
            s2 = s * 0.45
            d.rectangle((x + s * 0.7, y - s * 0.9, x + s * 0.7 + s2, y - s * 0.9 + s2), fill=col + (255,))
        placed += 1


# ---------------------------------------------------------------- キャラ


def render_char(T, pose: Pose, cam_pos, target, fov: float, size: int) -> Image.Image:
    cam = Camera(cam_pos, target, fov, size, size)
    c = np.array(cam_pos, float) - np.array(target, float)
    key = c / np.linalg.norm(c) + np.array([0.2, 0.9, 0]) + 0.5 * cam.R[0]
    L = Lights(ambient=(0.5, 0.48, 0.56), dirs=[(tuple(key), (0.72, 0.68, 0.62)),
                                                ((-c[0], 0.4, -c[2]), (0.5, 0.3, 0.75))])
    faces, _ = player(T, None, None, (0, 0, 0), pose)
    color, depth, _ = render(faces, cam, L)
    rgb = (np.clip(tonemap(color), 0, 1) * 255 + 0.5).astype(np.uint8)
    alpha = (np.isfinite(depth) * 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]), "RGBA")


def sticker(char: Image.Image, width: int, shadow=(0, 0, 0, 90)) -> Image.Image:
    """キャラのまわりに白いフチと、やわらかい影を付ける（小さく表示しても形が分かるように）。"""
    a = char.getchannel("A")
    edge = a.filter(ImageFilter.MaxFilter(2 * width + 1))
    out = Image.new("RGBA", char.size, (0, 0, 0, 0))
    sh = Image.new("RGBA", char.size, shadow)
    out.paste(sh, (width // 2, width), edge.filter(ImageFilter.GaussianBlur(width)))
    out.paste(Image.new("RGBA", char.size, (255, 255, 255, 255)), (0, 0), edge)
    out.alpha_composite(char)
    return out


# ---------------------------------------------------------------- 3つの案


def icon_a(T) -> Image.Image:
    """A：顔アップ。黄色い光の筋の上に、首をかしげたせくぬゆ。"""
    S = SIZE * SS
    bg = rays(radial(S, *YELLOW, center=(0.5, 0.42)), n=16, center=(0.5, 0.42))
    im = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).convert("RGBA")
    sparkles(im, 9, seed=3, keep_out=(0.5, 0.5, 0.36))
    pose = Pose(yaw=-14, head_yaw=14, head_pitch=4, head_roll=-7, arm_r=(-6, 10), arm_l=(6, 10))
    ch = render_char(T, pose, (0.82, 1.92, 2.15), (0.02, 1.6, 0), 30, S)
    im.alpha_composite(sticker(ch, 7 * SS))
    return im.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)


def icon_b(T) -> Image.Image:
    """B：手をふる。むらさきの背景に黄色い光の輪、まわりにエンダーのつぶつぶ。"""
    S = SIZE * SS
    bg = radial(S, *PURPLE, center=(0.5, 0.45), reach=0.8)
    im = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).convert("RGBA")
    ring = Image.new("L", (S, S), 0)
    ImageDraw.Draw(ring).ellipse((S * 0.17, S * 0.1, S * 0.83, S * 0.76), fill=255)
    im.paste(Image.new("RGBA", (S, S), (255, 210, 63, 255)), (0, 0), ring.filter(ImageFilter.GaussianBlur(S * 0.04)))
    sparkles(im, 12, seed=8, keep_out=(0.5, 0.45, 0.36))
    pose = Pose(yaw=-14, head_yaw=12, head_pitch=3, head_roll=6, arm_r=(-150, 38), arm_l=(6, 10))
    ch = render_char(T, pose, (0.85, 1.9, 2.45), (-0.1, 1.68, 0), 32, S)
    im.alpha_composite(sticker(ch, 7 * SS))
    return im.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)


def pixel_bust(skin: np.ndarray) -> np.ndarray:
    """C 用：スキンの顔（正面＋フード）に、ヘッドセットとパーカーの肩を足したドット絵（14×13）。"""
    rgb = lambda k: tuple(int(make_skin.PAL[k][i:i + 2], 16) for i in (1, 3, 5)) + (255,)
    g = np.zeros((13, 14, 4), np.uint8)
    face = skin[8:16, 8:16].copy()
    hat = skin[8:16, 40:48]
    face[hat[..., 3] > 0] = hat[hat[..., 3] > 0]
    g[1:9, 3:11] = face
    for x in range(2, 12):           # 頭の上のバンド
        g[0, x] = rgb("z")
    for y in range(1, 4):            # 耳あてまでおりるバンド
        g[y, 2] = g[y, 11] = rgb("z")
    for y in range(4, 8):            # 耳あて（黄色）
        for x in (1, 2, 11, 12):
            g[y, x] = rgb("y")
        g[y, 1] = g[y, 12] = rgb("Y")
    g[4, 2] = g[4, 11] = rgb("L")
    # パーカーの肩（ふちどり・Vネック・ファスナー）
    rows = ["KKPPPPssPPPPKK", "KPPPVPZZPPvPPK", "KPvPPPzZPPPPVK", "KPPPPPZzPPPvPK"]
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            g[9 + y, x] = rgb(ch)
    return g


def icon_c(skin: np.ndarray) -> Image.Image:
    """C：ドット絵。スキンそのままの顔を大きく。小さく表示しても一番はっきり見える。"""
    S = SIZE * SS
    bg = radial(S, (255, 240, 160), (255, 190, 30), center=(0.5, 0.4), reach=0.8)
    im = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(im)
    step = S // 16                   # うすいマス目（ブロックっぽさ）
    for i in range(0, S, step):
        d.line((i, 0, i, S), fill=(255, 255, 255, 60), width=SS * 2)
        d.line((0, i, S, i), fill=(255, 255, 255, 60), width=SS * 2)
    sparkles(im, 8, seed=5, keep_out=(0.5, 0.48, 0.42))
    g = pixel_bust(skin)
    cell = int(S * 0.058)
    art = Image.fromarray(g, "RGBA").resize((g.shape[1] * cell, g.shape[0] * cell), Image.NEAREST)
    canvas = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    canvas.alpha_composite(art, ((S - art.width) // 2, int(S * 0.13)))
    im.alpha_composite(sticker(canvas, 8 * SS))
    return im.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)


# ---------------------------------------------------------------- 見くらべ用


def circle(img: Image.Image, size: int) -> Image.Image:
    """YouTube と同じように丸く切り取った見本。"""
    s = img.resize((size, size), Image.LANCZOS).convert("RGBA")
    m = Image.new("L", (size * 4, size * 4), 0)
    ImageDraw.Draw(m).ellipse((0, 0, size * 4 - 1, size * 4 - 1), fill=255)
    s.putalpha(m.resize((size, size), Image.LANCZOS))
    return s


def options_sheet(icons: dict) -> Image.Image:
    W, H = 1280, 720
    bg = ov.gradient((W, H), [(0, (246, 246, 250)), (1, (226, 230, 240))]).convert("RGBA")
    d = ImageDraw.Draw(bg)
    f1, f2, f3 = ov.font(BLACK, 34), ov.font(BLACK, 30), ov.font(BLACK, 20)
    d.text((32, 22), "チャンネルアイコンの案（YouTube では丸く表示されます）", font=f1, fill=(30, 34, 52))
    names = {"A": "A：顔アップ＋光", "B": "B：手をふる", "C": "C：ドット絵"}
    for i, (k, img) in enumerate(icons.items()):
        x = 40 + i * 410
        bg.alpha_composite(circle(img, 340), (x + 15, 90))
        d.text((x + 185, 470), names[k], font=f2, fill=(30, 34, 52), anchor="mm")
        # 小さく表示されたとき（チャンネルページ・コメント欄・おすすめ欄くらい）
        for j, s in enumerate((98, 48, 36)):
            bx = x + 30 + j * 120
            bg.alpha_composite(circle(img, s), (bx + (98 - s) // 2, 520 + (98 - s) // 2))
            d.text((bx + 49, 640), f"{s}px", font=f3, fill=(90, 96, 120), anchor="mm")
    return bg.convert("RGB")


def main():
    ensure_fonts()
    skin = load_skin(CHANNEL_SKIN)
    T = skin_textures(skin)
    out = os.path.join(HERE, "output")
    os.makedirs(out, exist_ok=True)
    icons = {"A": icon_a(T), "B": icon_b(T), "C": icon_c(skin)}
    for k, img in icons.items():
        img.save(os.path.join(out, f"icon_{k}.png"))
    options_sheet(icons).save(os.path.join(HERE, "icon_options.png"))
    print("output/icon_A.png〜icon_C.png と icon_options.png を書き出しました")


if __name__ == "__main__":
    main()
