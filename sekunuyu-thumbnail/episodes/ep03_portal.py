"""#3：最初からネザーゲート造りまで（夕焼けの中、完成したゲートの前で大喜び）。"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image

from lib import fonts
from lib import overlay as ov
from lib import textures as TX
from lib.models import Pose, player
from lib.post import bloom, blur, depth_of_field, grade, tonemap
from lib.skin import load_skin, skin_textures
from lib.voxel import Camera, Face, Lights, render
from lib.world3d import DIRT, GRASS, LEAVES, LOG, OBSIDIAN, PLANKS, STONE, VoxelWorld

NAME = "thumbnail_03"
W, H = 1280, 720

# ネザーゲート：黒曜石のわく（外側 4×5）。中の渦は z の真ん中に板で置く
PORTAL_X, PORTAL_Z = 2, -6
PORTAL_IN = (PORTAL_X + 1, PORTAL_X + 3, 1, 4)   # 渦の範囲 x0, x1, y0, y1
SUN_DIR = np.array([0.12, 0.1, -1.0])             # 夕日の方向（キャラとゲートのあいだの奥・低い）
# キャラはチャンネルのマイクラスキン（エンダーパーカー＋黄色いヘッドセット）をそのまま着せる
SKIN = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "sekunuyu-skin", "sekunuyu_skin.png")


# ---------------------------------------------------------------- 地形


def tree(w: VoxelWorld, x, z, h=5, y=0):
    w.box(x - 2, x + 3, y + h - 2, y + h, z - 2, z + 3, LEAVES)
    w.box(x - 1, x + 2, y + h, y + h + 2, z - 1, z + 2, LEAVES)
    w.box(x, x + 1, y, y + h, z, z + 1, LOG)


def build_world(seed: int = 7) -> VoxelWorld:
    rng = np.random.default_rng(seed)
    w = VoxelWorld(-64, -8, -96, 128, 40, 104)
    # 地面：草・土・石
    w.box(-64, 64, -8, -4, -96, 8, STONE)
    w.box(-64, 64, -4, -1, -96, 8, DIRT)
    w.box(-64, 64, -1, 0, -96, 8, GRASS)
    # 奥のなだらかな丘と山
    for x in range(-64, 64):
        for z in range(-96, -14):
            d = -14 - z
            h = (2.5 * math.sin(x * 0.11 + 1.0) + 2.0 * math.sin(z * 0.09) + 1.2 * math.sin((x + z) * 0.23)) * \
                min(1.0, d / 18) + max(0, d - 40) * 0.28 + rng.random() * 0.6
            h = int(h)
            if h > 0:
                w.box(x, x + 1, 0, h - 1, z, z + 1, DIRT)
                w.box(x, x + 1, h - 1, h, z, z + 1, GRASS if h < 13 else STONE)
    # 木
    for x, z in [(-9, -7), (-15, -11), (-21, -6), (-6, -15), (9, -12), (14, -8), (19, -15), (-27, -14),
                 (24, -22), (-12, -24), (5, -26), (30, -12)]:
        top = 0
        while w.get(x, top, z) != 0:
            top += 1
        tree(w, x, z, h=4 + (x * 7 + z) % 3, y=top)
    # 最初に建てた家（左奥）
    hx, hz = -16, -19
    w.box(hx, hx + 6, 0, 4, hz, hz + 5, PLANKS)
    w.box(hx + 1, hx + 5, 0, 4, hz + 1, hz + 4, 0)
    w.box(hx + 2, hx + 3, 0, 2, hz + 4, hz + 5, 0)          # 入口
    w.box(hx + 4, hx + 5, 2, 3, hz + 4, hz + 5, 0)          # 窓
    w.box(hx - 1, hx + 7, 4, 5, hz - 1, hz + 6, LOG)
    w.box(hx, hx + 6, 5, 6, hz, hz + 5, LOG)
    # ネザーゲート（黒曜石のわく）
    px, pz = PORTAL_X, PORTAL_Z
    w.box(px, px + 4, 0, 5, pz, pz + 1, OBSIDIAN)
    w.box(px + 1, px + 3, 1, 4, pz, pz + 1, 0)
    return w


def block_textures():
    return {
        GRASS: {"top": TX.grass_top(), "side": TX.grass_side(), "bottom": TX.dirt()},
        DIRT: TX.dirt(),
        STONE: TX.stone(),
        LOG: {"top": TX.oak_log_top(), "side": TX.oak_log_side(), "bottom": TX.oak_log_top()},
        LEAVES: TX.oak_leaves(),
        PLANKS: TX.oak_planks(),
        OBSIDIAN: TX.obsidian(),
    }


def portal_faces():
    """渦は黒曜石のわくの厚みの真ん中に、両面の光る板として置く。"""
    x0, x1, y0, y1 = PORTAL_IN
    z = PORTAL_Z + 0.5
    return [Face(np.array([x0, y1, z], float), np.array([x1 - x0, 0, 0], float), np.array([0, -(y1 - y0), 0], float),
                 TX.portal(), emis=1.0, lit=False, double=True, rep=(x1 - x0, y1 - y0))]


# ---------------------------------------------------------------- シーン


class Scene:
    def __init__(self, w: int, h: int):
        # 顔を大きく（クリックされやすいサムネは顔が主役）、うしろにゲートが全部入る位置
        self.cam = Camera((0.6, 1.4, 2.0), (0.6, 2.0, -2.0), 52, w, h)
        pose = Pose(yaw=-22, lean=4, head_yaw=30, head_pitch=4, head_roll=-4,
                    arm_r=(10, 8), arm_l=(-12, 88), leg_r=-6, leg_l=6)   # 左手でゲートを指さす
        self.player_faces, self.pinfo = player(skin_textures(load_skin(SKIN)), None, None, (0, 0, 0), pose)
        x0, x1, y0, y1 = PORTAL_IN
        self.portal_c = np.array([(x0 + x1) / 2, (y0 + y1) / 2, PORTAL_Z + 0.5])
        sun_col = (1.15, 0.62, 0.32)
        # キャラ用：顔を照らすやわらかい光＋夕日の逆光＋ゲートの紫のふちどり
        self.hero = Lights(
            ambient=(0.26, 0.22, 0.34),
            dirs=[((-0.35, 0.45, 1.0), (0.85, 0.78, 0.8)), (tuple(SUN_DIR), sun_col),
                  ((0.9, 0.2, -0.6), (0.75, 0.3, 1.25))])
        # 地形用：夕日と空の色＋ゲートの紫
        self.lights = Lights(
            ambient=(0.24, 0.2, 0.34),
            dirs=[(tuple(SUN_DIR), sun_col), ((0.3, 1.0, 0.4), (0.22, 0.2, 0.3))],
            points=[(self.portal_c + np.array([0, 0, 0.6]), (1.6, 0.55, 2.6), 3.2)],
            blobs=[(0.0, 0.1, 0.6, 0.5)])
        for f in self.player_faces:
            f.lights = self.hero
        self.world = build_world()
        self.world_faces = self.world.faces(block_textures(), near_z=-8) + portal_faces()

    def render(self):
        return render(self.player_faces + self.world_faces, self.cam, self.lights)


# ---------------------------------------------------------------- 空・霧・つぶ


def sky_and_fog(color, depth, cam: Camera, ss: int):
    """夕焼けの空（下がオレンジ、上が紺）と四角い夕日、星。遠くはオレンジのもやに。"""
    h, w = depth.shape
    xs = (np.arange(w) + 0.5 - cam.cx) / cam.fpx
    ys = -(np.arange(h) + 0.5 - cam.cy) / cam.fpx
    d = (cam.R.T @ np.stack(np.broadcast_arrays(xs[None, :], ys[:, None], np.ones((h, w))), 0).reshape(3, -1))
    d = (d / np.linalg.norm(d, axis=0)).reshape(3, h, w)
    el = d[1]
    stops = [(-0.1, (1.25, 0.62, 0.28)), (0.05, (1.15, 0.5, 0.35)), (0.18, (0.85, 0.36, 0.5)),
             (0.36, (0.38, 0.22, 0.55)), (0.6, (0.1, 0.09, 0.28))]
    pos = np.array([p for p, _ in stops])
    sky = np.stack([np.interp(el, pos, [c[i] for _, c in stops]) for i in range(3)], -1)
    # 夕日（四角）
    sun = SUN_DIR / np.linalg.norm(SUN_DIR)
    sx, sy, sz = cam.project(cam.pos + sun * 500)
    if sz > 0:
        half = 46 * ss
        yy, xx = np.mgrid[0:h, 0:w]
        m = (np.abs(xx - sx) < half) & (np.abs(yy - sy) < half)
        sky[m] = (2.4, 1.6, 0.8)
        r = np.hypot(xx - sx, yy - sy) / (220 * ss)
        sky += np.exp(-r * r)[..., None] * np.array([0.8, 0.35, 0.12])
    # 星
    rng = np.random.default_rng(5)
    for _ in range(90):
        x, y = int(rng.uniform(0, w)), int(rng.uniform(0, h * 0.35))
        s = max(1, int(ss * rng.choice([2, 2, 3])))
        if el[min(y, h - 1), min(x, w - 1)] > 0.35:
            sky[y:y + s, x:x + s] += rng.uniform(0.4, 0.9)
    hit = np.isfinite(depth)
    D = np.where(hit, depth, 500.0)
    haze = np.array([1.05, 0.55, 0.38])
    k = (1 - np.exp(-np.clip(D - 10, 0, None) * 0.016))[..., None]
    out = np.where(hit[..., None], color * (1 - k) + haze * k, sky)
    return out.astype(np.float32)


def portal_particles(img, cam: Camera, ss: int, seed=11):
    """ゲートのまわりを漂う紫のつぶ（ドット）。"""
    rng = np.random.default_rng(seed)
    layer = np.zeros_like(img)
    x0, x1, y0, y1 = PORTAL_IN
    for _ in range(90):
        p = np.array([rng.uniform(x0 - 1.2, x1 + 1.2), rng.uniform(y0 - 0.6, y1 + 1.2),
                      PORTAL_Z + 0.5 + rng.uniform(-0.2, 1.6)])
        sx, sy, sz = cam.project(p)
        if sz <= 0:
            continue
        s = max(2, int(rng.choice([3, 4, 5, 6]) * ss * 3.0 / sz))
        x, y = int(sx), int(sy)
        if 0 <= x < img.shape[1] - s and 0 <= y < img.shape[0] - s:
            c = np.array([0.9, 0.4, 1.6]) if rng.random() < 0.7 else np.array([1.2, 0.9, 1.8])
            layer[y:y + s, x:x + s] += c * rng.uniform(0.6, 1.1)
    img += layer + blur(layer, 3 * ss) * 1.2


# ---------------------------------------------------------------- 文字入れ

TEXT = {
    "hook1": "この先、",
    "hook2": "地獄。",
}


def compose(art: Image.Image, S: int) -> Image.Image:
    """クリックされやすさ重視：文字はフック（この先、地獄。）だけにして大きく。"""
    c = art.convert("RGBA")
    dark = ov.DARK
    # 下の方を少し暗くして、大きい文字を読みやすく
    h = c.height
    grad = np.clip((np.arange(h) / h - 0.62) / 0.38, 0, 1) ** 1.4 * 150
    band = np.zeros((h, c.width, 4), np.uint8)
    band[..., 3] = grad.astype(np.uint8)[:, None]
    c.alpha_composite(Image.fromarray(band, "RGBA"))
    # フック（大きく）
    hb1 = ov.fx_text(c, (26 * S, 548 * S), TEXT["hook1"], ov.font(fonts.BLACK, 104 * S), (255, 255, 255, 255),
                     strokes=[(13 * S, dark)], shadow=(6 * S, 8 * S, (40, 0, 30, 220), 3 * S), spacing=-6 * S)
    ov.fx_text(c, (hb1[2] - 14 * S, 512 * S), TEXT["hook2"], ov.font(fonts.BLACK, 150 * S),
               [(0, (255, 250, 170)), (0.4, (255, 170, 40)), (1, (230, 30, 30))],
               strokes=[(17 * S, dark), (8 * S, (120, 0, 40, 255))], shadow=(6 * S, 8 * S, (40, 0, 30, 220), 3 * S),
               spacing=-8 * S, rotate=-3)
    # ゲートの渦を指す矢印
    ov.arrow(c, (812 * S, 600 * S), (930 * S, 470 * S), width=34 * S, head_len=62 * S,
             head_w=96 * S, stroke_w=7 * S, outline_w=5 * S)
    # 汗（顔の左側）
    for (sx, sy, ang, sc) in [(168 * S, 250 * S, 30, 6), (140 * S, 330 * S, 60, 5)]:
        c.alpha_composite(ov.sweat(sc * S, ang), (sx, sy))
    return c


def render_art(ss: int = 2) -> Image.Image:
    sc = Scene(W * ss, H * ss)
    color, depth, wy = sc.render()
    img = sky_and_fog(color, depth, sc.cam, ss)
    portal_particles(img, sc.cam, ss)
    img = depth_of_field(img, depth, near=12, far=45, sigma=2.0 * ss)
    img = bloom(img, strength=0.5, threshold=0.85)
    img = grade(tonemap(img), vignette=0.32, saturation=1.1)
    return Image.fromarray((img * 255 + 0.5).astype(np.uint8))
