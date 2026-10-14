"""#2：ネザー探検（ガストの火の玉から逃げる）。"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from lib import fonts
from lib import overlay as ov
from lib import textures as TX
from lib.models import Pose, ghast, player
from lib.post import bloom, blur, depth_of_field, grade, smoothstep, tonemap
from lib.skin import CHANNEL_SKIN, load_skin, skin_textures
from lib.voxel import Camera, Face, Lights, render
from lib.world3d import BRICK, FALL, GLOW, LAVA, MAGMA, QUARTZ, RACK, VoxelWorld

NAME = "thumbnail_02"
W, H = 1280, 720


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
        # キャラはチャンネルのマイクラスキン（エンダーパーカー）をそのまま着せる
        self.player_faces, self.pinfo = player(skin_textures(load_skin(CHANNEL_SKIN)), None, None, (0, 0, 0), pose)
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
    "catch1": "ネザー、",
    "catch2": "無理。",
}


def compose(art: Image.Image, S: int) -> Image.Image:
    """S 倍の大きさで文字・ハート・汗を描く。文字は真ん中下のキャッチだけ（左上はロゴなどを入れられるようにあける）。"""
    c = art.convert("RGBA")
    dark = ov.DARK
    # 下の方を少し暗くして、大きい文字を読みやすく
    h = c.height
    grad = np.clip((np.arange(h) / h - 0.62) / 0.38, 0, 1) ** 1.4 * 150
    band = np.zeros((h, c.width, 4), np.uint8)
    band[..., 3] = grad.astype(np.uint8)[:, None]
    c.alpha_composite(Image.fromarray(band, "RGBA"))
    # 飛び散る汗（頭の左側）
    for (sx, sy, ang, sc) in [(478, 300, 35, 5), (446, 362, 62, 4), (470, 426, 88, 4)]:
        c.alpha_composite(ov.sweat(sc * S, ang), (sx * S, sy * S))
    # 体力ハート（右上。残り1.5個で大ピンチ）
    ov.hearts_bar(c, S, ["full", "half"] + ["empty"] * 8, x=1280 - 26 - 9 * 44 - 42, y=24)
    # 真ん中下：キャッチ（#3 と同じ大きさ・色）。いったん別の層に描いて、左右のまん中にそろえる
    layer = Image.new("RGBA", c.size, (0, 0, 0, 0))
    shadow = (6 * S, 8 * S, (40, 0, 30, 220), 3 * S)
    b1 = ov.fx_text(layer, (200 * S, 548 * S), TEXT["catch1"], ov.font(fonts.BLACK, 104 * S), (255, 255, 255, 255),
                    strokes=[(13 * S, dark)], shadow=shadow, spacing=-6 * S)
    ov.fx_text(layer, (b1[2] - 14 * S, 512 * S), TEXT["catch2"], ov.font(fonts.BLACK, 150 * S),
               [(0, (255, 250, 170)), (0.4, (255, 170, 40)), (1, (230, 30, 30))],
               strokes=[(17 * S, dark), (8 * S, (120, 0, 40, 255))], shadow=shadow, spacing=-8 * S, rotate=-3)
    bb = layer.getbbox()
    c.alpha_composite(layer.crop((bb[0], 0, bb[2], c.height)), ((c.width - (bb[2] - bb[0])) // 2, 0))
    return c


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
