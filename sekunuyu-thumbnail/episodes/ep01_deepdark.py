"""#1：ディープダーク（たいまつを持って、うしろから迫るウォーデンから逃げる）。"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

from lib import fonts
from lib import overlay as ov
from lib import textures as TX
from lib.models import PX, Pose, player, warden
from lib.post import bloom, blur, depth_of_field, grade, tonemap
from lib.skin import CHANNEL_SKIN, load_skin, skin_textures
from lib.voxel import Camera, Lights, box_faces, render
from lib.world3d import DEEPSLATE, DTILES, SCULK, VoxelWorld

NAME = "thumbnail_01"
W, H = 1280, 720
WARDEN_POS = np.array([2.3, 0.0, -3.1])
SOUL_LIGHTS = [(-9.0, 5.5, -14.8), (5.0, 5.5, -14.8), (12.0, 5.5, -14.8)]   # 奥のやみをうっすら照らす水色の光


# ---------------------------------------------------------------- 地形


def build_world(seed: int = 3) -> VoxelWorld:
    rng = np.random.default_rng(seed)
    w = VoxelWorld(-40, -6, -60, 80, 30, 70)
    w.box(-40, 40, -6, 0, -60, 10, DEEPSLATE)
    # 床のスカルク（ウォーデンのまわりに多い）とでこぼこ
    for x in range(-40, 40):
        for z in range(-60, 10):
            d = math.hypot(x - WARDEN_POS[0], z - WARDEN_POS[2])
            n = math.sin(x * 0.55 + z * 0.3) + math.sin(x * 0.21 - z * 0.47) + rng.uniform(-0.6, 0.6)
            if n + 1.6 * math.exp(-d / 5) > 0.9:
                w.box(x, x + 1, -1, 0, z, z + 1, SCULK)
            if rng.random() < 0.05 and abs(x) > 2 and z < -5:
                w.box(x, x + 1, 0, 1, z, z + 1, SCULK if rng.random() < 0.5 else DEEPSLATE)
    # 奥：古代都市のかべ（タイル）と柱
    w.box(-40, 40, 0, 13, -20, -18, DTILES)
    for x in range(-30, 31, 7):
        w.box(x, x + 2, 0, 14, -18, -16, DEEPSLATE)
    w.box(-1, 5, 0, 7, -20, -18, 0)                     # 奥の入り口（暗い）
    # 左の岩かべ（スカルクがはっている）
    for z in range(-20, 8):
        depth = 8 + int(1.5 * math.sin(z * 0.6) + rng.integers(0, 2))
        top = 9 + int(3 * math.sin(z * 0.37))
        w.box(-40, -depth, 0, top, z, z + 1, DEEPSLATE)
        if rng.random() < 0.45:
            w.box(-depth - 1, -depth, rng.integers(0, 4), rng.integers(4, 8), z, z + 1, SCULK)
    return w


def block_textures():
    return {DEEPSLATE: [TX.deepslate(31), TX.deepslate(35), TX.deepslate(36)], DTILES: TX.deepslate_tiles(),
            SCULK: [TX.sculk(33), TX.sculk(37)]}


# ---------------------------------------------------------------- たいまつ


def torch_faces(info, scale=1.0):
    """右手に持ったたいまつ（棒と、光るほのお）。戻り値：面のリストと、ほのおの位置。"""
    Ma, ta = info["arm_r"]
    k = PX * scale
    c = TX.TORCH_COLORS
    stick = {f: TX.solid(c["stick_d"] if f in ("left", "right") else c["stick"]) for f in
             ("front", "back", "left", "right", "top", "bottom")}
    faces = box_faces(np.array([2, 2, 7]) * k, stick, M=Ma, t=ta, origin=np.array([-1, -10, -1]) * k)
    flame = {f: TX.solid(c["flame"] if f in ("front", "top") else c["flame_o"]) for f in
             ("front", "back", "left", "right", "top")}
    fl = box_faces(np.array([2.4, 2.4, 2.6]) * k, flame, M=Ma, t=ta, origin=np.array([-1.2, -10.2, 6]) * k,
                   emis=1.6, lit=False)
    return faces + fl, ta + Ma @ (np.array([0, -9, 7.5]) * k)



# ---------------------------------------------------------------- シーン


class Scene:
    def __init__(self, w: int, h: int):
        # キャラは手前で大きく。うしろ右から、光るウォーデンが迫ってくる
        self.cam = Camera((0.62, 1.3, 2.75), (0.95, 1.85, -2.0), 52, w, h)
        pose = Pose(yaw=-24, lean=10, head_yaw=44, head_pitch=4, head_roll=-5,
                    arm_r=(-104, -6), arm_l=(36, 8), leg_r=30, leg_l=-34)
        self.player_faces, info = player(skin_textures(load_skin(CHANNEL_SKIN)), None, None, (0, 0, 0), pose)
        tf, self.flame = torch_faces(info)
        self.player_faces += tf
        to_p = -WARDEN_POS
        yaw = math.degrees(math.atan2(to_p[0], to_p[2]))
        self.warden_faces, self.winfo = warden(TX.warden_textures(), WARDEN_POS, yaw=yaw, lean=10, head_pitch=-6,
                                               arm_r=-62, arm_l=-28, leg_r=-10, leg_l=12, scale=1.15)
        torch = (self.flame, (1.3, 0.72, 0.3), 3.6)
        soul = (self.winfo["chest"] + np.array([0, 0, 0.6]), (0.3, 1.4, 1.5), 2.6)
        # キャラ用：たいまつの暖かい光＋うしろのウォーデンからの水色のふちどり
        self.hero = Lights(ambient=(0.1, 0.12, 0.17),
                           dirs=[((0.05, 0.35, 1.0), (0.42, 0.42, 0.5)), ((0.8, 0.4, -0.6), (0.25, 0.85, 0.95))],
                           points=[torch])
        # ウォーデン用：暗い体に、水色のふちどりと、たいまつの光が少し
        self.wlights = Lights(ambient=(0.07, 0.1, 0.13),
                              dirs=[((0.6, 0.6, -0.8), (0.35, 1.1, 1.25)), ((-0.4, 0.3, 1.0), (0.14, 0.15, 0.19))],
                              points=[torch])
        # 地形用：とても暗い青。たいまつとウォーデンの胸のまわりだけ照らす
        souls = [(np.array(p), (0.25, 1.0, 1.1), 4.0) for p in SOUL_LIGHTS]
        self.lights = Lights(ambient=(0.05, 0.08, 0.11), dirs=[((0.2, 1.0, 0.3), (0.06, 0.12, 0.15))],
                             points=[torch, soul] + souls)
        for f in self.player_faces:
            f.lights = self.hero
        for f in self.warden_faces:
            f.lights = self.wlights
        self.world_faces = build_world().faces(block_textures(), near_z=-8)

    def render(self):
        return render(self.player_faces + self.warden_faces + self.world_faces, self.cam, self.lights)


# ---------------------------------------------------------------- 霧・つぶ


def fog(color, depth, ss):
    """遠くは青黒いやみに消えていく。背景（何もない所）はほぼ黒。"""
    h, w = depth.shape
    hit = np.isfinite(depth)
    D = np.where(hit, depth, 200.0)
    dark = np.array([0.012, 0.03, 0.045])
    k = (1 - np.exp(-np.clip(D - 5, 0, None) * 0.075))[..., None]
    yy = np.linspace(0, 1, h)[:, None, None]
    bg = dark * (0.6 + 0.8 * yy)
    out = np.where(hit[..., None], color * (1 - k) + dark * k, bg)
    return out.astype(np.float32)


def backglow(img, depth, cam: Camera, sc: Scene, ss: int):
    """ウォーデンのうしろのやみを水色にぼんやり光らせて、形を浮かび上がらせる。"""
    sx, sy, sz = cam.project(sc.winfo["chest"] + np.array([0, 0.5, 0]))
    h, w = depth.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((xx - sx) / (260 * ss), (yy - sy) / (330 * ss))
    behind = (~np.isfinite(depth)) | (depth > sz + 1.5)
    img += (np.exp(-r * r) * behind)[..., None] * np.array([0.05, 0.22, 0.26])


def particles(img, cam: Camera, sc: Scene, ss: int, seed=21):
    """ウォーデンのまわりを漂う水色のたましいのつぶと、たいまつの火の粉。"""
    rng = np.random.default_rng(seed)
    layer = np.zeros_like(img)
    pts = [(WARDEN_POS + np.array([rng.uniform(-2.5, 2.5), rng.uniform(0.2, 4.0), rng.uniform(-1.5, 2.0)]),
            np.array([0.3, 1.5, 1.6]), (3, 4, 5)) for _ in range(70)]
    pts += [(sc.flame + np.array([rng.uniform(-0.3, 0.3), rng.uniform(0.05, 0.8), rng.uniform(-0.3, 0.3)]),
             np.array([2.2, 1.1, 0.3]), (2, 3)) for _ in range(18)]
    for p, c, sizes in pts:
        sx, sy, sz = cam.project(p)
        if sz <= 0:
            continue
        s = max(2, int(rng.choice(sizes) * ss * 3.0 / sz))
        x, y = int(sx), int(sy)
        if 0 <= x < img.shape[1] - s and 0 <= y < img.shape[0] - s:
            layer[y:y + s, x:x + s] += c * rng.uniform(0.5, 1.1)
    img += layer + blur(layer, 3 * ss) * 1.3


# ---------------------------------------------------------------- 文字入れ

TEXT = {
    "catch1": "死んだら、",
    "catch2": "終わり。",
}


def compose(art: Image.Image, S: int) -> Image.Image:
    """文字は真ん中下のキャッチだけ（左上はロゴなどを入れられるようにあける）。#2・#3 と同じ作り。"""
    c = art.convert("RGBA")
    dark = ov.DARK
    h = c.height
    grad = np.clip((np.arange(h) / h - 0.62) / 0.38, 0, 1) ** 1.4 * 150
    band = np.zeros((h, c.width, 4), np.uint8)
    band[..., 3] = grad.astype(np.uint8)[:, None]
    c.alpha_composite(Image.fromarray(band, "RGBA"))
    # 飛び散る汗（頭の右側＝ウォーデンの方）
    for (sx, sy, ang, sc) in [(492, 238, -32, 5), (520, 300, -62, 4)]:
        c.alpha_composite(ov.sweat(sc * S, ang), (sx * S, sy * S))
    # 体力ハート（右上。1日目なので満タン）
    ov.hearts_bar(c, S, ["full"] * 10, x=1280 - 26 - 9 * 44 - 42, y=24)
    # 真ん中下：キャッチ（#2・#3 と同じ色。文字数が多いので少し小さめ）
    layer = Image.new("RGBA", c.size, (0, 0, 0, 0))
    shadow = (6 * S, 8 * S, (40, 0, 30, 220), 3 * S)
    b1 = ov.fx_text(layer, (200 * S, 560 * S), TEXT["catch1"], ov.font(fonts.BLACK, 94 * S), (255, 255, 255, 255),
                    strokes=[(12 * S, dark)], shadow=shadow, spacing=-6 * S)
    ov.fx_text(layer, (b1[2] - 14 * S, 520 * S), TEXT["catch2"], ov.font(fonts.BLACK, 140 * S),
               [(0, (255, 250, 170)), (0.4, (255, 170, 40)), (1, (230, 30, 30))],
               strokes=[(16 * S, dark), (8 * S, (120, 0, 40, 255))], shadow=shadow, spacing=-8 * S, rotate=-3)
    bb = layer.getbbox()
    c.alpha_composite(layer.crop((bb[0], 0, bb[2], c.height)), ((c.width - (bb[2] - bb[0])) // 2, 0))
    return c


def render_art(ss: int = 2) -> Image.Image:
    sc = Scene(W * ss, H * ss)
    color, depth, _ = sc.render()
    img = fog(color, depth, ss)
    backglow(img, depth, sc.cam, sc, ss)
    particles(img, sc.cam, sc, ss)
    img = depth_of_field(img, depth, near=6, far=26, sigma=2.2 * ss)
    img = bloom(img, strength=0.65, threshold=0.7)
    img = grade(tonemap(img), vignette=0.42, saturation=1.12)
    return Image.fromarray((img * 255 + 0.5).astype(np.uint8))
