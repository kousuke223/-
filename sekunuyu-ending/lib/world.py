"""背景まわり：ドット絵のワールドマップ、青空と草原、雲。

どれも 320x180 の低解像度グリッドに描き、最後に 6 倍して 1920x1080 にする。
"""
from __future__ import annotations

import math

import numpy as np

from .pixelart import (blank, blit, dilate, ellipse_mask, from_strings, hexc, line_mask, pad,
                       poly_mask, shift)

GW, GH = 320, 180

C = {k: hexc(v) for k, v in {
    "out": "#2A1E2E",
    "sea1": "#2B63C6", "sea2": "#3C86E6", "sea3": "#8FD0FF", "foam": "#EAF8FF",
    "sand": "#F7DE9C", "sand2": "#E6C27C",
    "gr0": "#95E36A", "gr1": "#78D35A", "gr2": "#5DBF4B", "gr3": "#47A441", "gr4": "#358B3A",
    "tl": "#6CD157", "tm": "#43A844", "td": "#2A7D39", "th": "#B5F08A",
    "trunk": "#8A5A2B", "trunk2": "#5E3A1C",
    "path": "#E8BF74", "path2": "#C99A52",
    "rock1": "#C4BDB2", "rock2": "#968D82", "rock3": "#6A625A",
    "snow": "#FFFFFF", "snow2": "#D5E6F7",
    "lava1": "#FF5A1F", "lava2": "#FFC23A", "volc1": "#9A5A44", "volc2": "#6E3B2E",
    "roofr": "#EF5A45", "roofr2": "#B23A30", "roofb": "#4F86EA", "roofb2": "#2F58B4",
    "roofy": "#FFC43D", "roofy2": "#D98A1E",
    "wall": "#FFF3D1", "wall2": "#E9CF9C", "glass": "#8FD0FF", "door": "#8A5A2B",
    "st1": "#D7DCE6", "st2": "#A9B1C1", "st3": "#78839A",
    "wood": "#C27A3A", "wood2": "#86491D", "wood3": "#E0A060", "gold": "#FFD23F",
    "yel": "#FFD23F", "yel2": "#F2A516", "white": "#FFFFFF", "pink": "#FF8FB1",
    "cloud": "#FFFFFF", "cloud2": "#D2E5F8", "cloud3": "#93BCE8",
    "sky0": "#3E97F2", "sky1": "#5AAAF6", "sky2": "#79BEF9", "sky3": "#9BD1FB", "sky4": "#BFE3FD",
    "hill1": "#A6E07E", "hill2": "#8ACF6A",
}.items()}

BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16.0


def bayer(h, w):
    return np.tile(BAYER4, (h // 4 + 1, w // 4 + 1))[:h, :w]


def value_noise(h, w, cell, seed):
    rng = np.random.default_rng(seed)
    g = rng.random((h // cell + 3, w // cell + 3))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) / cell
    x0, y0 = xx.astype(int), yy.astype(int)
    fx, fy = xx - x0, yy - y0
    sx, sy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a, b = g[y0, x0], g[y0, x0 + 1]
    c, d = g[y0 + 1, x0], g[y0 + 1, x0 + 1]
    return (a * (1 - sx) + b * sx) * (1 - sy) + (c * (1 - sx) + d * sx) * sy


def fbm(h, w, seed, cells=(40, 20, 10), weights=(0.55, 0.3, 0.15)):
    return sum(wt * value_noise(h, w, c, seed + i) for i, (c, wt) in enumerate(zip(cells, weights)))


# ---------------------------------------------------------------- スプライト

P = {
    "k": C["out"], "L": C["tl"], "M": C["tm"], "D": C["td"], "h": C["th"],
    "b": C["trunk"], "B": C["trunk2"], "W": C["wall"], "w": C["wall2"], "g": C["glass"],
    "d": C["door"], "s": C["rock1"], "S": C["rock2"], "y": C["yel"], "o": C["yel2"],
    "x": C["wood"], "X": C["wood2"], "z": C["wood3"], "G": C["gold"], "e": C["white"],
}

TREE = [
    "....kkkk....",
    "..kkLLLLkk..",
    ".kLhLLLLLMk.",
    "kLhLLLLLMMMk",
    "kLLLLLMMMMMk",
    "kLLMMMMMMMDk",
    "kMMMMMMMMDDk",
    ".kMMMMMDDDk.",
    "..kkDDDDkk..",
    "....kbBk....",
    "....kbBk....",
    "....kkkk....",
]
TREE_S = [
    "..kkkkk..",
    ".kLLLLMk.",
    "kLhLLMMMk",
    "kLLMMMMDk",
    "kMMMMMDDk",
    ".kkMDDkk.",
    "...kbk...",
    "...kbk...",
    "...kkk...",
]
BUSH = [
    ".kkkkk.",
    "kLLLMMk",
    "kLMMMDk",
    "kMMMDDk",
    ".kkkkk.",
]
HOUSE = [
    "....kkkkk....",
    "...krrrrrk...",
    "..krrrrrrRk..",
    ".krrrrrrRRRk.",
    "kkkkkkkkkkkkk",
    ".kWWWWWWWWwk.",
    ".kWgWkkkWgwk.",
    ".kWgWkdkWgwk.",
    ".kWWWkdkWWwk.",
    ".kkkkkkkkkkk.",
]
ROCK = [
    ".kkkk.",
    "kssSSk",
    "ksSSSk",
    ".kkkk.",
]
CHEST = [
    ".kkkkkkkkk.",
    "kzzzzzzzzXk",
    "kxxxxxxxxXk",
    "kkkkkGGkkkk",
    "kxxxkGGkxXk",
    "kxxxxkkxxXk",
    "kxxxxxxxxXk",
    "kkkkkkkkkkk",
]
SIGN = [
    "kkkkkkkkk",
    "kzzzzzzzk",
    "kzXXzXXzk",
    "kzzzzzzzk",
    "kkkkkkkkk",
    "...kxk...",
    "...kxk...",
    "...kkk...",
]
FLOWER_Y = [".y.", "yoy", ".y."]
FLOWER_W = [".e.", "eee", ".e."]


def house_sprite(roof="r"):
    pal = dict(P)
    if roof == "r":
        pal["r"], pal["R"] = C["roofr"], C["roofr2"]
    elif roof == "b":
        pal["r"], pal["R"] = C["roofb"], C["roofb2"]
    else:
        pal["r"], pal["R"] = C["roofy"], C["roofy2"]
    return from_strings(HOUSE, pal)


def mountain_sprite(w, h, snow=True, seed=0):
    rng = np.random.default_rng(seed)
    sw, sh = w + 2, h + 2
    ax = w / 2 + rng.uniform(-w * 0.08, w * 0.08)
    pts = [(1, sh - 1)]
    # 左斜面（少しゴツゴツ）
    for i in range(1, 5):
        f = i / 5
        pts.append((1 + (ax - 1) * f + rng.uniform(-1, 1), sh - 1 - (sh - 3) * f + rng.uniform(-1, 1)))
    pts.append((ax + 1, 1))
    for i in range(1, 5):
        f = i / 5
        pts.append((ax + 1 + (w - ax) * f + rng.uniform(-1, 1), 1 + (sh - 3) * f + rng.uniform(-1, 1)))
    pts.append((sw - 1, sh - 1))
    m = poly_mask(sh, sw, pts)
    spr = blank(sw, sh)
    yy, xx = np.mgrid[0:sh, 0:sw]
    ridge_x = ax + 1 + (yy - 1) * 0.18
    spr[m] = C["rock1"]
    spr[m & (xx > ridge_x)] = C["rock2"]
    # 斜面の筋
    streak = m & ((xx + yy * 2) % 7 == 0) & (yy > h * 0.35)
    spr[streak & (xx <= ridge_x)] = C["rock2"]
    spr[streak & (xx > ridge_x)] = C["rock3"]
    if snow:
        zig = 0.34 * h + np.where((xx // 3) % 2 == 0, 0, 2.5)
        cap = m & (yy < zig)
        spr[cap] = C["snow"]
        spr[cap & (xx > ridge_x)] = C["snow2"]
    ring = dilate(m, 1, diag=False) & ~m
    spr[ring] = C["out"]
    return spr


def volcano_sprite(w=62, h=46):
    sw, sh = w + 2, h + 2
    cx = sw / 2
    top_w = 16
    pts = [(1, sh - 1), (cx - top_w / 2 - 3, 9), (cx - top_w / 2, 5), (cx + top_w / 2, 5), (cx + top_w / 2 + 3, 9),
           (sw - 1, sh - 1)]
    m = poly_mask(sh, sw, pts)
    spr = blank(sw, sh)
    yy, xx = np.mgrid[0:sh, 0:sw]
    spr[m] = C["volc1"]
    spr[m & (xx > cx + (yy - 5) * 0.12)] = C["volc2"]
    streak = m & ((xx - yy * 2) % 9 == 0) & (yy > 14)
    spr[streak] = C["volc2"]
    crater = ellipse_mask(sh, sw, cx, 7, top_w / 2 - 1, 2.6)
    spr[crater] = C["lava1"]
    spr[crater & ellipse_mask(sh, sw, cx, 7, top_w / 2 - 4, 1.4)] = C["lava2"]
    # 溶岩の流れ
    for pts_l in ([(cx - 3, 8), (cx - 6, 16), (cx - 5, 24), (cx - 11, 33), (cx - 13, 42)],
                  [(cx + 4, 8), (cx + 6, 15), (cx + 12, 22), (cx + 13, 30)]):
        lm = line_mask(sh, sw, pts_l, width=3) & m
        spr[lm] = C["lava1"]
        lc = line_mask(sh, sw, pts_l, width=1) & m
        spr[lc] = C["lava2"]
    ring = dilate(m, 1, diag=False) & ~m
    spr[ring] = C["out"]
    return spr


def castle_sprite():
    w, h = 48, 44
    spr = blank(w, h)
    m_all = np.zeros((h, w), bool)
    yy, xx = np.mgrid[0:h, 0:w]

    def rect(x0, y0, x1, y1):
        return (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)

    def stone(mask, light=True):
        spr[mask] = C["st1"] if light else C["st2"]
        brick = mask & (((yy % 4) == 0) | ((((xx + (yy // 4) * 3) % 6) == 0) & ((yy % 4) != 0)))
        spr[brick] = C["st2"] if light else C["st3"]

    # 城壁
    wall = rect(6, 22, 42, 43)
    merl = rect(6, 19, 42, 22) & (((xx - 6) // 3) % 2 == 0)
    stone(wall | merl)
    m_all |= wall | merl
    # 中央の塔
    keep = rect(17, 10, 31, 30)
    stone(keep, light=False)
    stone(keep & (xx < 24))
    m_all |= keep
    # 左右の塔
    for x0 in (1, 37):
        tw = rect(x0, 14, x0 + 10, 43)
        stone(tw)
        stone(tw & (xx >= x0 + 6), light=False)
        m_all |= tw
        roof = poly_mask(h, w, [(x0 - 1, 15), (x0 + 5, 2), (x0 + 11, 15)])
        spr[roof] = C["roofb"]
        spr[roof & (xx >= x0 + 5)] = C["roofb2"]
        m_all |= roof
        spr[rect(x0 + 4, 22, x0 + 6, 26)] = C["out"]
    roofk = poly_mask(h, w, [(15, 11), (24, -2), (33, 11)])
    spr[roofk] = C["roofr"]
    spr[roofk & (xx >= 24)] = C["roofr2"]
    m_all |= roofk
    # 窓と門
    spr[rect(22, 15, 26, 20)] = C["out"]
    spr[rect(23, 16, 25, 19)] = C["lava2"]
    gate = rect(19, 33, 29, 43) | ellipse_mask(h, w, 24, 33.5, 5, 3.5)
    spr[gate] = C["out"]
    spr[gate & ((xx % 2) == 0) & (yy > 32)] = C["trunk2"]
    ring = dilate(m_all, 1, diag=False) & ~m_all
    spr[ring] = C["out"]
    return pad(spr, 1)


def flag_sprite(phase: int):
    rows = ["kyyk.", "kyyyk", "kyyk.", "k....", "k...."] if phase == 0 else ["kyk..", "kyyyk", "kyyyk", "k....", "k...."]
    return from_strings(rows, {"k": C["out"], "y": C["yel"]})


# ---------------------------------------------------------------- ワールドマップ


class WorldMap:
    """参考動画の「ゲームのワールドマップ」風の背景（オリジナル）。"""

    def __init__(self, seed: int = 3):
        self.seed = seed
        rng = np.random.default_rng(seed)
        h, w = GH, GW
        yy, xx = np.mgrid[0:h, 0:w]
        n = fbm(h, w, seed)
        d = bayer(h, w)
        ground = blank(w, h)
        # 草地（ノイズ＋ディザで濃淡）
        v = n + (d - 0.5) * 0.08
        ground[...] = C["gr2"]
        ground[v > 0.58] = C["gr3"]
        ground[v < 0.40] = C["gr1"]
        ground[v < 0.30] = C["gr0"]
        tuft = (rng.random((h, w)) < 0.018)
        ground[tuft & (v >= 0.40)] = C["gr3"]
        ground[shift(tuft, 1, -1) & (v >= 0.40)] = C["gr4"]

        # 海（左下）
        coast = 120 + xx * 0.42 + (fbm(h, w, seed + 10, cells=(18, 9, 5)) - 0.5) * 16
        sea = yy > coast
        dist = yy - coast
        self.sea = sea
        beach = ~sea & dilate(sea, 3, diag=True)
        ground[beach] = C["sand"]
        ground[beach & ~dilate(sea, 2, diag=True)] = C["sand2"]
        ground[sea] = C["sea2"]
        ground[sea & (dist > 16)] = C["sea1"]
        ground[sea & (dist > 13) & (dist <= 16) & (d > 0.5)] = C["sea1"]

        # 川（山から海へ）
        river_pts = [(122, 30), (112, 40), (96, 50), (80, 60), (70, 74), (64, 90), (54, 106), (44, 120), (34, 136)]
        river = line_mask(h, w, river_pts, width=6) & ~sea
        bank = dilate(river, 1, diag=False) & ~river & ~sea
        ground[bank] = C["gr4"]
        ground[river] = C["sea2"]
        center = line_mask(h, w, river_pts, width=1) & river
        self.river_center = center
        self.river = river
        # 池（右下の森の手前）
        pond = ellipse_mask(h, w, 214, 158, 14, 7)
        ground[dilate(pond, 1, diag=False) & ~pond] = C["gr4"]
        ground[pond] = C["sea2"]
        self.pond = pond

        # 道
        paths = [
            [(40, 44), (52, 56), (68, 64), (90, 70), (120, 80), (160, 88), (200, 92), (236, 96), (262, 96)],
            [(160, 88), (158, 110), (156, 132), (150, 150), (140, 166)],
            [(276, 86), (278, 74), (270, 62)],
            [(200, 92), (212, 120), (226, 140), (246, 150)],
        ]
        pm = np.zeros((h, w), bool)
        for p in paths:
            pm |= line_mask(h, w, p, width=4)
        pm &= ~sea
        edge = dilate(pm, 1, diag=False) & ~pm & ~sea & ~river
        ground[edge & ~beach] = C["path2"]
        ground[pm] = C["path"]
        ground[pm & (rng.random((h, w)) < 0.12)] = C["path2"]
        # 橋（道と川が交わるところ）
        cross = pm & river
        bridge = dilate(cross, 2, diag=True) & (river | dilate(river, 1))
        ground[bridge] = C["wood3"]
        ground[bridge & ((xx + yy) % 3 == 0)] = C["wood"]
        ground[dilate(bridge, 1, diag=False) & ~bridge] = C["out"]
        self.ground = ground

        # ---- 物を置く（y 座標順に描く）
        items = []

        def put(spr, x, y):
            items.append((y + spr.shape[0], spr, int(x), int(y)))

        # 山脈（上中央）と火山（右上）
        for i, (x, y, mw, mh) in enumerate([(98, 6, 30, 26), (120, -2, 38, 34), (150, 4, 30, 26),
                                             (172, -4, 40, 36), (200, 6, 30, 24)]):
            put(mountain_sprite(mw, mh, seed=seed + i), x, y)
        self.volcano_xy = (242, 10)
        put(volcano_sprite(), *self.volcano_xy)
        # 村（左上）
        for i, (x, y, roof) in enumerate([(34, 22, "r"), (50, 18, "b"), (66, 24, "y"), (26, 36, "b"),
                                           (44, 38, "r"), (62, 40, "r")]):
            put(house_sprite(roof), x, y)
        # 城（右）
        self.castle_xy = (262, 58)
        put(castle_sprite(), *self.castle_xy)
        # 宝箱と看板（まわりに木を生やさない）
        self.chest_xy = (244, 140)
        put(from_strings(CHEST, P), *self.chest_xy)
        put(from_strings(SIGN, P), 164, 138)
        # 森（右下・右上・左）
        forest = [(self.chest_xy[0] - 2, self.chest_xy[1] - 4), (self.chest_xy[0] + 6, self.chest_xy[1] - 6)]
        for (x0, y0, x1, y1, cnt) in [(236, 108, 316, 150, 26), (290, 40, 318, 64, 6), (4, 50, 40, 100, 12),
                                      (82, 20, 104, 40, 5), (212, 28, 240, 52, 6)]:
            tries = 0
            placed = 0
            while placed < cnt and tries < 400:
                tries += 1
                x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
                if any((x - fx) ** 2 + (y - fy) ** 2 < 60 for fx, fy in forest):
                    continue
                cy, cx = int(min(h - 1, y + 10)), int(min(w - 1, x + 6))
                if sea[cy, cx] or river[cy, cx] or pm[cy, cx] or pond[cy, cx]:
                    continue
                forest.append((x, y))
                placed += 1
                put(from_strings(TREE if rng.random() < 0.7 else TREE_S, P), x, y)
        # 散らばった木・茂み・岩
        for x, y in [(84, 84), (100, 100), (70, 128), (118, 150), (182, 150), (196, 140), (226, 84),
                     (8, 108), (20, 118), (290, 156), (132, 44), (64, 44)]:
            put(from_strings(TREE_S if (x + y) % 3 else TREE, P), x, y)
        for x, y in [(96, 124), (176, 128), (250, 80), (78, 38), (30, 60), (206, 70), (122, 64)]:
            put(from_strings(BUSH, P), x, y)
        for x, y in [(110, 116), (236, 66), (190, 74), (8, 44), (300, 104)]:
            put(from_strings(ROCK, P), x, y)
        # 花（たんぽぽ）
        flowers = blank(w, h)
        for _ in range(140):
            x, y = int(rng.uniform(100, 230)), int(rng.uniform(130, 175))
            if sea[min(h - 1, y + 1), x] or pm[min(h - 1, y + 1), x] or pond[min(h - 1, y + 1), x]:
                continue
            blit(flowers, from_strings(FLOWER_Y if rng.random() < 0.7 else FLOWER_W, P), x, y)
        for _ in range(40):
            x, y = int(rng.uniform(10, 80)), int(rng.uniform(40, 60))
            if pm[min(h - 1, y + 1), x] or river[min(h - 1, y + 1), x]:
                continue
            blit(flowers, from_strings(FLOWER_Y if rng.random() < 0.5 else FLOWER_W, P), x, y)
        self.flowers = flowers
        items.sort(key=lambda it: it[0])
        objs = blank(w, h)
        for _, spr, x, y in items:
            # 影（右下に落ちる楕円）
            sh_w = spr.shape[1]
            shadow = ellipse_mask(h, w, x + sh_w / 2 + 1, y + spr.shape[0] - 0.5, sh_w / 2.2, 1.6)
            objs[shadow & (objs[..., 3] == 0)] = (0, 0, 0, 55)
            blit(objs, spr, x, y)
        self.objs = objs

        # 海のきらめき用の点
        self.sparkle_pts = [(int(x), int(y)) for x, y in zip(rng.uniform(0, w, 260), rng.uniform(0, h, 260))
                            if sea[int(y), int(x)] or pond[int(y), int(x)]]
        self.rng = rng

    def render(self, t: float) -> np.ndarray:
        img = self.ground.copy()
        h, w = GH, GW
        # 波のきらめき（3段階でゆっくり切り替え）
        phase = int(t * 4)
        for i, (x, y) in enumerate(self.sparkle_pts):
            if (i + phase) % 3 == 0:
                continue
            dx = (phase + i) % 2
            x0 = x + dx
            if 0 <= y < h and x0 + 2 < w:
                seg = img[y, x0:x0 + 3]
                ok = (self.sea[y, x0:x0 + 3] | self.pond[y, x0:x0 + 3])
                seg[ok] = C["sea3"]
        # 川の流れ
        rc = self.river_center & (((np.arange(w)[None, :] + np.arange(h)[:, None] - int(t * 12)) % 6) < 2)
        img[rc] = C["sea3"]
        # 波打ち際の白波
        foam = self.sea & dilate(~self.sea, 1, diag=False)
        if int(t * 3) % 2:
            foam = self.sea & dilate(~self.sea, 2, diag=False) & ~foam
        img[foam & ~self.river] = C["foam"]
        blit(img, self.flowers, 0, 0)
        blit(img, self.objs, 0, 0)
        # 旗
        fx, fy = self.castle_xy
        ph = int(t * 5) % 2
        for dx, dy in ((6, -4), (25, -8), (42, -4)):
            blit(img, flag_sprite(ph), fx + dx, fy + dy)
        # 火山の煙
        vx, vy = self.volcano_xy
        for i in sorted(range(5), key=lambda i: -((t * 0.5 + i / 5) % 1.0)):
            age = (t * 0.5 + i / 5) % 1.0
            r = 3 + age * 7
            cx = vx + 32 + math.sin(age * 5 + i) * 3 + age * 14
            cy = vy + 5 - age * 26
            m = ellipse_mask(h, w, cx, cy, r, r * 0.85)
            img[m] = C["st1"] if age < 0.55 else C["st2"]
            img[m & ~shift(m, 0, -2)] = C["st2"] if age < 0.55 else C["st3"]
        # 宝箱のキラッ
        if int(t * 2.5) % 3 == 0:
            cx, cy = self.chest_xy
            for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                img[cy - 3 + dy, cx + 12 + dx] = C["white"]
        return img


# ---------------------------------------------------------------- 雲




def draw_puffs(h, w, puffs, base: np.ndarray | None = None) -> np.ndarray:
    """もこもこ雲。1粒ずつ右下に影を付けて、下にある粒ほど手前に重ねる。"""
    out = blank(w, h)
    union = np.zeros((h, w), bool) if base is None else base.copy()
    if base is not None:
        out[base] = C["cloud"]
    yy, xx = np.mgrid[0:h, 0:w]
    for cx, cy, r in sorted(puffs, key=lambda p: p[1]):
        x0, x1 = max(0, int(cx - r - 1)), min(w, int(cx + r + 2))
        y0, y1 = max(0, int(cy - r - 1)), min(h, int(cy + r + 2))
        if x0 >= x1 or y0 >= y1:
            continue
        sx, sy = xx[y0:y1, x0:x1] + 0.5, yy[y0:y1, x0:x1] + 0.5
        m = (sx - cx) ** 2 + (sy - cy) ** 2 <= r * r
        lift = max(1.6, r * 0.16)
        lit = (sx - cx + r * 0.06) ** 2 + (sy - cy + lift) ** 2 <= (r * 0.98) ** 2
        sub = out[y0:y1, x0:x1]
        sub[m] = C["cloud2"]
        sub[m & lit] = C["cloud"]
        union[y0:y1, x0:x1] |= m
    ring = dilate(union, 1, diag=False) & ~union
    out[ring] = C["cloud3"]
    return out


class CloudFrame:
    """画面のフチを囲むもこもこ雲 ＋ 真ん中から左右に開くカーテン雲。"""

    def __init__(self, seed: int = 11):
        rng = np.random.default_rng(seed)
        puffs = []
        # 上下
        for x in np.arange(-8, GW + 16, 17):
            puffs.append((x + rng.uniform(-3, 3), -3 + rng.uniform(-2, 3), rng.uniform(11, 16)))
            puffs.append((x + rng.uniform(-3, 3), GH + 3 + rng.uniform(-3, 2), rng.uniform(11, 16)))
        # 左右
        for y in np.arange(-8, GH + 16, 16):
            puffs.append((-3 + rng.uniform(-2, 3), y + rng.uniform(-3, 3), rng.uniform(11, 16)))
            puffs.append((GW + 2 + rng.uniform(-3, 2), y + rng.uniform(-3, 3), rng.uniform(11, 16)))
        # 四隅は大きめ
        for cx, cy in ((0, 0), (GW, 0), (0, GH), (GW, GH)):
            sx, sy = (1 if cx == 0 else -1), (1 if cy == 0 else -1)
            puffs.append((cx + sx * 4, cy + sy * 3, rng.uniform(26, 30)))
            puffs.append((cx + sx * 30, cy + sy * 2, rng.uniform(15, 18)))
            puffs.append((cx + sx * 3, cy + sy * 28, rng.uniform(15, 18)))
        self.frame = [(x, y, r, rng.uniform(0, 6.28)) for x, y, r in puffs]
        # カーテン雲のふちと、中身のもこもこ
        self.edge = [(y, rng.uniform(-4, 4), rng.uniform(13, 19)) for y in np.arange(-12, GH + 24, 13)]
        self.inner = [(rng.uniform(12, 170), rng.uniform(-10, GH + 10), rng.uniform(10, 18)) for _ in range(28)]

    def frame_layer(self, t: float, amount: float = 1.0) -> np.ndarray:
        """amount=1 でフチ雲表示。0 に近いほど外へ引っ込む。"""
        puffs = []
        for x, y, r, ph in self.frame:
            dx, dy = x - GW / 2, y - GH / 2
            k = 1 + (1 - amount) * 0.5
            bob = math.sin(t * 1.6 + ph) * 1.2
            puffs.append((GW / 2 + dx * k, GH / 2 + dy * k + bob, r))
        return draw_puffs(GH, GW, puffs)

    def curtain(self, c: float, t: float = 0.0) -> np.ndarray:
        """c=0 で画面全体を覆う、c=1 で完全に開いた状態。"""
        half = GW / 2
        open_x = c * (half + 40)  # 中心からの開き量
        puffs = []
        for y, jx, r in self.edge:
            wob = math.sin(t * 3 + y * 0.3) * 1.0
            puffs.append((half - open_x + jx + wob, y, r))
            puffs.append((half + open_x - jx - wob, y + 6, r))
        for dx, y, r in self.inner:
            puffs.append((half - open_x - dx, y, r))
            puffs.append((half + open_x + dx, y + 9, r))
        xx = np.arange(GW)[None, :]
        base = np.broadcast_to((xx < half - open_x) | (xx > half + open_x), (GH, GW))
        return draw_puffs(GH, GW, puffs, base=base)


# ---------------------------------------------------------------- 青空と草原


def cloud_sprite(w, h, seed):
    """底が平らな入道雲っぽい空の雲。"""
    rng = np.random.default_rng(seed)
    puffs = []
    n = max(3, int(w / 8))
    for i in range(n):
        f = (i + 0.5) / n
        r = h * (0.28 + 0.38 * math.sin(math.pi * f)) + rng.uniform(-0.5, 1.5)
        puffs.append((3 + f * (w - 6), h + 1 - r * 0.85, r))
    for _ in range(max(1, n // 3)):
        f = rng.uniform(0.3, 0.7)
        puffs.append((3 + f * (w - 6), h * 0.42 + rng.uniform(-1, 1), h * rng.uniform(0.3, 0.4)))
    spr = draw_puffs(h + 4, w + 6, puffs)
    solid = spr[..., 3] > 0
    spr[h + 1:] = 0
    base_row = spr[h]
    base_row[solid[h]] = C["cloud3"]
    above = spr[h - 1]
    above[solid[h - 1] & (above[:, 1] > 200)] = C["cloud2"]
    ys, xs = np.nonzero(spr[..., 3])
    return spr[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()


def dandelion_sprite(kind: int):
    if kind == 0:  # 黄色い花
        rows = [
            ".y.",
            "yoy",
            ".y.",
            ".g.",
            ".g.",
        ]
    else:  # 綿毛
        rows = [
            ".e.e.",
            "e.e.e",
            ".eee.",
            "e.e.e",
            ".e.e.",
            "..g..",
            "..g..",
            "..g..",
        ]
    return from_strings(rows, {"y": C["yel"], "o": C["yel2"], "e": C["white"], "g": C["gr4"]})


class Meadow:
    """チャンネルのバナー（たんぽぽの草原）をドット絵にした背景。"""

    def __init__(self, seed: int = 5):
        rng = np.random.default_rng(seed)
        h, w = GH, GW
        yy, xx = np.mgrid[0:h, 0:w]
        d = bayer(h, w)
        sky = blank(w, h)
        cols = [C["sky0"], C["sky1"], C["sky2"], C["sky3"], C["sky4"]]
        t = yy / (h * 0.78) + (d - 0.5) * 0.12
        idx = np.clip((t * len(cols)).astype(int), 0, len(cols) - 1)
        for i, c in enumerate(cols):
            sky[idx == i] = c
        self.sky = sky
        # 遠くの丘
        far = blank(w, h)
        hill_y = 128 + 6 * np.sin(xx[0] / 23.0 + 1.3) + 4 * np.sin(xx[0] / 9.0)
        m = yy > hill_y[None, :]
        far_pal = {**P, "k": hexc("#6FB955"), "L": hexc("#9ADB74"), "M": hexc("#86CB65"), "D": hexc("#76BD5A"),
                   "h": hexc("#B8EC92"), "b": hexc("#6FB955"), "B": hexc("#6FB955")}
        for x in range(4, w, 23):
            tx = x + int(rng.uniform(-6, 6))
            spr = from_strings(TREE if rng.random() < 0.5 else TREE_S, far_pal)
            ty = int(hill_y[min(w - 1, max(0, tx + spr.shape[1] // 2))]) - spr.shape[0] + 3
            blit(far, spr, tx, ty)
        far[m] = C["hill1"]
        far[m & ~shift(m, 0, 2)] = C["hill2"]
        self.far = far
        # 手前の草原
        near = blank(w, h)
        ground_y = 146 + 3 * np.sin(xx[0] / 31.0) + 2 * np.sin(xx[0] / 7.0 + 2)
        g = yy > ground_y[None, :]
        near[g] = C["gr1"]
        near[g & (yy > ground_y[None, :] + 8 + (d - 0.5) * 3)] = C["gr2"]
        near[g & (yy > ground_y[None, :] + 20 + (d - 0.5) * 3)] = C["gr3"]
        top_edge = g & ~shift(g, 0, 1)
        near[top_edge] = C["gr0"]
        tuft = g & (rng.random((h, w)) < 0.05) & (yy > ground_y[None, :] + 2)
        near[tuft] = C["gr3"]
        near[shift(tuft, 0, -1) & g] = C["gr4"]
        self.near = near
        self.ground_y = ground_y
        # たんぽぽ
        self.flowers = []
        for _ in range(70):
            x = int(rng.uniform(2, w - 4))
            y = int(ground_y[x] + rng.uniform(2, h - ground_y[x] - 4))
            self.flowers.append((x, y, 0 if rng.random() < 0.72 else 1, rng.uniform(0, 6.28)))
        self.flowers.sort(key=lambda f: f[1])
        self.fl_spr = [dandelion_sprite(0), dandelion_sprite(1)]
        # 空の雲
        self.clouds = []
        for i, (y, cw, ch, spd) in enumerate([(14, 46, 14, 3.0), (34, 30, 10, 2.0), (52, 60, 16, 3.6),
                                              (24, 36, 11, 2.4), (70, 26, 8, 1.6)]):
            self.clouds.append((cloud_sprite(cw, ch, seed + i), rng.uniform(0, w + 80), y, spd))
        # 舞う綿毛
        self.seeds = [(rng.uniform(0, w), rng.uniform(40, h), rng.uniform(3, 7), rng.uniform(0, 6.28))
                      for _ in range(26)]

    def render(self, t: float) -> np.ndarray:
        img = self.sky.copy()
        w = GW
        for spr, x0, y, spd in self.clouds:
            span = w + spr.shape[1] + 20
            x = int((x0 - t * spd) % span) - spr.shape[1]
            blit(img, spr, x, y)
        blit(img, self.far, 0, 0)
        blit(img, self.near, 0, 0)
        for x, y, kind, ph in self.flowers:
            sway = 1 if math.sin(t * 2.6 + ph) > 0.6 else 0
            spr = self.fl_spr[kind]
            blit(img, spr, x + sway, y - spr.shape[0])
        for sx, sy, spd, ph in self.seeds:
            x = int((sx + t * spd) % (w + 10)) - 5
            y = int((sy - t * spd * 0.6 + math.sin(t * 1.5 + ph) * 3) % GH)
            if 0 <= x < w and 0 <= y < GH:
                img[y, x] = C["white"]
                if x + 1 < w and int(t * 4 + ph) % 2:
                    img[y, x + 1] = C["cloud2"]
        return img
