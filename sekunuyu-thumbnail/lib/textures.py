"""ブロック・キャラ・ガストのドット絵テクスチャ（すべてプログラムで描いたオリジナル）。"""
from __future__ import annotations

import numpy as np


def hexc(s: str, a: int = 255):
    s = s.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), a)


def from_strings(rows: list[str], pal: dict) -> np.ndarray:
    h, w = len(rows), max(len(r) for r in rows)
    out = np.zeros((h, w, 4), np.uint8)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in pal:
                out[y, x] = pal[ch]
    return out


def _noise(h, w, seed, cell=4):
    rng = np.random.default_rng(seed)
    g = rng.random((h // cell + 2, w // cell + 2))
    yy, xx = np.mgrid[0:h, 0:w] / cell
    x0, y0 = xx.astype(int), yy.astype(int)
    fx, fy = xx - x0, yy - y0
    a, b = g[y0, x0], g[y0, x0 + 1]
    c, d = g[y0 + 1, x0], g[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def _palette_map(v, cols):
    idx = np.clip((v * len(cols)).astype(int), 0, len(cols) - 1)
    out = np.zeros(v.shape + (4,), np.uint8)
    for i, c in enumerate(cols):
        out[idx == i] = hexc(c)
    return out


# ---------------------------------------------------------------- ブロック


def netherrack(seed=1):
    n = 0.6 * _noise(16, 16, seed, 4) + 0.4 * np.random.default_rng(seed + 9).random((16, 16))
    return _palette_map(n, ["#4A1717", "#5E1F1F", "#6E2626", "#7C2D2D", "#8E3838", "#A44646"])


def quartz_ore(seed=2):
    t = netherrack(seed)
    rng = np.random.default_rng(seed)
    for _ in range(5):
        x, y = rng.integers(1, 14, 2)
        t[y, x:x + 2] = hexc("#F2EEE6")
        t[y + 1, x] = hexc("#CFC6B8")
    return t


def lava(seed=3):
    n = 0.55 * _noise(16, 16, seed, 5) + 0.3 * _noise(16, 16, seed + 1, 2) + 0.15 * np.random.default_rng(seed).random((16, 16))
    return _palette_map(n, ["#A82A08", "#C93C0C", "#E0520F", "#F26A12", "#FF8A1E", "#FFB43A"])


def lavafall(seed=4):
    """縦に流れる溶岩（滝用）。"""
    rng = np.random.default_rng(seed)
    cols = rng.random(16)
    yy = np.arange(16)[:, None]
    v = (np.sin(yy * 0.9 + cols[None, :] * 9) * 0.25 + 0.5 + 0.25 * cols[None, :])
    return _palette_map(np.clip(v, 0, 0.999), ["#D0440E", "#F0600F", "#FF8A1E", "#FFB436", "#FFE08A"])


def glowstone(seed=5):
    rng = np.random.default_rng(seed)
    cells = rng.random((5, 5))
    yy, xx = np.mgrid[0:16, 0:16]
    v = cells[(yy * 5) // 16, (xx * 5) // 16] * 0.7 + rng.random((16, 16)) * 0.3
    edge = ((yy % 3) == 0) & ((xx % 4) == 1)
    out = _palette_map(v, ["#8A5A22", "#C7862F", "#E8A63E", "#F7C95A", "#FFE58A", "#FFF6C8"])
    out[edge] = hexc("#7A4A1C")
    return out


def nether_bricks():
    out = np.zeros((16, 16, 4), np.uint8)
    out[...] = hexc("#2C1218")
    rng = np.random.default_rng(6)
    for row in range(4):
        y0 = row * 4
        off = 0 if row % 2 == 0 else 4
        for bx in range(-1, 3):
            x0 = bx * 8 + off
            xs = slice(max(0, x0 + 1), max(0, min(16, x0 + 8)))
            out[y0 + 1:y0 + 4, xs] = hexc(["#3A171F", "#431B25", "#361520"][rng.integers(3)])
            out[y0 + 1, xs] = hexc("#55263A")
        out[y0, :] = hexc("#1A080C")
    return out


def magma(seed=7):
    """暗いかたまりのすき間がオレンジに光るブロック。"""
    rng = np.random.default_rng(seed)
    pts = rng.random((7, 2)) * 16
    yy, xx = np.mgrid[0:16, 0:16] + 0.5
    d = []
    for px, py in pts:
        for ox in (-16, 0, 16):
            for oy in (-16, 0, 16):
                d.append(np.hypot(xx - px - ox, yy - py - oy))
    d = np.sort(np.stack(d), axis=0)
    border = (d[1] - d[0]) < 1.1
    out = np.zeros((16, 16, 4), np.uint8)
    out[...] = hexc("#4A170E")
    out[(d[0] < 3.2) & ~border] = hexc("#5E2012")
    out[(d[0] < 1.6) & ~border] = hexc("#6E2A16")
    out[border] = hexc("#E8601A")
    out[border & ((d[1] - d[0]) < 0.45)] = hexc("#FFB02E")
    emask = np.where(border, 0.8, 0.0).astype(np.float32)
    return out, emask


def fire(seed=9):
    """ネザーラックの上で燃える炎（下が黄色、上が赤、上の方ほど細くなる）。"""
    rng = np.random.default_rng(seed)
    out = np.zeros((16, 16, 4), np.uint8)
    cols = [hexc("#FFF2A8"), hexc("#FFD23F"), hexc("#FF9A1F"), hexc("#F2541A"), hexc("#B8240E")]
    tops = np.clip(rng.integers(2, 9, 16) + (np.abs(np.arange(16) - 7.5) * 0.5).astype(int), 1, 15)
    for x in range(16):
        for y in range(tops[x], 16):
            k = (16 - y) / (16 - tops[x] + 1e-6)
            if rng.random() < 0.08 and y < 13:
                continue
            out[y, x] = cols[min(4, int(k * 4.6))]
    return out


def solid(color, w=4, h=4):
    t = np.zeros((h, w, 4), np.uint8)
    t[...] = hexc(color)
    return t


# ---------------------------------------------------------------- ガスト

GHAST_FACE = [
    "eeeeeeeeeeeeeeee",
    "eeeeeeeeeeeeeeee",
    "eeeeeeeeeeeeeeee",
    "eeekkkkeekkkkeee",
    "eeekRRkeekRRkeee",
    "eeekkkkeekkkkeee",
    "eeegeeeeeeeegeee",
    "eeegeeeeeeeegeee",
    "eeegeeeeeeeegeee",
    "eeeeeekkkkeeeeee",
    "eeeeekmmmmkeeeee",
    "eeeeekmrrmkeeeee",
    "eeeeekmmmmkeeeee",
    "eeeeeekkkkeeeeee",
    "eeeeeeeeeeeeeeee",
    "eeeeeeeeeeeeeeee",
]


def ghast_textures():
    pal = {"e": hexc("#F2F2F0"), "k": hexc("#1A1418"), "R": hexc("#D81E16"), "g": hexc("#B8B6B8"),
           "m": hexc("#3A0A10"), "r": hexc("#C21E2A")}
    face = from_strings(GHAST_FACE, pal)
    rng = np.random.default_rng(11)

    def skin():
        t = np.zeros((16, 16, 4), np.uint8)
        t[...] = pal["e"]
        t[rng.random((16, 16)) < 0.1] = hexc("#DCDAD8")
        t[rng.random((16, 16)) < 0.04] = hexc("#C8C6C4")
        return t

    spk = rng.random((16, 16)) < 0.1
    face[spk & (face[..., 0] > 200)] = hexc("#DCDAD8")
    emask = np.zeros((16, 16), np.float32)
    emask[4, 4:6] = 1.0
    emask[4, 10:12] = 1.0
    body = {"front": face, "back": skin(), "left": skin(), "right": skin(), "top": skin(), "bottom": skin()}
    tent = np.zeros((32, 4, 4), np.uint8)
    tent[...] = pal["e"]
    tent[::5] = hexc("#D8D6D4")
    tentacle = {k: tent for k in ("front", "back", "left", "right")}
    tentacle["bottom"] = solid("#D8D6D4")
    return body, emask, tentacle


# ---------------------------------------------------------------- オーバーワールド（#3）


def grass_top(seed=21):
    n = 0.6 * _noise(16, 16, seed, 4) + 0.4 * np.random.default_rng(seed).random((16, 16))
    return _palette_map(n, ["#3F7F25", "#4C9230", "#59A23A", "#66B043", "#77BF4F"])


def dirt(seed=22):
    n = 0.5 * _noise(16, 16, seed, 3) + 0.5 * np.random.default_rng(seed).random((16, 16))
    return _palette_map(n, ["#5A3B22", "#6B4628", "#79512F", "#875C36", "#966A40"])


def grass_side(seed=23):
    """土の上に草がたれている側面。"""
    t = dirt(seed)
    g = grass_top(seed + 1)
    drip = 3 + np.random.default_rng(seed).integers(0, 3, 16)
    for x in range(16):
        t[:drip[x], x] = g[:drip[x], x]
    return t


def stone(seed=24):
    n = 0.55 * _noise(16, 16, seed, 4) + 0.45 * np.random.default_rng(seed).random((16, 16))
    return _palette_map(n, ["#6A6A6A", "#777777", "#848484", "#919191", "#9E9E9E"])


def oak_log_side(seed=25):
    rng = np.random.default_rng(seed)
    v = 0.6 * rng.random(16)[None, :] + 0.4 * rng.random((16, 16))
    return _palette_map(v, ["#3E2C18", "#4E3820", "#5E4528", "#6B5030"])


def oak_log_top():
    yy, xx = np.mgrid[0:16, 0:16] + 0.5
    r = np.hypot(xx - 8, yy - 8)
    out = _palette_map((np.sin(r * 1.6) * 0.5 + 0.5) * 0.999, ["#9A7444", "#B08A55", "#C29A60"])
    out[r > 6.6] = hexc("#4E3820")
    return out


def oak_leaves(seed=26):
    n = np.random.default_rng(seed).random((16, 16))
    out = _palette_map(n, ["#1F4F14", "#2A631B", "#367822", "#438C2A"])
    out[n < 0.08] = hexc("#163A0E")
    return out


def oak_planks():
    out = np.zeros((16, 16, 4), np.uint8)
    cols = ["#A57B48", "#B4874F", "#97703F", "#AD8150"]
    for row in range(4):
        out[row * 4:row * 4 + 4] = hexc(cols[row])
        out[row * 4] = hexc("#6E5130")
        cut = (row * 5 + 3) % 16
        out[row * 4:row * 4 + 4, cut] = hexc("#6E5130")
    return out


def obsidian(seed=27):
    n = 0.5 * _noise(16, 16, seed, 3) + 0.5 * np.random.default_rng(seed).random((16, 16))
    out = _palette_map(n, ["#0E0A16", "#160F24", "#1E1530", "#271B40"])
    out[np.random.default_rng(seed + 1).random((16, 16)) < 0.05] = hexc("#4A3580")
    return out


def portal(seed=28):
    """ネザーゲートの渦（紫）。ゆらゆらした筋がななめに流れる。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:16, 0:16] + 0.5
    warp = _noise(16, 16, seed, 4) * 6
    v = 0.5 + 0.3 * np.sin((xx + yy * 0.6 + warp) * 0.9) + 0.2 * rng.random((16, 16))
    return _palette_map(np.clip(v, 0, 0.999), ["#2A0A66", "#3E1290", "#5218B8", "#6A2AD8", "#9A62F4"])


# ---------------------------------------------------------------- ディープダーク（#1）


def deepslate(seed=31):
    """深層岩：黒っぽい灰色に、たての筋。"""
    rng = np.random.default_rng(seed)
    xx = np.mgrid[0:16, 0:16][1]
    n = 0.45 * _noise(16, 16, seed, 4) + 0.35 * rng.random((16, 16))
    n += 0.2 * (0.5 + 0.5 * np.sin(xx * 1.7 + rng.random() * 6))
    return _palette_map(np.clip(n, 0, 0.999), ["#24242A", "#2C2C33", "#35353D", "#3E3E47", "#4A4A53"])


def deepslate_tiles(seed=32):
    """深層岩タイル：4×4 のタイルと暗いすき間（古代都市のかべ）。"""
    t = deepslate(seed)
    rng = np.random.default_rng(seed)
    for ty in range(4):
        for tx in range(4):
            blk = t[ty * 4:ty * 4 + 4, tx * 4:tx * 4 + 4, :3].astype(float) * rng.uniform(0.85, 1.12)
            t[ty * 4:ty * 4 + 4, tx * 4:tx * 4 + 4, :3] = np.clip(blk, 0, 255).astype(np.uint8)
    t[3::4, :] = hexc("#141418")
    t[:, 3::4] = hexc("#1A1A20")
    return t


def sculk(seed=33):
    """スカルク：暗い青緑に、光る水色のつぶ（つぶの所だけ光る）。"""
    rng = np.random.default_rng(seed)
    n = 0.5 * _noise(16, 16, seed, 4) + 0.5 * rng.random((16, 16))
    out = _palette_map(n, ["#06161D", "#091F28", "#0C2833", "#0F323F"])
    spk = rng.random((16, 16)) < 0.08
    out[spk] = hexc("#25C8CC")
    out[spk & (rng.random((16, 16)) < 0.5)] = hexc("#9AF6F0")
    return out, np.where(spk, 1.0, 0.0).astype(np.float32)


TORCH_COLORS = {"stick": "#6B4A2A", "stick_d": "#4E3418", "ember": "#3A2412", "flame": "#FFE680", "flame_o": "#FFA020"}


WARDEN_PAL = ["#030A10", "#06121A", "#0A1A24", "#0D222E", "#112B3A"]


def _warden_skin(w, h, seed):
    rng = np.random.default_rng(seed)
    n = 0.55 * _noise(h, w, seed, 3) + 0.45 * rng.random((h, w))
    out = _palette_map(n, WARDEN_PAL)
    out[rng.random((h, w)) < 0.05] = hexc("#1A4252")
    return out


def warden_textures():
    """ウォーデン（オリジナルのドット絵）。光る所は (テクスチャ, emask) にしてある。"""
    glow, glow_l = hexc("#4FF0EA"), hexc("#B8FFF8")

    def box(w, h, d, seed):
        return {"front": _warden_skin(w, h, seed), "back": _warden_skin(w, h, seed + 1),
                "left": _warden_skin(d, h, seed + 2), "right": _warden_skin(d, h, seed + 3),
                "top": _warden_skin(w, d, seed + 4), "bottom": _warden_skin(w, d, seed + 5)}

    body = box(18, 21, 11, 40)
    # 胸：光るあばら骨と、閉じこめられたたましい（丸い顔）
    chest = body["front"]
    em = np.zeros((21, 18), np.float32)
    for y, x0, x1 in [(4, 4, 14), (7, 3, 15), (10, 4, 14), (13, 5, 13)]:
        for x in range(x0, x1):
            if x not in (8, 9):            # まん中は背骨の線であける
                chest[y, x] = glow
                em[y, x] = 1.0
    for cx, cy in [(6, 5), (11, 8), (7, 11), (12, 5)]:
        chest[cy:cy + 2, cx:cx + 2] = glow_l
        em[cy:cy + 2, cx:cx + 2] = 1.0
    body["front"] = (chest, em)
    head = box(16, 16, 10, 50)
    face = head["front"]
    face[10:12, 3:13] = hexc("#020506")    # 口
    face[10, 4:12:2] = hexc("#5A7A7E")     # 歯
    face[2:4, 2:5] = hexc("#173C44")
    face[2:4, 11:14] = hexc("#173C44")
    head["front"] = face
    arm = box(8, 28, 8, 60)
    leg = box(6, 13, 6, 70)
    # 頭の横の触角（板1枚）：ふちは暗く、中が水色に光る
    tw, th = 9, 16
    ten = _warden_skin(tw, th, 80)
    tem = np.zeros((th, tw), np.float32)
    yy, xx = np.mgrid[0:th, 0:tw]
    inner = (xx >= 2) & (xx <= tw - 3) & (yy >= 2) & (yy <= th - 3) & (((xx + yy) % 3) != 0)
    ten[inner] = glow
    ten[inner & (yy < 6)] = glow_l
    tem[inner] = 1.0
    ten[0, :] = (0, 0, 0, 0)
    ten[:, -1] = (0, 0, 0, 0)
    return {"body": body, "head": head, "arm": arm, "leg": leg, "tendril": (ten, tem)}
