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


# ---------------------------------------------------------------- せくぬゆ（プレイヤー）

SKIN = {
    "h": hexc("#1C1820"), "H": hexc("#3E3848"), "g": hexc("#2B2533"),
    "s": hexc("#F2C29B"), "S": hexc("#D9A07A"), "l": hexc("#FFDDBE"), "n": hexc("#C98A66"),
    "w": hexc("#FFFFFF"), "k": hexc("#14101A"), "b": hexc("#2A1C1C"),
    "m": hexc("#2E060C"), "t": hexc("#FFFFFF"), "r": hexc("#F0506A"),
    "d": hexc("#8FD8FF"), "D": hexc("#E8F8FF"), "p": hexc("#8A92D0"),
    "T": hexc("#F7F7F7"), "u": hexc("#D9DCE6"), "U": hexc("#B9BDCB"),
    "j": hexc("#2E3F6E"), "J": hexc("#22305A"), "o": hexc("#3C3C46"), "O": hexc("#E6E6EA"),
    "y": hexc("#FFD23F"), "Y": hexc("#E08A1E"),
}

FACE_SCREAM = [
    "hhhhhhhhhhhhhhhh",
    "hhHhhhhhhhhhHhhh",
    "hhhhhhhhhhhhhhhh",
    "shhhhhshhhhhhhhs",
    "ssspssspssspsssd",
    "sssbbssssssbbssD",
    "sbbsssssssssssbb",
    "swwwwssssswwwwss",
    "swwwwssssswwwwss",
    "swkwwssSssswkwss",
    "swwwwssSsswwwwss",
    "sssssmmmmmmsssss",
    "ssssmmttttmmssss",
    "ssssmmmmmmmmssss",
    "ssssmmmrrmmmssss",
    "sssssmmrrmmsssss",
]

HEAD_SIDE = [
    "hhhhhhhhhhhhhhhh",
    "hhhHhhhhhhhhhhhh",
    "hhhhhhhhhHhhhhhh",
    "hhhhhhhhhhhhhhhh",
    "hhhhhhhhhhhhhhhs",
    "hhhhhhhhhhhhhsss",
    "hhhhhhhhhhhhssss",
    "hhhhhhhhhhhsssss",
    "hhhhhhhhhhssssss",
    "hhhhhhhhhhssssss",
    "hhhhhhhhhsssssss",
    "hhhhhhhhssssssss",
    "hhhhhhhsssssssss",
    "hhhhhhssssssSsss",
    "hhhhhsssssssssss",
    "hhhhssssssssssss",
]


def player_textures(expression=FACE_SCREAM):
    face = from_strings(expression, SKIN)
    side_l = from_strings(HEAD_SIDE, SKIN)          # +x 面（キャラの左）：右側が顔側
    side_r = side_l[:, ::-1].copy()                 # -x 面（キャラの右）：左側が顔側
    hair = np.zeros((16, 16, 4), np.uint8)
    hair[...] = SKIN["h"]
    rng = np.random.default_rng(3)
    hl = rng.random((16, 16)) < 0.12
    hair[hl] = SKIN["H"]
    back = hair.copy()
    back[13:] = SKIN["s"]
    back[13:, 0:3] = SKIN["h"]
    back[13:, 13:] = SKIN["h"]
    chin = np.zeros((16, 16, 4), np.uint8)
    chin[...] = SKIN["S"]
    head = {"front": face, "back": back, "left": side_l, "right": side_r, "top": hair, "bottom": chin}

    # 髪のボリューム（帽子レイヤー）：顔の部分は透明
    hat_front = np.zeros((16, 16, 4), np.uint8)
    fringe = [4, 5, 5, 4, 4, 3, 3, 3, 3, 4, 3, 3, 2, 2, 3, 4]  # 前髪の長さ（列ごと）
    for x, n in enumerate(fringe):
        hat_front[:n, x] = SKIN["h"]
        hat_front[0, x] = SKIN["g"]
    hat_front[1, 2:5] = SKIN["H"]
    hat_front[1, 9:11] = SKIN["H"]
    hat_side = np.zeros((16, 16, 4), np.uint8)
    for x in range(16):
        n = 6 if x < 10 else 4
        hat_side[:n, x] = SKIN["h"]
    hat_side[1, 3:6] = SKIN["H"]
    hat_side_r = hat_side[:, ::-1].copy()
    hat_back = np.zeros((16, 16, 4), np.uint8)
    hat_back[:12] = SKIN["h"]
    hat_back[2, 4:8] = SKIN["H"]
    hat_top = hair.copy()
    hat_top[3, 4:9] = SKIN["H"]
    hat = {"front": hat_front, "back": hat_back, "left": hat_side, "right": hat_side_r, "top": hat_top}

    # 胴体（白T）
    def tee(w, h, front=False):
        t = np.zeros((h, w, 4), np.uint8)
        t[...] = SKIN["T"]
        rng2 = np.random.default_rng(w * 31 + h)
        t[rng2.random((h, w)) < 0.08] = SKIN["u"]
        t[-2:] = SKIN["u"]
        if front:
            t[0, w // 2 - 3:w // 2 + 3] = SKIN["s"]
            t[1, w // 2 - 2:w // 2 + 2] = SKIN["s"]
            t[2, w // 2 - 1:w // 2 + 1] = SKIN["U"]
            # 胸の小さなたんぽぽマーク
            t[7, 4] = SKIN["y"]
            t[6:9, 4] = SKIN["y"]
            t[7, 3:6] = SKIN["y"]
            t[7, 4] = SKIN["Y"]
        return t

    body = {"front": tee(16, 24, True), "back": tee(16, 24), "left": tee(8, 24), "right": tee(8, 24),
            "top": tee(16, 8), "bottom": tee(16, 8)}

    def arm(w, h):
        t = np.zeros((h, w, 4), np.uint8)
        t[...] = SKIN["s"]
        t[:7] = SKIN["T"]
        t[6] = SKIN["u"]
        t[-4:] = SKIN["S"]
        t[:, -1:] = SKIN["S"]
        return t

    arms = {k: arm(8, 24) for k in ("front", "back", "left", "right")}
    arms["top"] = np.full((8, 8, 4), SKIN["T"], np.uint8)
    arms["bottom"] = np.full((8, 8, 4), SKIN["S"], np.uint8)

    def leg(w, h):
        t = np.zeros((h, w, 4), np.uint8)
        t[...] = SKIN["j"]
        t[:, ::3] = SKIN["J"]
        t[-5:] = SKIN["o"]
        t[-1:] = SKIN["O"]
        return t

    legs = {k: leg(8, 24) for k in ("front", "back", "left", "right")}
    legs["top"] = np.full((8, 8, 4), SKIN["j"], np.uint8)
    legs["bottom"] = np.full((8, 8, 4), SKIN["O"], np.uint8)
    return {"head": head, "hat": hat, "body": body, "arm": arms, "leg": legs}


def solid(color, w=4, h=4):
    t = np.zeros((h, w, 4), np.uint8)
    t[...] = hexc(color)
    return t


def headset_textures():
    cup = solid("#FFD23F")
    cup[0, :] = hexc("#FFF0A0")
    cup[-1, :] = hexc("#E08A1E")
    band = solid("#34364A")
    return {"cup": {k: cup for k in ("front", "back", "left", "right", "top", "bottom")},
            "cup_in": {k: solid("#34364A") for k in ("front", "back", "left", "right", "top", "bottom")},
            "band": {k: band for k in ("front", "back", "left", "right", "top", "bottom")}}


SWORD = [
    "...........kkkk.",
    "..........kddDk.",
    ".........kddDDk.",
    "........kddDDk..",
    ".......kddDDk...",
    "......kddDDk....",
    ".kk..kddDDk.....",
    ".kgkkddDDk......",
    "..kgkdDDk.......",
    "...kgkDk........",
    "...kbgkk........",
    "..kbkkgk........",
    ".kbk..kgk.......",
    "kkk....kk.......",
    "kk..............",
    "................",
]


def sword_texture():
    return from_strings(SWORD, {"k": hexc("#16232A"), "d": hexc("#4AEDD9"), "D": hexc("#B8FFF4"),
                                "g": hexc("#2B8F86"), "b": hexc("#6B4A2A")})


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
