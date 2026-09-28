"""ドット絵を描くための小さな道具箱。

スプライトはすべて (高さ, 幅, 4) の uint8 RGBA 配列で扱う。
低解像度で描いてから最近傍補間で拡大することで、くっきりしたドット感を出す。
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def hexc(s: str, a: int = 255) -> tuple[int, int, int, int]:
    s = s.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), a)


def blank(w: int, h: int) -> np.ndarray:
    return np.zeros((h, w, 4), np.uint8)


def up(arr: np.ndarray, s: int) -> np.ndarray:
    """整数倍の最近傍拡大。"""
    if s == 1:
        return arr
    return arr.repeat(s, axis=0).repeat(s, axis=1)




def shift(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """はみ出した分は捨てる平行移動（np.roll と違って回り込まない）。"""
    out = np.zeros_like(mask)
    h, w = mask.shape[:2]
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    if xs0 < xs1 and ys0 < ys1:
        out[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx] = mask[ys0:ys1, xs0:xs1]
    return out


def dilate(mask: np.ndarray, r: int = 1, diag: bool = True) -> np.ndarray:
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if (dx or dy) and (diag or abs(dx) + abs(dy) <= r):
                out |= shift(mask, dx, dy)
    return out


def pad(arr: np.ndarray, n: int) -> np.ndarray:
    widths = ((n, n), (n, n)) + ((0, 0),) * (arr.ndim - 2)
    return np.pad(arr, widths)




def blit(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    """src を dst の (x, y) に重ねる（アルファ合成・はみ出しは切り捨て）。"""
    h, w = src.shape[:2]
    H, W = dst.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    d = dst[y0:y1, x0:x1].astype(np.float32)
    sa = s[..., 3:4] / 255.0
    da = d[..., 3:4] / 255.0
    oa = sa + da * (1 - sa)
    rgb = (s[..., :3] * sa + d[..., :3] * da * (1 - sa)) / np.maximum(oa, 1e-6)
    dst[y0:y1, x0:x1, :3] = np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
    dst[y0:y1, x0:x1, 3] = np.clip(oa[..., 0] * 255 + 0.5, 0, 255).astype(np.uint8)


def grid(h: int, w: int):
    return np.mgrid[0:h, 0:w]


def ellipse_mask(h, w, cx, cy, rx, ry) -> np.ndarray:
    yy, xx = grid(h, w)
    return ((xx + 0.5 - cx) / max(rx, 1e-6)) ** 2 + ((yy + 0.5 - cy) / max(ry, 1e-6)) ** 2 <= 1.0


def poly_mask(h, w, pts) -> np.ndarray:
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in pts], fill=255)
    return np.asarray(im) > 0


def line_mask(h, w, pts, width=1) -> np.ndarray:
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).line([tuple(p) for p in pts], fill=255, width=width)
    return np.asarray(im) > 0


def round_rect_mask(h, w, x0, y0, x1, y1, r) -> np.ndarray:
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).rounded_rectangle((x0, y0, x1, y1), radius=r, fill=255)
    return np.asarray(im) > 0




def from_strings(rows: list[str], palette: dict[str, tuple]) -> np.ndarray:
    h, w = len(rows), max(len(r) for r in rows)
    out = blank(w, h)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in palette:
                out[y, x] = palette[ch]
    return out


# ---------------------------------------------------------------- 文字


_font_cache: dict = {}


def _font(path: str, size: int):
    key = (path, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, size)
    return _font_cache[key]


def text_mask(text: str, font_path: str, size: int, bold: bool = False) -> np.ndarray:
    """アンチエイリアスなしで文字を描いた bool マスク（上下左右の余白は詰める）。"""
    f = _font(font_path, size)
    lines = text.split("\n")
    asc, desc = f.getmetrics()
    line_h = asc + desc
    widths = [max(1, int(f.getlength(ln))) for ln in lines]
    w = max(widths) + 4
    h = line_h * len(lines) + 4
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    d.fontmode = "1"
    for i, ln in enumerate(lines):
        x = 2 + (max(widths) - widths[i]) // 2
        d.text((x, 2 + i * line_h), ln, font=f, fill=255)
    m = np.asarray(im) > 0
    if bold:
        m = m | shift(m, 1, 0)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return m[:1, :1]
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def glyph_cells(text: str, font_path: str, size: int, bold: bool = False) -> list[np.ndarray]:
    """1文字ずつのマスク。ベースラインと高さを全文字でそろえる（文字ごとのアニメ用）。"""
    f = _font(font_path, size)
    asc, desc = f.getmetrics()
    cells = []
    for ch in text:
        w = max(1, int(round(f.getlength(ch))))
        im = Image.new("L", (w + 2, asc + desc + 2), 0)
        d = ImageDraw.Draw(im)
        d.fontmode = "1"
        d.text((1, 1), ch, font=f, fill=255)
        m = np.asarray(im) > 0
        if bold:
            m = m | shift(m, 1, 0)
        cells.append(m)
    rows = np.zeros(cells[0].shape[0], bool)
    for m in cells:
        rows |= m.any(axis=1)
    ys = np.nonzero(rows)[0]
    return [m[ys.min():ys.max() + 1] for m in cells]


def styled(mask: np.ndarray, fill_top, fill_bottom=None, outlines=(), shadow=None,
           bands: int = 0) -> np.ndarray:
    """文字マスクに グラデーション塗り＋多重フチ＋影 を付ける。

    outlines: [(color, 太さ), ...] 内側から順に。
    shadow:   (color, dx, dy)
    bands:    グラデーションを何段の色帯にするか（0 ならなめらか）。ドット絵らしさを出すなら 3〜4。
    """
    total = sum(r for _, r in outlines)
    sdx, sdy = (shadow[1], shadow[2]) if shadow else (0, 0)
    p = total + max(abs(sdx), abs(sdy))
    m = pad(mask, p)
    h, w = m.shape
    out = blank(w, h)
    layers = []
    cur = m
    for color, r in outlines:
        nxt = dilate(cur, r, diag=True)
        layers.append((nxt, color))
        cur = nxt
    if shadow:
        out[shift(cur, sdx, sdy)] = shadow[0]
    for lm, color in reversed(layers):
        out[lm] = color
    ys = np.nonzero(m.any(axis=1))[0]
    y0, y1 = (ys.min(), ys.max()) if len(ys) else (0, h - 1)
    t = np.clip((np.arange(h) - y0) / max(1, y1 - y0), 0, 1)
    if bands:
        t = np.floor(t * bands) / max(1, bands - 1)
        t = np.clip(t, 0, 1)
    top = np.array(fill_top[:3], np.float32)
    bot = np.array((fill_bottom or fill_top)[:3], np.float32)
    col = (top[None, :] * (1 - t[:, None]) + bot[None, :] * t[:, None]).astype(np.uint8)
    rgb = np.broadcast_to(col[:, None, :], (h, w, 3))
    out[m, :3] = rgb[m]
    out[m, 3] = 255
    return out


# ---------------------------------------------------------------- 小物スプライト


def star_sprite(color=hexc("#FFD23F"), line=hexc("#2A1E2E"), shine=hexc("#FFFFFF")) -> np.ndarray:
    rows = [
        "....k....",
        "...kyk...",
        "..kyyyk..",
        "kkkywyykk",
        "kyyyyyyyk",
        ".kyyyyyk.",
        "..kyyyk..",
        ".kyykyyk.",
        ".kyk.kyk.",
        ".kk...kk.",
    ]
    return from_strings(rows, {"k": line, "y": color, "w": shine})


def sparkle_sprite(color=hexc("#FFFFFF")) -> np.ndarray:
    rows = [
        "..w..",
        "..w..",
        "wwwww",
        "..w..",
        "..w..",
    ]
    return from_strings(rows, {"w": color})


def heart_sprite(color=hexc("#FF6F9C"), line=hexc("#2A1E2E"), shine=hexc("#FFFFFF")) -> np.ndarray:
    rows = [
        ".kk.kk.",
        "kwrkrrk",
        "krrrrrk",
        "krrrrrk",
        ".krrrk.",
        "..krk..",
        "...k...",
    ]
    return from_strings(rows, {"k": line, "r": color, "w": shine})


def coin_sprite(t: float = 0.0) -> np.ndarray:
    """くるくる回るコイン（t は 0〜1 の回転位相）。"""
    k, y, o, w = hexc("#2A1E2E"), hexc("#FFD23F"), hexc("#E08A1E"), hexc("#FFF6C2")
    phase = int((t % 1.0) * 4)
    if phase == 0:
        rows = [
            "..kkkk..",
            ".kyyyyk.",
            "kywyyoyk",
            "kywyyoyk",
            "kywyyoyk",
            "kyyyyoyk",
            ".kyoooK.",
            "..kkkk..",
        ]
    elif phase in (1, 3):
        rows = [
            "...kk...",
            "..kyyk..",
            "..kwok..",
            "..kwok..",
            "..kwok..",
            "..kyok..",
            "..kyok..",
            "...kk...",
        ]
    else:
        rows = [
            "...kk...",
            "...kk...",
            "...kk...",
            "...kk...",
            "...kk...",
            "...kk...",
            "...kk...",
            "...kk...",
        ]
    return from_strings(rows, {"k": k, "K": k, "y": y, "o": o, "w": w})
