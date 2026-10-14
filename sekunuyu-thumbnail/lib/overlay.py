"""サムネの文字・ハート・バッジ（太いフチ取り文字など）。"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

_fonts: dict = {}


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    key = (path, size)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(path, size)
    return _fonts[key]


def gradient(size, stops):
    """上から下へのグラデーション画像。stops = [(位置0〜1, (r,g,b)), ...]"""
    w, h = size
    t = np.linspace(0, 1, h)[:, None]
    pos = np.array([p for p, _ in stops])
    cols = np.array([c for _, c in stops], float)
    rgb = np.stack([np.interp(t[:, 0], pos, cols[:, i]) for i in range(3)], axis=1)
    arr = np.broadcast_to(rgb[:, None, :], (h, w, 3)).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(arr), "RGB")


def fx_text(canvas: Image.Image, xy, text: str, fnt, fill, strokes=(), shadow=None, rotate: float = 0.0,
            spacing: int = 0) -> tuple[int, int, int, int]:
    """太いフチ取り文字。fill は色 か グラデーションの stops。strokes は外側から順に (太さ, 色)。
    shadow = (dx, dy, 色, ぼかし)。戻り値は描いた範囲。"""
    maxw = max([w for w, _ in strokes], default=0)
    pad = maxw + (max(abs(shadow[0]), abs(shadow[1])) + shadow[3] * 2 if shadow else 0) + 4
    # 文字ごとに並べる（字間を調整できるように）
    widths = [fnt.getlength(ch) for ch in text]
    total_w = int(sum(widths) + spacing * (len(text) - 1))
    asc, desc = fnt.getmetrics()
    W, H = total_w + pad * 2, asc + desc + pad * 2
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    def draw_all(img, **kw):
        d = ImageDraw.Draw(img)
        x = pad
        for ch, cw in zip(text, widths):
            d.text((x, pad), ch, font=fnt, **kw)
            x += cw + spacing

    if shadow:
        dx, dy, col, bl = shadow
        sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw_all(sh, fill=col, stroke_width=maxw, stroke_fill=col)
        if bl:
            sh = sh.filter(ImageFilter.GaussianBlur(bl))
        layer.alpha_composite(sh.crop((0, 0, W - dx, H - dy)), (dx, dy))
    for wdt, col in strokes:
        draw_all(layer, fill=col, stroke_width=wdt, stroke_fill=col)
    mask = Image.new("L", (W, H), 0)
    draw_all(mask, fill=255)
    if isinstance(fill, (list, tuple)) and fill and isinstance(fill[0], tuple) and len(fill[0]) == 2:
        bb = mask.getbbox()
        g = gradient((W, bb[3] - bb[1]), fill)
        gfull = Image.new("RGB", (W, H))
        gfull.paste(g, (0, bb[1]))
        layer.paste(gfull, (0, 0), mask)
    else:
        layer.paste(Image.new("RGBA", (W, H), fill), (0, 0), mask)
    if rotate:
        layer = layer.rotate(rotate, resample=Image.BICUBIC, expand=True)
    x, y = int(xy[0] - pad), int(xy[1] - pad)
    canvas.alpha_composite(layer, (max(0, x), max(0, y)), (max(0, -x), max(0, -y)))
    bb = layer.getbbox()
    return (x + bb[0], y + bb[1], x + bb[2], y + bb[3])


HEART_FULL = [
    ".kk.kk.",
    "kwrkrrk",
    "kwrrrrk",
    "krrrrRk",
    ".krrRk.",
    "..kRk..",
    "...k...",
]
HEART_HALF = [
    ".kk.kk.",
    "kwrkddk",
    "kwrrddk",
    "krrrddk",
    ".krrdk.",
    "..kRk..",
    "...k...",
]
HEART_EMPTY = [
    ".kk.kk.",
    "kddkddk",
    "kdddddk",
    "kdddddk",
    ".kdddk.",
    "..kdk..",
    "...k...",
]


def heart(kind: str, scale: int) -> Image.Image:
    rows = {"full": HEART_FULL, "half": HEART_HALF}.get(kind, HEART_EMPTY)
    pal = {"k": (24, 8, 10, 255), "w": (255, 210, 210, 255), "r": (230, 30, 40, 255), "R": (160, 10, 22, 255),
           "d": (60, 24, 28, 230)}
    h, w = len(rows), len(rows[0])
    arr = np.zeros((h, w, 4), np.uint8)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in pal:
                arr[y, x] = pal[ch]
    return Image.fromarray(arr, "RGBA").resize((w * scale, h * scale), Image.NEAREST)


def badge(canvas, xy, text, fnt, pad=(26, 10), border=10, fill=(255, 255, 255, 255), line=(220, 24, 30, 255),
          ink=(20, 14, 18, 255), rotate=0.0, radius=18):
    bb = fnt.getbbox(text)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    w, h = tw + pad[0] * 2 + border * 2, th + pad[1] * 2 + border * 2
    m = 24
    layer = Image.new("RGBA", (w + m * 2, h + m * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle((m + 6, m + 8, m + w + 6, m + h + 8), radius, fill=(0, 0, 0, 150))
    d.rounded_rectangle((m, m, m + w, m + h), radius, fill=line)
    d.rounded_rectangle((m + border, m + border, m + w - border, m + h - border), max(4, radius - border), fill=fill)
    d.text((m + border + pad[0] - bb[0], m + border + pad[1] - bb[1]), text, font=fnt, fill=ink)
    if rotate:
        layer = layer.rotate(rotate, resample=Image.BICUBIC, expand=True)
    canvas.alpha_composite(layer, (int(xy[0] - m), int(xy[1] - m)))
    return (int(xy[0]), int(xy[1]), int(xy[0] + w), int(xy[1] + h))


SWEAT = [
    "....k....",
    "...kdk...",
    "...kdk...",
    "..kwddk..",
    "..kwddk..",
    ".kwdddDk.",
    ".kddddDk.",
    ".kdddDDk.",
    "..kDDDk..",
    "...kkk...",
]


def sweat(scale: int, angle: float = 0.0) -> Image.Image:
    """マンガっぽい汗のしずく（ドット絵）。angle で向きを変える。"""
    pal = {"k": (22, 40, 86, 255), "d": (130, 214, 255, 255), "D": (60, 160, 232, 255), "w": (255, 255, 255, 255)}
    h, w = len(SWEAT), len(SWEAT[0])
    arr = np.zeros((h, w, 4), np.uint8)
    for y, row in enumerate(SWEAT):
        for x, ch in enumerate(row):
            if ch in pal:
                arr[y, x] = pal[ch]
    img = Image.fromarray(arr, "RGBA").resize((w * scale, h * scale), Image.NEAREST)
    return img.rotate(angle, resample=Image.NEAREST, expand=True) if angle else img


DARK = (34, 6, 8, 255)


def title_block(c: Image.Image, S: int, series: tuple[str, str], catch: tuple[str, str], badge_text: str,
                black: str, dela: str, catch_size: tuple[int, int] = (86, 104)) -> tuple[int, int, int, int]:
    """全エピソード共通の文字組み：シリーズ名（2語）＋キャッチ（白＋強調）＋番号バッジ。S 倍で描く。"""
    shadow = (6 * S, 8 * S, (0, 0, 0, 170), 4 * S)
    f = font(black, 118 * S)
    b1 = fx_text(c, (22 * S, 2 * S), series[0], f,
                 [(0, (255, 240, 120)), (0.5, (255, 206, 40)), (1, (255, 138, 18))],
                 strokes=[(16 * S, DARK), (9 * S, (214, 28, 28, 255))], shadow=shadow, spacing=-4 * S)
    fx_text(c, (b1[2] + 4 * S, 2 * S), series[1], f,
            [(0, (255, 92, 70)), (0.55, (236, 28, 30)), (1, (176, 10, 16))],
            strokes=[(16 * S, DARK), (9 * S, (255, 255, 255, 255))], shadow=shadow, spacing=-4 * S)
    s1, s2 = catch_size
    b2 = fx_text(c, (30 * S, 150 * S), catch[0], font(black, s1 * S), (255, 255, 255, 255),
                 strokes=[(11 * S, DARK)], shadow=(5 * S, 7 * S, (60, 0, 0, 220), 2 * S), spacing=-4 * S)
    b3 = fx_text(c, (b2[2] - 10 * S, (150 - (s2 - s1) * 7 // 9) * S), catch[1], font(black, s2 * S),
                 [(0, (255, 250, 170)), (0.45, (255, 214, 60)), (1, (255, 120, 20))],
                 strokes=[(13 * S, DARK)], shadow=(5 * S, 7 * S, (60, 0, 0, 220), 2 * S), spacing=-6 * S, rotate=-4)
    return badge(c, (b3[2] + 6 * S, 156 * S), badge_text, font(dela, 64 * S), pad=(20 * S, 6 * S),
                 border=8 * S, radius=16 * S, rotate=6)


def hearts_bar(c: Image.Image, S: int, kinds: list[str], x: int = 26, y: int = 652) -> None:
    """マイクラ風の体力ハート（左下に横一列）。"""
    for i, k in enumerate(kinds):
        c.alpha_composite(heart(k, 6 * S), ((x + i * 44) * S, y * S))


def arrow(c: Image.Image, start, end, width: int, head_len: int, head_w: int, fill=(232, 28, 36, 255),
          stroke=(255, 255, 255, 255), outline=DARK, stroke_w: int = 8, outline_w: int = 6) -> None:
    """白フチ＋黒フチの太い矢印（start → end の向き）。"""
    from PIL import ImageFilter as F
    sx, sy = start
    ex, ey = end
    L = float(np.hypot(ex - sx, ey - sy))
    ux, uy = (ex - sx) / L, (ey - sy) / L
    nx, ny = -uy, ux
    pts_local = [(0, -width / 2), (L - head_len, -width / 2), (L - head_len, -head_w / 2), (L, 0),
                 (L - head_len, head_w / 2), (L - head_len, width / 2), (0, width / 2)]
    pts = [(sx + ux * a + nx * b, sy + uy * a + ny * b) for a, b in pts_local]
    m = stroke_w + outline_w + 4
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, y0 = int(min(xs)) - m, int(min(ys)) - m
    x1, y1 = int(max(xs)) + m, int(max(ys)) + m
    mask = Image.new("L", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(mask).polygon([(x - x0, y - y0) for x, y in pts], fill=255)
    white = mask.filter(F.MaxFilter(2 * stroke_w + 1))
    dark = white.filter(F.MaxFilter(2 * outline_w + 1))
    layer = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    shadow = Image.new("RGBA", mask.size, (0, 0, 0, 150))
    layer.paste(shadow, (0, 0), dark.filter(F.GaussianBlur(outline_w)))
    layer.paste(Image.new("RGBA", mask.size, outline), (0, 0), dark)
    layer.paste(Image.new("RGBA", mask.size, stroke), (0, 0), white)
    layer.paste(Image.new("RGBA", mask.size, fill), (0, 0), mask)
    c.alpha_composite(layer, (max(0, x0), max(0, y0)), (max(0, -x0), max(0, -y0)))


STONE = [(118, 118, 122), (134, 134, 138), (150, 150, 154), (168, 168, 172)]
RED_STONE = [(150, 26, 30), (176, 36, 38), (204, 52, 48), (228, 78, 64)]


def block_title(c: Image.Image, xy, text: str, font_path: str, cell: int, colors=STONE, extrude: int = 0,
                outline_w: int = 0, side=(52, 52, 58), outline=(16, 16, 20, 255), seed: int = 0) -> tuple:
    """ドット1つを石ブロック1個にして組んだ英字（マイクラ風の立体タイトル）。xy は左上。戻り値は描いた範囲。"""
    from scipy import ndimage as ndi
    f = ImageFont.truetype(font_path, 8)
    bb = f.getbbox(text)
    im = Image.new("L", (bb[2] - bb[0] + 4, bb[3] - bb[1] + 4), 0)
    d = ImageDraw.Draw(im)
    d.fontmode = "1"
    d.text((2 - bb[0], 2 - bb[1]), text, font=f, fill=255)
    bm = np.asarray(im) > 0
    ys, xs = np.nonzero(bm)
    bm = bm[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = bm.shape
    rng = np.random.default_rng(seed)
    # 石の模様（ブロックごとに少し明るさを変えて、細かいつぶつぶを入れる）
    sub = max(2, cell // 4)
    speck = rng.random((h * cell // sub + 1, w * cell // sub + 1)).repeat(sub, 0).repeat(sub, 1)[:h * cell, :w * cell]
    pal = np.array(colors, float)
    face = pal[np.clip((speck * len(pal)).astype(int), 0, len(pal) - 1)]
    face *= (0.9 + 0.2 * rng.random((h, w))).repeat(cell, 0).repeat(cell, 1)[..., None]
    # ブロックのふち（上と左は明るく、下と右は暗く）
    yy, xx = np.mgrid[0:h * cell, 0:w * cell]
    by, bx = yy % cell, xx % cell
    bev = max(1, cell // 6)
    face = np.where(((by < bev) | (bx < bev))[..., None], np.minimum(face * 1.1 + 8, 255), face)
    face = np.where(((by >= cell - bev) | (bx >= cell - bev))[..., None], face * 0.84, face)
    face *= np.linspace(1.12, 0.86, h * cell)[:, None, None]   # 上が明るい
    face = np.clip(face, 0, 255)
    on = bm.repeat(cell, 0).repeat(cell, 1)
    E, O = extrude, outline_w
    Hc, Wc = h * cell + E + 2 * O, w * cell + 2 * O
    sil = np.zeros((Hc, Wc), bool)
    sil[O:O + h * cell, O:O + w * cell] = on
    ext = sil.copy()
    for k in range(1, E + 1):  # 下に厚みを出す
        ext[k:] |= sil[:-k]
    outl = ndi.binary_dilation(ext, structure=np.ones((3, 3), bool), iterations=O) if O else ext
    arr = np.zeros((Hc, Wc, 4), np.uint8)
    arr[outl] = outline
    sidek = np.linspace(1.15, 0.7, Hc)[:, None, None]
    side_rgb = np.clip(np.array(side, float)[None, None, :] * sidek, 0, 255)
    side_mask = ext & ~sil
    arr[..., :3][side_mask] = np.broadcast_to(side_rgb, (Hc, Wc, 3))[side_mask].astype(np.uint8)
    arr[..., 3][side_mask] = 255
    fm = np.zeros((Hc, Wc, 3), float)
    fm[O:O + h * cell, O:O + w * cell] = face
    arr[..., :3][sil] = fm[sil].astype(np.uint8)
    arr[..., 3][sil] = 255
    layer = Image.fromarray(arr, "RGBA")
    # やわらかい影
    m = 4 * max(1, O)
    sh = Image.new("RGBA", (Wc + 2 * m, Hc + 2 * m), (0, 0, 0, 0))
    sh.paste(Image.new("RGBA", (Wc, Hc), (0, 0, 0, 160)), (m, m), Image.fromarray((outl * 255).astype(np.uint8)))
    sh = sh.filter(ImageFilter.GaussianBlur(m / 2))
    x, y = int(xy[0]), int(xy[1])
    c.alpha_composite(sh, (max(0, x - m + O), max(0, y - m + 2 * O)))
    c.alpha_composite(layer, (x, y))
    return (x, y, x + Wc, y + Hc)
