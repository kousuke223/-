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


# ---------------------------------------------------------------- マイクラ風のロゴ文字（ひびの入った石＋厚み＋黒フチ）

# 1マス単位の太いブロック文字（高さ 8 マス・線の太さ 2 マス）。A は上を四角くして、そこにクリーパーの顔を描く
LOGO_LETTERS = {
    "M": ["##......##", "###....###", "####..####", "##.####.##", "##..##..##", "##......##", "##......##",
          "##......##"],
    "I": ["##"] * 8,
    "N": ["##....##", "###...##", "####..##", "##.##.##", "##..####", "##...###", "##....##", "##....##"],
    "E": ["#######", "#######", "##.....", "######.", "######.", "##.....", "#######", "#######"],
    "C": [".######", "#######", "##.....", "##.....", "##.....", "##.....", "#######", ".######"],
    "R": ["######.", "#######", "##...##", "#######", "######.", "##..##.", "##...##", "##...##"],
    "A": [".#######.", "#########", "#########", "#########", "#########", "#########", "##.....##",
          "##.....##"],
    "F": ["#######", "#######", "##.....", "######.", "######.", "##.....", "##.....", "##....."],
    "T": ["########", "########", "...##...", "...##...", "...##...", "...##...", "...##...", "...##..."],
    "H": ["##...##", "##...##", "##...##", "#######", "#######", "##...##", "##...##", "##...##"],
    "D": ["######.", "#######", "##...##", "##...##", "##...##", "##...##", "#######", "######."],
    "O": [".#####.", "#######", "##...##", "##...##", "##...##", "##...##", "#######", ".#####."],
}
# クリーパーの顔（6×6）
LOGO_CREEPER = ["##..##", "##..##", "..##..", ".####.", ".####.", ".#..#."]
# ハードコアのハート（9×8）。r=赤 R=濃い赤 w=光 p=うすい赤 d=ハードコアの黒い目
LOGO_HEART = [
    ".rrr.rrr.",
    "rwwprrrrr",
    "rwprrrrrR",
    "rrdrrrdRR",
    ".rrrrrRR.",
    "..rrrRR..",
    "...rRR...",
    "....R....",
]
LOGO_HEART_COLORS = {"r": (236, 30, 42), "R": (150, 8, 26), "w": (255, 240, 240), "p": (255, 128, 136),
                     "d": (54, 4, 12)}


def _shift(m: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """配列を (dy, dx) だけずらす（はみ出た所は 0）。"""
    out = np.zeros_like(m)
    H, W = m.shape[:2]
    out[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)] = \
        m[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
    return out


def _logo_cracks(stone: np.ndarray, u: int, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """石の面に入れるひび（枝分かれする折れ線）。戻り値：ひびの線・そのすぐ下の明るい線"""
    H, W = stone.shape
    dark, light = Image.new("L", (W, H), 0), Image.new("L", (W, H), 0)
    dd, dl = ImageDraw.Draw(dark), ImageDraw.Draw(light)
    lw = max(2, round(u / 5))
    ys, xs = np.nonzero(stone)

    def walk(x, y, ang, steps):
        pts = [(x, y)]
        for _ in range(steps):
            ang += rng.uniform(-0.9, 0.9)
            x, y = x + np.cos(ang) * rng.uniform(0.45, 0.9) * u, y + np.sin(ang) * rng.uniform(0.45, 0.9) * u
            pts.append((x, y))
            if rng.random() < 0.3:
                walk(x, y, ang + rng.choice([-1.2, 1.2]), int(rng.integers(1, 3)))
        dd.line(pts, fill=255, width=lw)
        dl.line([(px, py + lw) for px, py in pts], fill=255, width=max(1, lw // 2))

    for _ in range(n):
        j = rng.integers(len(xs))
        walk(float(xs[j]), float(ys[j]), rng.uniform(0, 2 * np.pi), int(rng.integers(3, 6)))
    cr = (np.asarray(dark) > 0) & stone
    return cr, (np.asarray(light) > 0) & stone & ~cr


def logo_size(word: str, unit: int, depth: float = 1.7, outline_w: float = 0.5,
              heart_for: str | None = None) -> tuple[int, int]:
    """logo_title で描く範囲の大きさ (幅, 高さ)。"""
    E, O = int(round(depth * unit)), max(2, int(np.ceil(outline_w * unit)))
    cols = sum(len((LOGO_HEART if ch == heart_for else LOGO_LETTERS[ch])[0]) for ch in word) + len(word) - 1
    return cols * unit + 2 * O, 8 * unit + E + 2 * O


def logo_title(c: Image.Image, xy, word: str, unit: int, front=((218, 214, 208), (156, 152, 148)),
               side=(92, 88, 86), crack=(66, 62, 60), outline=(14, 12, 14, 255), depth: float = 1.7,
               outline_w: float = 0.5, heart_for: str | None = None, creeper_in: str | None = "A",
               seed: int = 0) -> tuple:
    """マイクラ風のロゴ文字：ひびの入った石の太いブロック文字＋下向きの厚み＋黒フチ。unit は1マスの大きさ(px)。
    heart_for の文字はハートに置きかえ、creeper_in の文字にはクリーパーの顔を描く。戻り値は描いた範囲。"""
    from scipy import ndimage as ndi
    u = unit
    rng = np.random.default_rng(seed)
    E, O = int(round(depth * u)), max(2, int(np.ceil(outline_w * u)))
    glyphs = [LOGO_HEART if ch == heart_for else LOGO_LETTERS[ch] for ch in word]
    Wc, Hc = logo_size(word, u, depth, outline_w, heart_for)
    # 文字を 1 マスずつあけて並べる
    face = np.zeros((Hc, Wc), bool)
    heart = np.zeros((Hc, Wc), bool)
    heart_rgb = np.zeros((Hc, Wc, 3), np.uint8)
    creeper = np.zeros((Hc, Wc), bool)
    x = O
    for ch, g in zip(word, glyphs):
        w = len(g[0])
        for r, row in enumerate(g):
            for q, p in enumerate(row):
                if p == ".":
                    continue
                cell = (slice(O + r * u, O + (r + 1) * u), slice(x + q * u, x + (q + 1) * u))
                face[cell] = True
                if g is LOGO_HEART:
                    heart[cell] = True
                    heart_rgb[cell] = LOGO_HEART_COLORS[p]
        if ch == creeper_in:   # 上の四角い所（6 マス）のまん中に顔
            f = max(2, int(round(0.8 * u)))
            cx, cy = x + (w * u - 6 * f) // 2, O + (6 * u - 6 * f) // 2
            pat = np.array([[p == "#" for p in row] for row in LOGO_CREEPER])
            creeper[cy:cy + 6 * f, cx:cx + 6 * f] = pat.repeat(f, 0).repeat(f, 1)
        x += (w + 1) * u
    # 下向きの厚み（面からの距離もおぼえておく）と黒フチ
    ext = face.copy()
    dist = np.zeros((Hc, Wc))
    for k in range(1, E + 1):
        s = _shift(face, k, 0)
        dist[s & ~ext] = k
        ext |= s
    outl = ndi.binary_dilation(ext, structure=np.ones((3, 3), bool), iterations=O)
    arr = np.zeros((Hc, Wc, 4), np.uint8)
    arr[outl] = outline
    sm = ext & ~face
    side_rgb = np.array(side, float)[None, None, :] * (1.0 - 0.55 * dist / max(1, E))[..., None]
    arr[sm, :3] = np.clip(side_rgb[sm], 0, 255).astype(np.uint8)
    arr[sm, 3] = 255
    # 正面：上から下へのグラデーション＋ドット絵っぽいざらつき＋大きなまだら
    t = np.clip((np.arange(Hc, dtype=float) - O) / (8 * u), 0, 1)[:, None, None]
    col = np.broadcast_to(np.array(front[0], float) * (1 - t) + np.array(front[1], float) * t, (Hc, Wc, 3)).copy()
    tx = max(2, u // 4)
    speck = rng.random((Hc // tx + 1, Wc // tx + 1)).repeat(tx, 0).repeat(tx, 1)[:Hc, :Wc]
    col *= (0.93 + 0.1 * np.floor(speck * 4) / 3)[..., None]
    blot = Image.fromarray((rng.random((Hc // (2 * u) + 2, Wc // (2 * u) + 2)) * 255).astype(np.uint8))
    col *= (0.94 + 0.1 * np.asarray(blot.resize((Wc, Hc), Image.BILINEAR), float) / 255)[..., None]
    # ふちの面取り（上と左は明るく、右と下は暗く）
    b = max(2, u // 5)
    col[face & ~_shift(face, 0, b)] *= 1.06
    col[face & ~_shift(face, 0, -b)] *= 0.88
    col[face & ~_shift(face, -b, 0)] *= 0.76
    top = face & ~_shift(face, b, 0)
    col[top] = col[top] * 1.14 + 14
    # ひび
    cr, cl = _logo_cracks(face & ~heart & ~creeper, u, len(word) + 2, rng)
    col[cl] = col[cl] * 1.16 + 18
    col[cr] = crack
    arr[face, :3] = np.clip(col[face], 0, 255).astype(np.uint8)
    arr[face, 3] = 255
    arr[heart, :3] = heart_rgb[heart]
    arr[creeper, :3] = (18, 20, 18)
    img = Image.fromarray(arr, "RGBA")
    # やわらかい影
    m = 3 * O
    sh = Image.new("RGBA", (Wc + 2 * m, Hc + 2 * m), (0, 0, 0, 0))
    sh.paste(Image.new("RGBA", (Wc, Hc), (0, 0, 0, 170)), (m, m), Image.fromarray((outl * 255).astype(np.uint8)))
    sh = sh.filter(ImageFilter.GaussianBlur(m / 2))
    x, y = int(xy[0]), int(xy[1])
    sx, sy = x - m + O, y - m + 2 * O
    c.alpha_composite(sh, (max(0, sx), max(0, sy)), (max(0, -sx), max(0, -sy)))
    c.alpha_composite(img, (x, y))
    return (x, y, x + Wc, y + Hc)
