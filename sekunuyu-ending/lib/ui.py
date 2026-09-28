"""ロゴ・テロップ・終了画面の枠など、画面に重ねる部品。"""
from __future__ import annotations

import os

import numpy as np

from .pixelart import (blank, blit, dilate, ellipse_mask, from_strings, glyph_cells, hexc, round_rect_mask,
                       shift, styled, text_mask)

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "..", "fonts")
FONT_DOT = os.path.join(FONT_DIR, "DotGothic16-Regular.ttf")
FONT_ROUND = os.path.join(FONT_DIR, "MPLUSRounded1c-Bold.ttf")
FONT_8BIT = os.path.join(FONT_DIR, "PressStart2P-Regular.ttf")

INK = hexc("#2A1E2E")
WHITE = hexc("#FFFFFF")
YELLOW = hexc("#FFD23F")
CREAM = hexc("#FFF4B8")
ORANGE = hexc("#FF8A1F")
PINK = hexc("#FF5C8A")
RED = hexc("#FF3D3D")

BOX_COLORS = {
    "blue": ("#A8E4FF", "#5CC3FF", "#3FA2F5", "#2F7FE6"),
    "green": ("#C2F5A8", "#6ADB6A", "#48C155", "#2F9E48"),
    "orange": ("#FFE3A3", "#FFB347", "#FF9A2E", "#F57C1F"),
    "pink": ("#FFD0E2", "#FF8FB8", "#FF6B9E", "#E84C84"),
}


def box(w: int, h: int, colors=BOX_COLORS["blue"], radius: int = 5, border=WHITE, shadow: int = 2) -> np.ndarray:
    """ゲームのウィンドウ風の角丸ボックス（外フチ＋白フチ＋3段グラデ＋ハイライト＋影）。"""
    hi, c1, c2, c3 = (hexc(c) for c in colors)
    W, H = w + 2, h + 2 + shadow
    out = blank(W, H)
    outer = round_rect_mask(H, W, 0, 0, w + 1, h + 1, radius + 1)
    if shadow:
        out[shift(outer, 0, shadow)] = INK
    out[outer] = INK
    inner = round_rect_mask(H, W, 1, 1, w, h, radius)
    out[inner] = border
    fill = round_rect_mask(H, W, 2, 2, w - 1, h - 1, max(1, radius - 1))
    yy = np.arange(H)[:, None]
    band = np.clip(((yy - 2) / max(1, h - 3) * 3).astype(int), 0, 2)
    for i, c in enumerate((c1, c2, c3)):
        out[fill & (band == i)] = c
    out[fill & (yy == 3) & ~shift(~fill, 0, 1)] = hi
    return out


def logo(text: str = "せくぬゆ", colors=BOX_COLORS["blue"], size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """ロゴのスプライトと、文字の塗り部分のマスク（キラーン演出用）を返す。"""
    m = text_mask(text, FONT_ROUND, size)
    txt = styled(m, WHITE, CREAM, outlines=[(ORANGE, 1), (INK, 1)], shadow=(INK, 0, 2), bands=3)
    th, tw = txt.shape[:2]
    bw, bh = tw + 16, th + 8
    b = box(bw, bh, colors, radius=6, shadow=3)
    out = blank(b.shape[1], b.shape[0])
    blit(out, b, 0, 0)
    ox, oy = (bw + 2 - tw) // 2, (bh + 2 - th) // 2 - 1
    blit(out, txt, ox, oy)
    p = 2 + 2  # styled() の余白 = フチの合計 + 影のずれ
    fill = np.zeros(out.shape[:2], bool)
    fill[oy + p:oy + p + m.shape[0], ox + p:ox + p + m.shape[1]] = m
    return out, fill


def ribbon(text: str = "GAME CHANNEL", fill=PINK, dark=hexc("#C23A66"), light=hexc("#FF9DBA")) -> np.ndarray:
    """両端に切れ込みのあるリボン。"""
    m = text_mask(text, FONT_8BIT, 8)
    txt = styled(m, WHITE, WHITE, outlines=[(INK, 1)])
    th, tw = txt.shape[:2]
    bw, bh = tw + 8, th + 3
    tail = 7
    W, H = bw + 2 * tail + 2, bh + 6
    out = blank(W, H)
    yy, xx = np.mgrid[0:H, 0:W]
    # しっぽ（本体より少し下、切れ込み付き）
    notch = (bh + 1) // 2
    tl = (xx >= 1) & (xx < tail + 3) & (yy >= 3) & (yy < bh + 4) & (xx - 1 >= notch - np.abs(yy - 3 - notch))
    tr = (xx < W - 1) & (xx >= W - tail - 3) & (yy >= 3) & (yy < bh + 4) & (W - 2 - xx >= notch - np.abs(yy - 3 - notch))
    for t in (tl, tr):
        out[dilate(t, 1, diag=False)] = INK
        out[t] = dark
    body = (xx >= tail + 1) & (xx < tail + 1 + bw) & (yy >= 1) & (yy < 1 + bh)
    out[dilate(body, 1, diag=False)] = INK
    out[body] = fill
    out[body & (yy == 1)] = light
    blit(out, txt, tail + 1 + (bw - tw) // 2, 1 + (bh - th) // 2)
    return out


def telop_chars(text: str, fill_top=WHITE, fill_bottom=CREAM, accent=ORANGE, font=FONT_DOT, size=16,
                bold=True) -> list[np.ndarray | None]:
    """1文字ずつ別スプライトにしたテロップ（文字ごとに跳ねさせるため）。スペースは None。"""
    cells = glyph_cells(text, font, size, bold=bold)
    out = []
    for ch, m in zip(text, cells):
        if ch.isspace():
            out.append(None)
            continue
        out.append(styled(m, fill_top, fill_bottom, outlines=[(INK, 1), (accent, 1), (INK, 1)], shadow=(INK, 0, 1),
                          bands=2))
    return out


def text_sprite(text: str, fill=WHITE, fill_bottom=None, accent=None, font=FONT_DOT, size=16, bold=True,
                shadow=True, outline=True) -> np.ndarray:
    """テロップ風の文字。accent を指定すると 黒→accent→黒 の3重フチになる。"""
    m = text_block(text, font=font, size=size, bold=bold)
    if not outline:
        outs = []
    elif accent:
        outs = [(INK, 1), (accent, 1), (INK, 1)]
    else:
        outs = [(INK, 1)]
    return styled(m, fill, fill_bottom or fill, outlines=outs, shadow=(INK, 0, 1) if shadow else None, bands=2)


# ---------------------------------------------------------------- 終了画面の部品


def video_frame(inner_w: int, inner_h: int) -> np.ndarray:
    """YouTube の「動画」要素を置く枠（グリッド単位）。内側は半透明で、要素を置けば隠れる。"""
    b = 3  # 枠の太さ
    W, H = inner_w + 2 * b, inner_h + 2 * b
    out = blank(W + 2, H + 4)
    yy, xx = np.mgrid[0:H + 4, 0:W + 2]
    outer = round_rect_mask(H + 4, W + 2, 0, 0, W + 1, H + 1, 4)
    out[shift(outer, 0, 2)] = INK
    out[outer] = INK
    ring = round_rect_mask(H + 4, W + 2, 1, 1, W, H, 3)
    out[ring] = WHITE
    out[ring & (yy >= H - 1)] = hexc("#C9D6EA")
    inner_line = round_rect_mask(H + 4, W + 2, b, b, W + 1 - b, H + 1 - b, 1)
    out[inner_line] = INK
    inner = round_rect_mask(H + 4, W + 2, b + 1, b + 1, W - b, H - b, 1)
    out[inner] = (22, 32, 78, 225)
    stripe = inner & (((xx + yy) % 8) < 1)
    out[stripe] = (48, 64, 128, 225)
    # まん中の再生マーク
    cx, cy = (W + 2) / 2, (H + 2) / 2
    circ = ellipse_mask(H + 4, W + 2, cx, cy, 11, 11)
    out[circ] = (64, 84, 150, 235)
    tri = (xx >= cx - 3) & (xx <= cx + 5) & (np.abs(yy + 0.5 - cy) <= (cx + 5 - xx) * 0.62)
    out[tri & circ] = (200, 214, 240, 235)
    return out


def tab(text: str, fill=YELLOW, dark=hexc("#E08A1E")) -> np.ndarray:
    """枠の上に乗せる見出しタブ。"""
    txt = text_sprite(text, INK, shadow=False, outline=False)
    th, tw = txt.shape[:2]
    w, h = tw + 8, th + 5
    out = blank(w + 2, h + 2)
    body = round_rect_mask(h + 2, w + 2, 1, 1, w, h + 1, 2)
    out[dilate(body, 1, diag=False)] = INK
    out[body] = fill
    out[body & (np.arange(h + 2)[:, None] >= h - 1)] = dark
    blit(out, txt, 1 + (w - tw) // 2, 1 + (h - th) // 2)
    return out


def subscribe_ring(r_in: float, thick: int, t: float) -> np.ndarray:
    """「登録」要素を置く丸い枠。白い粒がくるくる回る。"""
    R = r_in + thick + 2
    S = int(2 * R + 4)
    c = S / 2
    yy, xx = np.mgrid[0:S, 0:S]
    d = np.hypot(xx + 0.5 - c, yy + 0.5 - c)
    out = blank(S, S)
    out[d <= r_in] = (255, 255, 255, 60)
    band = (d > r_in + 1) & (d <= r_in + 1 + thick)
    out[(d > r_in) & (d <= r_in + 2 + thick)] = INK
    out[band] = RED
    out[band & (d > r_in + thick - 0.5)] = hexc("#C62828")
    out[band & (d <= r_in + 2)] = hexc("#FF8A8A")
    ang = (np.arctan2(yy + 0.5 - c, xx + 0.5 - c) / (2 * np.pi) + 1.0) % 1.0
    dots = band & (((ang * 12 + t * 0.9) % 1.0) < 0.22) & (np.abs(d - (r_in + 1 + thick / 2)) < thick / 2 - 0.3)
    out[dots] = WHITE
    return out


def text_block(text: str, shown: int | None = None, font=FONT_DOT, size=16, bold=True, gap=3) -> np.ndarray:
    """複数行テキストのマスク。shown 文字目までだけ描く（枠の大きさは全文で固定＝タイプライター用）。"""
    lines = text.split("\n")
    rows = []
    count = 0
    for ln in lines:
        cells = glyph_cells(ln, font, size, bold=bold)
        parts = []
        for m in cells:
            count += 1
            parts.append(m if shown is None or count <= shown else np.zeros_like(m))
        rows.append(np.concatenate(parts, axis=1))
    w = max(r.shape[1] for r in rows)
    h = sum(r.shape[0] for r in rows) + gap * (len(rows) - 1)
    out = np.zeros((h, w), bool)
    y = 0
    for r in rows:
        x = (w - r.shape[1]) // 2
        out[y:y + r.shape[0], x:x + r.shape[1]] = r
        y += r.shape[0] + gap
    return out


def bubble(text: str, tail: str = "left", shown: int | None = None) -> np.ndarray:
    """吹き出し（ドット）。text は改行可。shown で表示する文字数を指定。"""
    m = text_block(text, shown)
    th, tw = m.shape
    pad_x, pad_y = 7, 5
    w, h = tw + pad_x * 2, th + pad_y * 2
    W, H = w + 2, h + 10
    out = blank(W, H)
    body = round_rect_mask(H, W, 1, 1, w, h, 5)
    yy, xx = np.mgrid[0:H, 0:W]
    if tail == "left":
        tl = (yy > h - 2) & (yy <= h + 7) & (xx >= 8) & (xx <= 8 + (h + 7 - yy) * 0.9) & (xx >= 8 + (yy - h) * 0.5)
    else:
        tl = (yy > h - 2) & (yy <= h + 7) & (xx <= w - 8) & (xx >= w - 8 - (h + 7 - yy) * 0.9)
    shape = body | tl
    out[shift(dilate(shape, 1, diag=False), 1, 2)] = (42, 30, 46, 110)
    out[dilate(shape, 1, diag=False)] = INK
    out[shape] = WHITE
    out[shape & (yy >= h - 1) & ~tl] = hexc("#E3ECF7")
    ink = np.zeros_like(shape)
    ink[1 + pad_y:1 + pad_y + th, 1 + pad_x:1 + pad_x + tw] = m
    out[ink] = INK
    return out


THUMB = [
    ".......kkk..........",
    "......kwyyk.........",
    "......kyyyk.........",
    ".....kyyyyk.........",
    ".....kyyyok.........",
    "....kyyyyk..........",
    "...kyyyyyk..........",
    "kkkkyyyyyykkkkkkk...",
    "kuuukyyyyyyyyyyyyk..",
    "kuuukyyyyyyyyyyyok..",
    "kuuukyyyyyykkkkkkk..",
    "kuuukyyyyyyyyyyyyyk.",
    "kuuukyyyyyyyyyyyyok.",
    "kuuukyyyyyykkkkkkk..",
    "kUUukyyyyyyyyyyyyk..",
    "kUUukyyyyyyyyyyyok..",
    "kUUukyyyyyykkkkkk...",
    "kUUukoyyyyyyyyyyk...",
    "kUUukooooooooook....",
    "kkkkkkkkkkkkkkkk....",
]

HAND = [
    "....kk.......",
    "...kwwk......",
    "...kwwk......",
    "...kwwk......",
    "...kwwkkk....",
    "...kwwkwwkk..",
    "kk.kwwkwwkwkk",
    "kwkkwwwwwwwwk",
    "kwwkwwwwwwwwk",
    ".kwwwwwwwwwwk",
    ".kwwwwwwwwwk.",
    "..kwwwwwwwwk.",
    "...kwwwwwwk..",
    "...kssssssk..",
    "...kkkkkkkk..",
]


def thumb_sprite() -> np.ndarray:
    return from_strings(THUMB, {"k": INK, "y": YELLOW, "o": hexc("#E08A1E"), "w": WHITE, "u": hexc("#3FA2F5"),
                                "U": hexc("#2F7FE6")})


def hand_sprite() -> np.ndarray:
    return from_strings(HAND, {"k": INK, "w": WHITE, "s": hexc("#C9D6EA")})
