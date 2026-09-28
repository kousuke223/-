#!/usr/bin/env python3
"""せくぬゆ チャンネルのエンディング動画を作る。

    python3 make_ending.py            # ending / opening / 終了画面ガイド をまとめて書き出し
    python3 make_ending.py --only ending
    python3 make_ending.py --preview  # 確認用の静止画だけ書き出す

出力: output/sekunuyu_ending.mp4（20秒）, output/sekunuyu_opening.mp4（約4秒）,
      output/endscreen_guide.png
"""
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lib import timeline as T  # noqa: E402

W, H = 1920, 1080
FPS = 30

# ---------------------------------------------------------------- 文言（ここを書き換えればOK）
TEXT = {
    "name": "せくぬゆ",
    "tagline": "GAME CHANNEL",
    "title1": "ご視聴ありがとう！",
    "title2": "また見てね！",
    "bubble1": "チャンネル登録\nよろしくね！",
    "bubble2": "バイバーイ！",
    "label1": "おすすめ動画",
    "label2": "最新の動画",
    "like": "高評価も\nおねがい！",
}

# ---------------------------------------------------------------- フォント（Google Fonts / OFL）
FONTS = {
    "DotGothic16-Regular.ttf": "DotGothic16",
    "PressStart2P-Regular.ttf": "Press+Start+2P",
    "MPLUSRounded1c-Bold.ttf": "M+PLUS+Rounded+1c:wght@700",
}


def ensure_fonts() -> None:
    fdir = os.path.join(HERE, "fonts")
    os.makedirs(fdir, exist_ok=True)
    for fname, family in FONTS.items():
        path = os.path.join(fdir, fname)
        if os.path.exists(path):
            continue
        print(f"フォントをダウンロード中: {fname}")
        req = urllib.request.Request(f"https://fonts.googleapis.com/css2?family={family}",
                                     headers={"User-Agent": "Wget/1.12"})
        css = urllib.request.urlopen(req).read().decode()
        url = re.search(r"url\((https://[^)]+\.ttf)\)", css).group(1)
        urllib.request.urlretrieve(url, path)


# ---------------------------------------------------------------- イージング


def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def ease_out_cubic(x):
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def ease_in_cubic(x):
    x = clamp01(x)
    return x ** 3


def ease_out_back(x, s=1.70158):
    x = clamp01(x)
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def ease_out_bounce(x):
    x = clamp01(x)
    n, d = 7.5625, 2.75
    if x < 1 / d:
        return n * x * x
    if x < 2 / d:
        x -= 1.5 / d
        return n * x * x + 0.75
    if x < 2.5 / d:
        x -= 2.25 / d
        return n * x * x + 0.9375
    x -= 2.625 / d
    return n * x * x + 0.984375


# ---------------------------------------------------------------- 描画ヘルパー


def pil(arr: np.ndarray, s: int = 1) -> Image.Image:
    from lib.pixelart import up
    return Image.fromarray(up(arr, s), "RGBA")


def put(frame: Image.Image, img: Image.Image, x: float, y: float, anchor: str = "tl", scale: float = 1.0,
        sx: float | None = None, sy: float | None = None, alpha: float = 1.0) -> None:
    """img を frame に重ねる。anchor: tl(左上) / c(中心) / b(下中央) / bl(左下)。拡大はニアレストでドットを保つ。"""
    if alpha <= 0 or scale <= 0:
        return
    kx = scale if sx is None else scale * sx
    ky = scale if sy is None else scale * sy
    if abs(kx - 1) > 1e-3 or abs(ky - 1) > 1e-3:
        w, h = max(1, round(img.width * kx)), max(1, round(img.height * ky))
        img = img.resize((w, h), Image.NEAREST)
    if alpha < 1:
        a = img.getchannel("A").point(lambda v: int(v * alpha))
        img = img.copy()
        img.putalpha(a)
    w, h = img.size
    if anchor == "c":
        x, y = x - w / 2, y - h / 2
    elif anchor == "b":
        x, y = x - w / 2, y - h
    elif anchor == "bl":
        y = y - h
    x, y = int(round(x)), int(round(y))
    # 画面外の部分を切り落としてから合成
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame.width, x + w), min(frame.height, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    if (x0, y0, x1, y1) != (x, y, x + w, y + h):
        img = img.crop((x0 - x, y0 - y, x1 - x, y1 - y))
    frame.alpha_composite(img, (x0, y0))


class Particles:
    """星・キラキラ・ハート・コインのはじける演出。"""

    def __init__(self, sprites: dict[str, Image.Image], coin_frames: list[Image.Image]):
        self.sprites = sprites
        self.coin = coin_frames
        self.items = []

    def burst(self, t0, x, y, n, seed, speed=(500, 1100), spread=None, kinds=("star", "sparkle", "heart", "coin"),
              life=(0.9, 1.4), up=350, gravity=1500, area=(0, 0)):
        rng = np.random.default_rng(seed)
        for i in range(n):
            ang = rng.uniform(0, 2 * math.pi) if spread is None else rng.uniform(*spread)
            sp = rng.uniform(*speed)
            px = x + rng.uniform(-area[0], area[0])
            py = y + rng.uniform(-area[1], area[1])
            self.items.append(dict(t0=t0, x=px, y=py, vx=math.cos(ang) * sp, vy=math.sin(ang) * sp - up,
                                   life=rng.uniform(*life), kind=kinds[i % len(kinds)], g=gravity,
                                   ph=rng.uniform(0, 1)))

    def draw(self, frame: Image.Image, t: float) -> None:
        for p in self.items:
            a = t - p["t0"]
            if a < 0 or a > p["life"]:
                continue
            x = p["x"] + p["vx"] * a
            y = p["y"] + p["vy"] * a + 0.5 * p["g"] * a * a
            fade = clamp01((p["life"] - a) / 0.3)
            if p["kind"] == "coin":
                img = self.coin[int((a * 8 + p["ph"] * 4)) % 4]
            else:
                img = self.sprites[p["kind"]]
            if p["kind"] == "sparkle" and int((a + p["ph"]) * 12) % 2:
                continue  # チカチカ
            put(frame, img, x, y, "c", alpha=fade)


# ---------------------------------------------------------------- 素材の準備


class Assets:
    def __init__(self, avatar_path: str):
        from lib import ui
        from lib.avatar import make_avatar
        from lib.pixelart import coin_sprite, heart_sprite, sparkle_sprite, star_sprite
        from lib.world import CloudFrame, Meadow, WorldMap

        self.ui = ui
        self.world = WorldMap()
        self.clouds = CloudFrame()
        self.meadow = Meadow()
        av = make_avatar(avatar_path, 72)
        self.avatar_a = pil(av, 6)
        self.avatar_b = pil(av, 4)
        self.logo, self.logo_fill = ui.logo(TEXT["name"], ui.BOX_COLORS["blue"])
        self.logo_img = pil(self.logo, 7)
        self.ribbon = pil(ui.ribbon(TEXT["tagline"]), 5)
        self.title1 = [None if c is None else pil(c, 6) for c in ui.telop_chars(TEXT["title1"])]
        self.title2 = [None if c is None else pil(c, 6) for c in ui.telop_chars(TEXT["title2"])]
        self.frame = pil(ui.video_frame(160, 90), 4)
        self.tabs = [pil(ui.tab(TEXT["label1"]), 3), pil(ui.tab(TEXT["label2"]), 3)]
        self.thumb = pil(ui.thumb_sprite(), 5)
        self.like = pil(ui.text_sprite(TEXT["like"], ui.WHITE, ui.CREAM, accent=ui.ORANGE), 4)
        self.hand = pil(ui.hand_sprite(), 5)
        self.heart_big = pil(heart_sprite(), 5)
        sprites = {"star": pil(star_sprite(), 5), "sparkle": pil(sparkle_sprite(), 5),
                   "heart": pil(heart_sprite(), 5), "ystar": pil(sparkle_sprite(ui.YELLOW), 5)}
        self.sparkle_big = pil(sparkle_sprite(), 6)
        self.coin = [pil(coin_sprite(i / 4), 5) for i in range(4)]
        self.sprites = sprites
        self._bubble_cache: dict = {}

    def bubble(self, text: str, shown: int) -> Image.Image:
        key = (text, shown)
        if key not in self._bubble_cache:
            self._bubble_cache[key] = pil(self.ui.bubble(text, shown=shown), 3)
        return self._bubble_cache[key]


# ---------------------------------------------------------------- 配置（px）

LOGO_C = (960, 560 + 172)      # ロゴの中心
AVATAR_A_BOTTOM = (960, 632)   # シーンAのアバター（下中央）
RIBBON_C = (960, 902)
TITLE_Y = 18
FRAME_INNER = [(250, 246), (1030, 246)]   # 動画要素を置く内側の左上（640x360）
RING_C = (960, 850)
AVATAR_B = (50, 1080 + 40)     # 左下
BUBBLE_TL = (300, 650)
THUMB_TL = (1215, 780)
LIKE_TL = (1350, 760)
CLICK_POINT = (1086, 972)      # 登録リングの右下のふち


class Renderer:
    def __init__(self, assets: Assets, mode: str = "ending"):
        self.a = assets
        self.mode = mode
        self.fx = Particles(assets.sprites, assets.coin)
        lx, ly = LOGO_C
        lw, lh = assets.logo_img.size
        # ロゴ着地のはじける星
        self.fx.burst(T.A_LOGO_HIT, lx, ly, 30, seed=1, area=(lw / 2.2, lh / 3))
        if mode == "ending":
            for i, tc in enumerate(T.B_CLICKS):
                self.fx.burst(tc, *CLICK_POINT, 12, seed=10 + i, speed=(300, 700), kinds=("star", "sparkle", "heart"),
                              life=(0.6, 0.9), up=250)
            self.fx.burst(T.B_FINAL, 960, TITLE_Y + 70, 24, seed=30, speed=(400, 900), area=(420, 20),
                          kinds=("star", "sparkle", "coin", "heart"), life=(1.0, 1.3), up=250, gravity=1300)
        self.title_w1 = sum(c.width for c in assets.title1 if c) if assets.title1 else 0

    # ------------------------------------------------ 雲のカーテン
    def curtain(self, t: float) -> float | None:
        o0, o1 = T.A_CURTAIN_OPEN
        c0, c1 = T.A_CURTAIN_CLOSE
        b0, b1 = T.B_CURTAIN_OPEN
        if t < o1:
            return ease_out_cubic((t - o0) / (o1 - o0))
        if c0 <= t < c1:
            return 1 - ease_in_cubic((t - c0) / (c1 - c0))
        if t >= c1 and (self.mode == "opening" or t < b0):
            return 0.0
        if b0 <= t < b1:
            return ease_out_cubic((t - b0) / (b1 - b0))
        return None

    # ------------------------------------------------ シーンA
    def scene_a(self, t: float) -> Image.Image:
        a = self.a
        from lib.pixelart import up
        big = Image.fromarray(up(a.world.render(t), 6), "RGBA")
        z = 1.0 + 0.22 * (1 - ease_out_cubic(t / 3.4))
        if z > 1.0005:
            cx, cy = 960, 540 + (z - 1) * 260
            big = big.resize((W, H), Image.BILINEAR, box=(cx - 960 / z, cy - 540 / z, cx + 960 / z, cy + 540 / z))
        frame = big
        frame.alpha_composite(Image.fromarray(up(a.clouds.frame_layer(t), 6), "RGBA"))
        # 星やコインはアバターとロゴの後ろから飛び出す
        self.fx.draw(frame, t)

        # アバター：ポンッと出て、ロゴ着地後は拍に合わせてぴょこぴょこ
        p = (t - T.A_AVATAR_POP) / 0.42
        if p > 0:
            s = ease_out_back(p, 2.2)
            bob = 0
            if t > T.A_LOGO_HIT:
                bob = -6 if ((t - T.A_LOGO_HIT) / T.BEAT) % 1.0 < 0.5 else 0
            put(frame, a.avatar_a, AVATAR_A_BOTTOM[0], AVATAR_A_BOTTOM[1] + bob, "b", scale=s)

        # ロゴ：上から落ちてきてドンッ！
        lx, ly = LOGO_C
        if t >= T.A_LOGO_DROP:
            q = (t - T.A_LOGO_DROP) / (T.A_LOGO_HIT - T.A_LOGO_DROP)
            if q < 1:
                s = 2.6 - 1.6 * q * q
                put(frame, a.logo_img, lx, ly, "c", scale=s, alpha=clamp01(q * 3))
            else:
                k = t - T.A_LOGO_HIT
                sq = 0.12 * math.exp(-k * 7) * math.cos(k * 26)
                img = a.logo_img
                sw0, sw1 = T.A_SHINE, T.A_SHINE + 0.5
                if sw0 <= t < sw1:
                    img = pil(self.shine(a.logo, a.logo_fill, (t - sw0) / (sw1 - sw0)), 7)
                put(frame, img, lx, ly, "c", sx=1 + sq, sy=1 - sq)
                # キラッと光る星（ロゴの角）
                for i, (dx, dy) in enumerate(((-0.47, -0.42), (0.46, -0.38), (0.44, 0.36), (-0.45, 0.33))):
                    ph = (t - T.A_LOGO_HIT - 0.4 - i * 0.37) % 1.5
                    if 0 <= ph < 0.36 and t > T.A_LOGO_HIT + 0.4:
                        sc = math.sin(ph / 0.36 * math.pi)
                        put(frame, a.sparkle_big, lx + dx * a.logo_img.width, ly + dy * a.logo_img.height, "c",
                            scale=sc)
        # リボン
        p = (t - T.A_RIBBON) / 0.3
        if p > 0:
            put(frame, a.ribbon, RIBBON_C[0], RIBBON_C[1], "c", scale=ease_out_back(p, 2.5))

        # 着地の瞬間のフラッシュ
        fl = 0.4 * (1 - clamp01((t - T.A_LOGO_HIT) / 0.14)) if t >= T.A_LOGO_HIT else 0
        if fl > 0:
            frame.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255, int(255 * fl))))
        # 画面の揺れ
        k = t - T.A_LOGO_HIT
        if 0 <= k < 0.32:
            amp = 18 * (1 - k / 0.32) ** 2
            dx = int(amp * math.sin(k * 97))
            dy = int(amp * math.cos(k * 83))
            shaken = Image.new("RGBA", (W, H), (255, 255, 255, 255))
            shaken.paste(frame, (dx, dy))
            frame = shaken
        return frame

    @staticmethod
    def shine(spr: np.ndarray, fill: np.ndarray, p: float) -> np.ndarray:
        out = spr.copy()
        h, w = fill.shape
        yy, xx = np.mgrid[0:h, 0:w]
        pos = -20 + p * (w + 40)
        band = (xx + yy * 0.6 - pos)
        m = fill & (band >= 0) & (band < 5)
        out[m] = (255, 255, 255, 255)
        m2 = fill & (band >= 6) & (band < 8)
        out[m2] = (255, 255, 255, 255)
        return out

    # ------------------------------------------------ シーンB
    def title(self, frame, t, chars, t_in, t_out=None):
        total = sum(c.width if c else 36 for c in chars)
        x = 960 - total / 2
        for i, c in enumerate(chars):
            if c is None:
                x += 36
                continue
            p = (t - (t_in + i * T.B_TITLE_STEP)) / 0.34
            if p <= 0:
                x += c.width
                continue
            y = TITLE_Y - 320 * (1 - ease_out_bounce(p))
            alpha = 1.0
            if p >= 1:
                w = math.sin(2 * math.pi * t * 1.25 - i * 0.55)
                y += -6 if w > 0.35 else (6 if w < -0.35 else 0)
            if t_out is not None and t >= t_out:
                q = (t - t_out - i * 0.03) / 0.22
                if q >= 1:
                    x += c.width
                    continue
                if q > 0:
                    y -= 260 * ease_in_cubic(q)
                    alpha = 1 - q
            put(frame, c, x, y, alpha=alpha)
            x += c.width

    def scene_b(self, t: float) -> Image.Image:
        a = self.a
        from lib.pixelart import up
        frame = Image.fromarray(up(a.meadow.render(t), 6), "RGBA")
        beat = ((t - T.B_MUSIC) / T.BEAT) % 1.0

        # 動画枠＋見出しタブ（左右からスライドイン）
        for i, (ix, iy) in enumerate(FRAME_INNER):
            p = (t - T.B_FRAMES - i * 0.08) / 0.55
            if p <= 0:
                continue
            off = (1 - ease_out_back(p, 1.2)) * 1300 * (-1 if i == 0 else 1)
            put(frame, a.frame, ix - 16 + off, iy - 16)
            tb = a.tabs[i]
            put(frame, tb, ix + off, iy - 16 - tb.height + 12)

        # 登録ボタンの丸枠
        p = (t - T.B_RING) / 0.4
        if p > 0:
            ring = pil(a.ui.subscribe_ring(40, 4, t), 4)
            pulse = 1 + 0.035 * max(0.0, math.cos(2 * math.pi * ((t - T.B_MUSIC) / (2 * T.BEAT)))) ** 12
            put(frame, ring, RING_C[0], RING_C[1], "c", scale=ease_out_back(p, 2.0) * pulse)

        # アバター（左下からひょこっと）
        p = (t - T.B_AVATAR) / 0.45
        if p > 0:
            y = AVATAR_B[1] + 520 * (1 - ease_out_back(p, 1.6))
            if t > T.B_AVATAR + 0.45:
                y += -8 if beat < 0.5 else 0
            if T.B_SWAP <= t < T.B_SWAP + 0.4:
                y -= 80 * math.sin(math.pi * (t - T.B_SWAP) / 0.4)
            put(frame, a.avatar_b, AVATAR_B[0], y, "bl")

        # 吹き出し
        self.speech(frame, t)

        # 高評価
        p = (t - T.B_THUMB) / 0.35
        if p > 0:
            s = ease_out_back(p, 2.2)
            bob = -5 if (t > T.B_THUMB + 0.35 and int((t - T.B_MUSIC) / T.BEAT) % 2 == 0) else 0
            th = a.thumb
            put(frame, th, THUMB_TL[0] + th.width / 2, THUMB_TL[1] + th.height / 2 + bob, "c", scale=s)
            lk = a.like
            put(frame, lk, LIKE_TL[0] + lk.width / 2, LIKE_TL[1] + lk.height / 2, "c", scale=s)
            # ハートがふわふわ上る
            k = 0
            while True:
                t0 = T.B_THUMB + 0.6 + k * 1.3
                if t0 > t:
                    break
                age = (t - t0) / 1.5
                if age < 1:
                    hx = THUMB_TL[0] + 45 + 18 * math.sin(age * 7 + k) + (k % 3 - 1) * 20
                    hy = THUMB_TL[1] + 10 - age * 120
                    put(frame, a.heart_big, hx, hy, "c", alpha=clamp01((1 - age) / 0.35))
                k += 1

        # タイトル
        if t < T.B_SWAP + 0.4:
            self.title(frame, t, a.title1, T.B_TITLE, t_out=T.B_SWAP)
        if t >= T.B_SWAP + 0.1:
            self.title(frame, t, a.title2, T.B_SWAP + 0.1)

        # カーソルがポチッ
        for tc in T.B_CLICKS:
            self.cursor(frame, t, tc)
        self.fx.draw(frame, t)
        return frame

    def speech(self, frame, t):
        a = self.a
        n1 = len(TEXT["bubble1"].replace("\n", ""))
        n2 = len(TEXT["bubble2"].replace("\n", ""))
        if t < T.B_BUBBLE:
            return
        if t < T.B_SWAP:
            text, t_pop, t_type, n = TEXT["bubble1"], T.B_BUBBLE, T.B_TYPE, n1
            close = 0.0
        elif t < T.B_SWAP + 0.12:
            text, t_pop, t_type, n = TEXT["bubble1"], T.B_BUBBLE, T.B_TYPE, n1
            close = (t - T.B_SWAP) / 0.12
        else:
            text, t_pop, t_type, n = TEXT["bubble2"], T.B_SWAP + 0.12, T.B_SWAP_TYPE, n2
            close = 0.0
        # タイプライター：1文字目は t_type ちょうどに出る（効果音と同じタイミング）
        shown = 0 if t < t_type else min(n, int((t - t_type) / T.B_TYPE_STEP) + 1)
        img = a.bubble(text, shown)
        s = ease_out_back((t - t_pop) / 0.25, 2.0) * (1 - close)
        if s > 0:
            # 吹き出しのしっぽ（左下）を中心に拡大
            ax, ay = BUBBLE_TL[0] + 8 * 3, BUBBLE_TL[1] + img.height
            put(frame, img, ax - 8 * 3 * s, ay - img.height * s, scale=s)

    def cursor(self, frame, t, tc):
        a = self.a
        t_in, t_out = tc - 0.6, tc + 0.5
        if not (t_in <= t < t_out + 0.45):
            return
        start = (1560, 1200)
        tx, ty = CLICK_POINT
        if t < tc - 0.1:
            p = ease_out_cubic((t - t_in) / 0.5)
        elif t < t_out:
            p = 1.0
        else:
            p = 1 - ease_in_cubic((t - t_out) / 0.45)
        x = start[0] + (tx - start[0]) * p
        y = start[1] + (ty - start[1]) * p
        press = 0.86 if tc <= t < tc + 0.12 else 1.0
        # 指先（スプライトの (4.5, 0) ドット）がクリック位置に来るように
        put(frame, a.hand, x - 4.5 * 5 * press, y, scale=press)

    # ------------------------------------------------ 1フレーム
    def render(self, t: float) -> Image.Image:
        from lib.pixelart import up
        if self.mode == "opening" or t < T.B_START:
            frame = self.scene_a(min(t, T.B_START - 1e-3))
        else:
            frame = self.scene_b(t)
        c = self.curtain(t)
        if c is not None and c < 0.999:
            frame.alpha_composite(Image.fromarray(up(self.a.clouds.curtain(c, t), 6), "RGBA"))
        return frame.convert("RGB")


# ---------------------------------------------------------------- 書き出し


def encode(renderer: Renderer, dur: float, audio: np.ndarray, out_path: str, crf: int = 16) -> None:
    from lib.chiptune import write_wav
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "audio.wav")
        write_wav(wav, audio)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
               "-i", wav,
               "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
               "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-tune", "animation",
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
               "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
               "-movflags", "+faststart", "-shortest", out_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        n = int(round(dur * FPS))
        for i in range(n):
            t = i / FPS
            proc.stdin.write(renderer.render(t).tobytes())
            if i % FPS == 0:
                print(f"  {out_path.split(os.sep)[-1]}: {t:4.1f}s / {dur:.1f}s", flush=True)
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit("ffmpeg でエラーが出ました")


def guide_image(renderer: Renderer, out_path: str) -> None:
    """YouTube Studio の終了画面をどこに置けばいいかの説明画像。"""
    img = renderer.render(12.0).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    fpath = os.path.join(HERE, "fonts", "MPLUSRounded1c-Bold.ttf")
    font = ImageFont.truetype(fpath, 40)
    small = ImageFont.truetype(fpath, 28)

    def dashed_rect(box, color, width=6, dash=24):
        x0, y0, x1, y1 = box
        for x in range(x0, x1, dash * 2):
            d.line([(x, y0), (min(x + dash, x1), y0)], fill=color, width=width)
            d.line([(x, y1), (min(x + dash, x1), y1)], fill=color, width=width)
        for y in range(y0, y1, dash * 2):
            d.line([(x0, y), (x0, min(y + dash, y1))], fill=color, width=width)
            d.line([(x1, y), (x1, min(y + dash, y1))], fill=color, width=width)

    def label(cx, cy, lines, color, f=font):
        """cx, cy を中心に、角丸の札に文字を書く。"""
        text = "\n".join(lines)
        bb = d.multiline_textbbox((0, 0), text, font=f, spacing=8, align="center")
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        x, y = cx - w / 2, cy - h / 2
        d.rounded_rectangle((x - 20, y - 14, x + w + 20, y + h + 14), 16, fill=color)
        d.multiline_text((x - bb[0], y - bb[1]), text, font=f, fill=(255, 255, 255, 255), spacing=8, align="center")

    red, blue = (235, 50, 60, 255), (30, 110, 240, 255)
    for i, (ix, iy) in enumerate(FRAME_INNER):
        d.rectangle((ix, iy, ix + 640, iy + 360), fill=(255, 64, 64, 60))
        dashed_rect((ix, iy, ix + 640, iy + 360), red)
        label(ix + 320, iy + 110, [f"動画要素 {i + 1}", "（615×345くらい）"], red)
    r = 160
    d.ellipse((RING_C[0] - r, RING_C[1] - r, RING_C[0] + r, RING_C[1] + r), fill=(40, 120, 255, 70),
              outline=blue, width=6)
    label(RING_C[0], RING_C[1], ["登録要素"], blue)
    label(1560, 1010, [f"表示するのは エンディング開始{T.ENDSCREEN_FROM:.0f}秒後 〜 最後まで",
                       "（20秒のエンディングなら 残り15秒）"], (20, 20, 40, 225), f=small)
    img.alpha_composite(ov)
    img.convert("RGB").save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--avatar", default=os.path.join(HERE, "assets", "avatar_cutout.png"),
                    help="背景を透過したアイコン画像（PNG）。ドット絵に変換して使う")
    ap.add_argument("--out", default=os.path.join(HERE, "output"))
    ap.add_argument("--only", choices=["ending", "opening", "guide"], action="append")
    ap.add_argument("--preview", action="store_true", help="確認用の静止画だけ書き出す")
    ap.add_argument("--crf", type=int, default=16)
    args = ap.parse_args()

    ensure_fonts()
    os.makedirs(args.out, exist_ok=True)
    from lib import soundtrack

    assets = Assets(args.avatar)
    todo = args.only or ["ending", "opening", "guide"]

    if args.preview:
        r = Renderer(assets, "ending")
        for t in (0.2, 0.5, 1.0, 1.1, 1.5, 2.1, 3.3, 3.8, 4.4, 4.9, 5.5, 6.5, 8.0, 12.0, 17.3, 19.5):
            r.render(t).save(os.path.join(args.out, f"preview_{t:05.2f}.png"))
        print("プレビューを書き出しました")
        return

    if "ending" in todo:
        r = Renderer(assets, "ending")
        audio = soundtrack.build_ending(len(TEXT["title1"]), len(TEXT["bubble1"].replace("\n", "")),
                                        len(TEXT["title2"]), len(TEXT["bubble2"].replace("\n", "")))
        encode(r, T.ENDING_DUR, audio, os.path.join(args.out, "sekunuyu_ending.mp4"), args.crf)
    if "opening" in todo:
        r = Renderer(assets, "opening")
        encode(r, T.OPENING_DUR, soundtrack.build_opening(), os.path.join(args.out, "sekunuyu_opening.mp4"),
               args.crf)
    if "guide" in todo:
        guide_image(Renderer(assets, "ending"), os.path.join(args.out, "endscreen_guide.png"))
    print("完了！ →", args.out)


if __name__ == "__main__":
    main()
