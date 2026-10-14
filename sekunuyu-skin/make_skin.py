#!/usr/bin/env python3
"""せくぬゆのマイクラ用スキン（64×64・クラシック体型）を作る。

    python3 make_skin.py   # sekunuyu_skin.png（スキン本体）と preview.png（見た目の確認用）を書き出す

サムネイルのキャラと同じ見た目（黒髪・白T・ジーンズ・黄色いヘッドセット）。
前髪のふくらみとヘッドセットは、頭の2枚目のレイヤー（帽子レイヤー）に描いている。
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
THUMB = os.path.join(HERE, "..", "sekunuyu-thumbnail")   # 確認用の 3D 描画はサムネと同じレンダラーを使う

PAL = {
    "h": "#1C1820", "H": "#3E3848", "g": "#2B2533",      # 髪・つや・ふち
    "s": "#F2C29B", "S": "#D9A07A", "n": "#C98A66",      # はだ・影・耳
    "w": "#FFFFFF", "k": "#14101A", "m": "#9A4A44",      # 白目・黒目・口
    "T": "#F7F7F7", "u": "#D9DCE6", "U": "#B9BDCB",      # 白T・影・濃い影
    "j": "#2E3F6E", "J": "#22305A",                      # ジーンズ・影
    "o": "#3C3C46", "O": "#E6E6EA",                      # くつ・くつ底
    "y": "#FFD23F", "Y": "#E08A1E", "L": "#FFF0A0",      # 黄色（ヘッドセット・たんぽぽ）・影・光
    "B": "#34364A",                                      # ヘッドセットのバンドとマイク
}

# 各パーツのスキン画像上の位置（左上）と大きさ（幅, 高さ, 奥行き）。Minecraft の決まりどおり
PARTS = {
    "head": ((0, 0), (8, 8, 8)),
    "hat": ((32, 0), (8, 8, 8)),
    "body": ((16, 16), (8, 12, 4)),
    "arm": ((40, 16), (4, 12, 4)),       # 右腕
    "arm_l": ((32, 48), (4, 12, 4)),
    "leg": ((0, 16), (4, 12, 4)),        # 右足
    "leg_l": ((16, 48), (4, 12, 4)),
}


def face_rects(u, v, w, h, d):
    """箱の6面がスキン画像のどこに入るか。right/left はキャラから見た右/左。"""
    return {"top": (u + d, v, w, d), "bottom": (u + d + w, v, w, d),
            "right": (u, v + d, d, h), "front": (u + d, v + d, w, h),
            "left": (u + d + w, v + d, d, h), "back": (u + 2 * d + w, v + d, w, h)}


# ---------------------------------------------------------------- ドット絵（"." は透明）
# 横の面は、right（キャラの右側）は右はしが顔側、left（キャラの左側）は左はしが顔側

HEAD = {
    "top": ["hhhhhhhh"] * 8,
    "bottom": ["SSSSSSSS"] * 8,
    "front": [
        "hhhhhhhh",
        "hhhhhhhh",
        "hssssssh",
        "hwksskwh",
        "swksskws",
        "ssssssss",
        "sssmmsss",
        "ssssssss",
    ],
    "right": [
        "hhhhhhhh",
        "hhhhhhhh",
        "hhhhhhhh",
        "hhhhnssh",
        "hhhhnsss",
        "hhhsssss",
        "hhssssss",
        "hsssssss",
    ],
    "back": ["hhhhhhhh", "hhhhhhhh", "hhhhhhhh", "hhhhhhhh", "hhhhhhhh", "hhhhhhhh", "hhhhhhhh", "hhSSSShh"],
}
HEAD["left"] = [r[::-1] for r in HEAD["right"]]

# 帽子レイヤー：前髪のふくらみ・ヘッドセット（耳あて、頭の上のバンド、右側のマイク）
HAT = {
    "top": ["hhhhhhhh", "hhghhhgh", "hhghhhgh", "BBBBBBBB", "BBBBBBBB", "hghhhghh", "hghhhghh", "gggggggg"],
    "bottom": ["........"] * 8,
    "front": [
        "gggggggg",
        "hhhhhhhh",
        "hhh.hh.h",
        "y......y",
        "y......y",
        "B......y",
        "BBy....Y",
        "........",
    ],
    "right": [
        "hhhBBhhh",
        "hhhBBhhh",
        "hh.BB.h.",
        "..LyyY..",
        "..yyyY..",
        "..yyyYBB",
        "..YYYY..",
        "........",
    ],
    "left": [
        "hhhBBhhh",
        "hhhBBhhh",
        ".h.BB.hh",
        "..LyyY..",
        "..yyyY..",
        "..yyyY..",
        "..YYYY..",
        "........",
    ],
    "back": ["gggggggg", "hhghhhgh", "hhghhhgh", "hhhhghhh", "hghhghhh", "h.hhhh.h", "........", "........"],
}

BODY = {
    "top": ["TTTTTTTT"] * 4,
    "bottom": ["uuuuuuuu"] * 4,
    "front": [
        "TTssssTT",
        "TTTssTTT",
        "TTTTTTTT",
        "TTyTTTTT",
        "TyYyTTTT",
        "TTyTTTTT",
        "TTTTTTTT",
        "TTTTTTTT",
        "TTTTTTTT",
        "uTTTTTTu",
        "uTTTTTTu",
        "uuuuuuuu",
    ],
    "back": ["TTuuuuTT"] + ["TTTTTTTT"] * 8 + ["uTTTTTTu", "uTTTTTTu", "uuuuuuuu"],
    "right": ["TTTT"] * 9 + ["uTTu", "uTTu", "uuuu"],
}
BODY["left"] = BODY["right"]

ARM = {   # 右腕（半そで）
    "top": ["TTTT"] * 4,
    "bottom": ["SSSS"] * 4,
    "front": ["TTTT", "TTTT", "TTTT", "uuuu"] + ["ssss"] * 7 + ["SSSS"],
    "back": ["TTTT", "TTTT", "TTTT", "uuuu"] + ["ssss"] * 7 + ["SSSS"],
    "right": ["TTTT", "TTTT", "TTTT", "uuuu"] + ["ssss"] * 7 + ["SSSS"],
    "left": ["uuuu", "uuuu", "uuuu", "UUUU"] + ["SSSS"] * 8,     # 体側（かげ）
}

LEG = {   # 右足（ジーンズ＋スニーカー）
    "top": ["jjjj"] * 4,
    "bottom": ["OOOO"] * 4,
    "front": ["jjjJ"] * 8 + ["JJJJ", "oooo", "oooo", "OOOO"],
    "back": ["Jjjj"] * 8 + ["JJJJ", "oooo", "oooo", "OOOO"],
    "right": ["jjjj"] * 8 + ["JJJJ", "oooo", "oooo", "OOOO"],
    "left": ["JJJJ"] * 8 + ["JJJJ", "oooo", "oooo", "OOOO"],     # 内側（かげ）
}


def mirror(part: dict) -> dict:
    """左右反転（右腕 → 左腕など）。"""
    flip = lambda rows: [r[::-1] for r in rows]
    return {"top": flip(part["top"]), "bottom": flip(part["bottom"]), "front": flip(part["front"]),
            "back": flip(part["back"]), "right": flip(part["left"]), "left": flip(part["right"])}


DESIGN = {"head": HEAD, "hat": HAT, "body": BODY, "arm": ARM, "arm_l": mirror(ARM), "leg": LEG,
          "leg_l": mirror(LEG)}


def rgba(ch: str) -> tuple:
    if ch == ".":
        return (0, 0, 0, 0)
    c = PAL[ch]
    return (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16), 255)


def build_skin() -> np.ndarray:
    skin = np.zeros((64, 64, 4), np.uint8)
    for name, ((u, v), (w, h, d)) in PARTS.items():
        for face, (x, y, fw, fh) in face_rects(u, v, w, h, d).items():
            rows = DESIGN[name][face]
            assert len(rows) == fh and all(len(r) == fw for r in rows), (name, face)
            skin[y:y + fh, x:x + fw] = [[rgba(ch) for ch in r] for r in rows]
            if name != "hat":   # 体の1枚目のレイヤーはすき間なく塗る（透明だとゲームで黒く見える）
                assert (skin[y:y + fh, x:x + fw, 3] == 255).all(), (name, face)
    return skin


# ---------------------------------------------------------------- 確認用の画像


def skin_to_textures(skin: np.ndarray) -> dict:
    """スキン画像を、サムネ用の 3D モデル（lib/models.player）が使う形に切り分ける。"""
    T = {}
    for name, ((u, v), (w, h, d)) in PARTS.items():
        T[name] = {f: skin[y:y + fh, x:x + fw].copy() for f, (x, y, fw, fh) in face_rects(u, v, w, h, d).items()}
    return T


def render_view(T, cam_pos, w, h, pose, ss=2, target=(0, 1.0, 0), fov=34):
    sys.path.insert(0, THUMB)
    from lib.models import player
    from lib.post import tonemap
    from lib.voxel import Camera, Lights, render
    cam = Camera(cam_pos, target, fov, w * ss, h * ss)
    # 光はカメラ側の右上から（どの向きから見ても明るく）＋反対側から弱い光
    c = np.array(cam_pos, float)
    key = c / np.linalg.norm(c) + np.array([0, 0.9, 0]) + 0.5 * cam.R[0]
    L = Lights(ambient=(0.42, 0.42, 0.48), dirs=[(tuple(key), (0.72, 0.7, 0.66)),
                                                 (tuple(-c * np.array([1, -0.3, 1])), (0.22, 0.24, 0.3))])
    faces, _ = player(T, None, None, (0, 0, 0), pose)
    color, depth, _ = render(faces, cam, L)
    img = Image.fromarray((np.clip(tonemap(color), 0, 1) * 255 + 0.5).astype(np.uint8))
    mask = Image.fromarray((np.isfinite(depth) * 255).astype(np.uint8))
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    return out.resize((w, h), Image.LANCZOS)


def preview(skin: np.ndarray) -> Image.Image:
    sys.path.insert(0, THUMB)
    from lib import overlay as ov
    from lib.fonts import BLACK, ensure_fonts
    from lib.models import Pose
    ensure_fonts()
    T = skin_to_textures(skin)
    W, H = 1280, 720
    bg = ov.gradient((W, H), [(0, (196, 226, 255)), (1, (236, 246, 255))]).convert("RGBA")
    pose = Pose(arm_r=(-12, 4), arm_l=(12, 4), leg_r=8, leg_l=-8)
    front = render_view(T, (1.7, 1.55, 4.2), 380, 600, pose)
    back = render_view(T, (-1.9, 1.6, -4.1), 380, 600, pose)
    bg.alpha_composite(front, (20, 90))
    bg.alpha_composite(back, (390, 90))
    # スキン画像そのもの（透明な所はチェック柄）
    k = 7
    flat = Image.fromarray(skin, "RGBA").resize((64 * k, 64 * k), Image.NEAREST)
    yy, xx = np.mgrid[0:64 * k, 0:64 * k]
    chk = np.where(((yy // (2 * k) + xx // (2 * k)) % 2)[..., None] == 0, 236, 214).astype(np.uint8)
    board = Image.fromarray(np.dstack([chk, chk, chk, np.full_like(chk, 255)]), "RGBA")
    board.alpha_composite(flat)
    fx, fy = 790, 150
    d = ImageDraw.Draw(bg)
    d.rectangle((fx - 6, fy - 6, fx + 64 * k + 5, fy + 64 * k + 5), fill=(60, 70, 96))
    bg.alpha_composite(board, (fx, fy))
    f1, f2 = ov.font(BLACK, 40), ov.font(BLACK, 26)
    d.text((30, 26), "せくぬゆ スキン（クラシック体型・64×64）", font=f1, fill=(34, 40, 64))
    for x, label in [(210, "正面"), (580, "うしろ"), (fx + 32 * k, "スキン画像")]:
        d.text((x, 666), label, font=f2, fill=(34, 40, 64), anchor="mm")
    return bg.convert("RGB")


def main():
    skin = build_skin()
    Image.fromarray(skin, "RGBA").save(os.path.join(HERE, "sekunuyu_skin.png"))
    preview(skin).save(os.path.join(HERE, "preview.png"))
    print("sekunuyu_skin.png / preview.png を書き出しました")


if __name__ == "__main__":
    main()
