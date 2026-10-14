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
    "b": "#6A4330",                                      # まゆ毛（ちょっと茶色）
    "T": "#F7F7F7", "u": "#D9DCE6", "U": "#B9BDCB",      # 白T・影・濃い影
    "j": "#2E3F6E", "J": "#22305A",                      # ジーンズ・影
    "a": "#6F93D6", "A": "#5577BC",                      # 明るいデニム・影
    "c": "#B9CDF3", "C": "#93ADE2",                      # 折り返したすそ・影
    "q": "#FFF8DE", "Q": "#FFE68C",                      # ギンガムチェックのクリーム色・うすい黄色
    "o": "#3C3C46", "O": "#E6E6EA",                      # くつ・くつ底
    "y": "#FFD23F", "Y": "#E08A1E", "L": "#FFF0A0",      # 黄色（ヘッドセット・たんぽぽ）・影・光
    "B": "#34364A",                                      # ヘッドセットのバンドとマイク
    "D": "#28183F", "K": "#120A1A",                      # パーカーのフード（黒むらさき）・ふち
    "P": "#43296A", "p": "#573A86",                      # パーカーのむらさき・明るいむらさき
    "Z": "#CFCBDA", "z": "#8C8799",                      # ファスナーの歯（明るいグレー・グレー）
    "e": "#A93BEB", "E": "#E3A6FF",                      # エンダーの目（むらさき・うすむらさき）
    "V": "#C25BFF", "v": "#7B3BB5",                      # エンダーのつぶつぶ（明るい・暗い）
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


def mirror(part: dict) -> dict:
    """左右反転（右腕 → 左腕など）。"""
    flip = lambda rows: [r[::-1] for r in rows]
    return {"top": flip(part["top"]), "bottom": flip(part["bottom"]), "front": flip(part["front"]),
            "back": flip(part["back"]), "right": flip(part["left"]), "left": flip(part["right"])}


def paste(rows: list[str], patch: list[str], x0: int, y0: int) -> list[str]:
    """ドット絵の上に小さなドット絵を重ねる（"." の所はそのまま）。"""
    rows = list(rows)
    for dy, pr in enumerate(patch):
        r = list(rows[y0 + dy])
        for dx, ch in enumerate(pr):
            if ch != ".":
                r[x0 + dx] = ch
        rows[y0 + dy] = "".join(r)
    return rows


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

# ---------------------------------------------------------------- エンダーパーカー（フードをかぶった服）
# フードは帽子レイヤーに描く：エンダーマンのまだら模様、前に光る目、顔のまわりは黒いふち。
# ヘッドセットはフードの上から付ける。服にはエンダーのつぶつぶ（V / v）を散らす


def ender_tex(w: int, h: int, seed: int) -> list[str]:
    """エンダーマンっぽい黒とむらさきのまだら模様。"""
    rng = np.random.default_rng(seed)
    return ["".join("P" if x < 0.12 else "D" if x < 0.55 else "K" for x in row) for row in rng.random((h, w))]


HEAD_HOOD = dict(HEAD, front=[
    "hhhhhhhh",
    "hhhhhhhh",
    "hssssssh",
    "sbbssbbs",     # 茶色いまゆ毛（前髪はフードの中）
    "swksskws",
    "swksskws",
    "ssssssss",
    "sssmmsss",
])

CUP = [   # フードの横に付けるヘッドセット（"." はフードのまま）
    "...zz...",
    "...zz...",
    "...zz...",
    "..LyyY..",
    "..yyyY..",
    "..yyyY..",
    "..YYYY..",
    "........",
]

HAT_HOOD = {
    "top": paste(ender_tex(8, 8, 1), ["zzzzzzzz", "zzzzzzzz"], 0, 3),
    "bottom": ["........"] * 8,
    "front": [
        "DKDKKDKD",
        "EeEKKEeE",
        "KKKKKKKK",
        "K......K",
        "K......K",
        "K......K",
        "K......K",
        "K......K",
    ],
    "right": paste(ender_tex(8, 8, 3), CUP, 0, 0),
    "left": paste(ender_tex(8, 8, 4), CUP, 0, 0),
    "back": ender_tex(8, 8, 5),
}

BODY_HOOD = {
    "top": ender_tex(8, 4, 6),
    "bottom": ["KKKKKKKK"] * 4,
    "front": [
        "KPZssZPK",
        "KPPZzPPK",
        "KPPZZPPK",     # ファスナーの引き手
        "KVPzZPPK",
        "KPPZzPvK",
        "KPPzZPPK",
        "KPvZzPPK",
        "KPPzZPVK",
        "KPPZzPPK",
        "KVPzZPPK",
        "KPPZzPvK",
        "KKKzZKKK",
    ],
    "back": ender_tex(8, 3, 7) + ["KPPPPPPK", "KPVPPPPK", "KPPPPvPK", "KPPPPPPK", "KvPPPPPK", "KPPPPVPK",
                                   "KPPPPPPK", "KPPvPPPK", "KKKKKKKK"],
    "right": ["PPPP", "PvPP", "PPPP", "PPVP", "PPPP", "PPPP", "PvPP", "PPPP", "PPPP", "PPPV", "PPPP", "KKKK"],
}
BODY_HOOD["left"] = [r[::-1] for r in BODY_HOOD["right"]]

ARM_HOOD = {   # 右腕（長そで、ふちどり＋つぶつぶのそで口）
    "top": ender_tex(4, 4, 8),
    "bottom": ["SSSS"] * 4,
    "front": ["KPPK"] * 10 + ["VKvK", "ssss"],
    "back": ["KPPK"] * 10 + ["KvKV", "ssss"],
    "right": ["KPPK"] * 10 + ["vKVK", "ssss"],
    "left": ["KDDK"] * 10 + ["KVKv", "SSSS"],     # 体側（かげ）
}

# 服の組み合わせ。どれを使うかは OUTFIT で選ぶ
OUTFITS = {
    "ender_hoodie": {"head": HEAD_HOOD, "hat": HAT_HOOD, "body": BODY_HOOD, "arm": ARM_HOOD},   # エンダーパーカー
    "tee": {"head": HEAD, "hat": HAT, "body": BODY, "arm": ARM},                                # 白T（最初の服）
}
OUTFIT = "ender_hoodie"


def gingham(h: int) -> list[str]:
    """黄色のギンガムチェック（2ドットごと）。"""
    return ["".join("y" if (y // 2) % 2 == 0 and (x // 2) % 2 == 0 else
                    "Q" if (y // 2) % 2 == 0 or (x // 2) % 2 == 0 else "q" for x in range(4)) for y in range(h)]


# ズボン（右足の形。左足は左右反転して作る）。どれを使うかは PANTS_STYLE で選ぶ
PANTS = {
    # 明るいデニムをロールアップ＋白ソックス＋黄色いスニーカー（右ひざにたんぽぽのワッペン）
    "rollup": {
        "top": ["aaaa"] * 4,
        "bottom": ["TTTT"] * 4,
        "front": ["aaaA"] * 6 + ["cccc", "CCCC", "TTTT", "yyyy", "yyyy", "TTTT"],
        "back": ["Aaaa"] * 6 + ["cccc", "CCCC", "TTTT", "yyyy", "yyyy", "TTTT"],
        "right": ["aaAa"] * 6 + ["cccc", "CCCC", "TTTT", "yyyy", "yyyy", "TTTT"],
        "left": ["AAAA"] * 6 + ["CCCC", "CCCC", "uuuu", "YYYY", "YYYY", "uuuu"],    # 内側（かげ）
    },
    # 黄色のギンガムチェック＋白ソックス＋黒いスニーカー
    "gingham": {
        "top": ["qqqq"] * 4,
        "bottom": ["OOOO"] * 4,
        **{f: gingham(9) + ["TTTT", "oooo", "OOOO"] for f in ("front", "back", "right", "left")},
    },
    # デニムのショートパンツ＋黄色と白のボーダーソックス
    "shorts": {
        "top": ["aaaa"] * 4,
        "bottom": ["OOOO"] * 4,
        "front": ["aaaA"] * 4 + ["cccc", "ssss", "yyyy", "TTTT", "yyyy", "TTTT", "oooo", "OOOO"],
        "back": ["Aaaa"] * 4 + ["cccc", "ssss", "yyyy", "TTTT", "yyyy", "TTTT", "oooo", "OOOO"],
        "right": ["aaAa"] * 4 + ["cccc", "ssss", "yyyy", "TTTT", "yyyy", "TTTT", "oooo", "OOOO"],
        "left": ["AAAA"] * 4 + ["CCCC", "SSSS", "YYYY", "uuuu", "YYYY", "uuuu", "oooo", "OOOO"],
    },
    # パーカーとおそろいのむらさき（ふちどり）＋つぶつぶの黒いくつ
    "ender": {
        "top": ["PPPP"] * 4,
        "bottom": ["KKKK"] * 4,
        "front": ["KPPK"] * 10 + ["KVKv", "KKKK"],
        "back": ["KPPK"] * 10 + ["vKVK", "KKKK"],
        "right": ["KPPK"] * 10 + ["KvKV", "KKKK"],
        "left": ["KDDK"] * 10 + ["VKvK", "KKKK"],
    },
    # 最初の濃いジーンズ＋スニーカー
    "jeans": {
        "top": ["jjjj"] * 4,
        "bottom": ["OOOO"] * 4,
        "front": ["jjjJ"] * 8 + ["JJJJ", "oooo", "oooo", "OOOO"],
        "back": ["Jjjj"] * 8 + ["JJJJ", "oooo", "oooo", "OOOO"],
        "right": ["jjjj"] * 8 + ["JJJJ", "oooo", "oooo", "OOOO"],
        "left": ["JJJJ"] * 8 + ["JJJJ", "oooo", "oooo", "OOOO"],
    },
}
# 右足だけに付けるワッペン：(面, 上からの行, 左からの列, ドット絵)
PATCHES = {"rollup": ("front", 2, 0, [".y.", "yYy", ".y."])}
PANTS_STYLE = "ender"


def legs(style: str) -> tuple[dict, dict]:
    """右足と左足。ワッペンは右足だけ。"""
    right = dict(PANTS[style])
    left = mirror(right)
    if style in PATCHES:
        face, y0, x0, patch = PATCHES[style]
        right[face] = paste(right[face], patch, x0, y0)
    return right, left


def design(outfit: str, pants: str) -> dict:
    o = OUTFITS[outfit]
    leg_r, leg_l = legs(pants)
    return {"head": o["head"], "hat": o["hat"], "body": o["body"], "arm": o["arm"], "arm_l": mirror(o["arm"]),
            "leg": leg_r, "leg_l": leg_l}


DESIGN = design(OUTFIT, PANTS_STYLE)


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
