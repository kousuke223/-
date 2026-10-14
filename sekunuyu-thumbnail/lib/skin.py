"""マイクラのスキン画像（64×64）の決まり（どのパーツが画像のどこにあるか）と、3D モデル用の切り分け。"""
from __future__ import annotations

import numpy as np
from PIL import Image

# 各パーツのスキン画像上の位置（左上）と大きさ（幅, 高さ, 奥行き）。Minecraft の決まりどおり
PARTS = {
    "head": ((0, 0), (8, 8, 8)),
    "hat": ((32, 0), (8, 8, 8)),         # 頭の2枚目（帽子レイヤー）
    "body": ((16, 16), (8, 12, 4)),
    "jacket": ((16, 32), (8, 12, 4)),    # 体の2枚目（ジャケット）
    "arm": ((40, 16), (4, 12, 4)),       # 右腕
    "sleeve": ((40, 32), (4, 12, 4)),    # 右腕の2枚目（そで）
    "arm_l": ((32, 48), (4, 12, 4)),
    "sleeve_l": ((48, 48), (4, 12, 4)),
    "leg": ((0, 16), (4, 12, 4)),        # 右足
    "pants": ((0, 32), (4, 12, 4)),      # 右足の2枚目（ズボン）
    "leg_l": ((16, 48), (4, 12, 4)),
    "pants_l": ((0, 48), (4, 12, 4)),
}
# 2枚目のレイヤーは本体より少し大きく、浮いて見える（透明にしてよい所）
OVERLAYS = {"hat", "jacket", "sleeve", "sleeve_l", "pants", "pants_l"}


def face_rects(u, v, w, h, d):
    """箱の6面がスキン画像のどこに入るか。right/left はキャラから見た右/左。"""
    return {"top": (u + d, v, w, d), "bottom": (u + d + w, v, w, d),
            "right": (u, v + d, d, h), "front": (u + d, v + d, w, h),
            "left": (u + d + w, v + d, d, h), "back": (u + 2 * d + w, v + d, w, h)}


def load_skin(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"))


def skin_textures(skin: np.ndarray) -> dict:
    """スキン画像を、models.player が使う形（パーツ → 面 → テクスチャ）に切り分ける。"""
    T = {}
    for name, ((u, v), (w, h, d)) in PARTS.items():
        T[name] = {f: skin[y:y + fh, x:x + fw].copy() for f, (x, y, fw, fh) in face_rects(u, v, w, h, d).items()}
    return T
