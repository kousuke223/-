"""マイクラ風のブロックを描くための、小さな 3D レンダラー（numpy だけ）。

面（テクスチャ付きの四角形）ごとに、画面のピクセルから視線を飛ばして面との交点を求める方式。
Z バッファで前後関係を正しく処理し、テクスチャは最近傍で拾うのでドットがくっきり出る。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------- 回転


def rot_x(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rot_z(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


# ---------------------------------------------------------------- カメラ


class Camera:
    def __init__(self, pos, target, fov_y: float, w: int, h: int, up=(0, 1, 0)):
        self.pos = np.asarray(pos, float)
        f = np.asarray(target, float) - self.pos
        f /= np.linalg.norm(f)
        r = np.cross(f, up)
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        self.R = np.stack([r, u, f])  # ワールド → カメラ（x右, y上, z奥）
        self.w, self.h = w, h
        self.fpx = (h / 2) / math.tan(math.radians(fov_y) / 2)
        self.cx, self.cy = w / 2, h / 2

    def to_cam(self, p):
        return (np.asarray(p, float) - self.pos) @ self.R.T

    def project(self, p):
        c = self.to_cam(p)
        return self.cx + self.fpx * c[..., 0] / c[..., 2], self.cy - self.fpx * c[..., 1] / c[..., 2], c[..., 2]


# ---------------------------------------------------------------- 面


@dataclass
class Face:
    P0: np.ndarray          # テクスチャ左上の角（ワールド座標）
    E1: np.ndarray          # テクスチャの右方向の辺
    E2: np.ndarray          # テクスチャの下方向の辺
    tex: np.ndarray         # (h, w, 4) uint8
    emis: float = 0.0       # 自分で光る強さ（溶岩・グロウストーン）
    emask: np.ndarray | None = None   # テクスチャのどこが光るか（0〜1）
    rep: tuple = (1, 1)     # テクスチャを何回くり返すか
    double: bool = False    # 裏面も描く
    tint: tuple = (1.0, 1.0, 1.0)
    lit: bool = True
    lights: object = None   # この面だけ別のライトを使う（キャラ用など）
    n: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.n is None:
            n = np.cross(self.E2, self.E1)  # E1=右, E2=下 なので 下×右 が手前向き
            self.n = n / np.linalg.norm(n)


# 箱の6面（ローカル座標 0..size）。テクスチャの左上・右方向・下方向
def box_faces(size, tex: dict, M=np.eye(3), t=np.zeros(3), origin=np.zeros(3), **kw):
    """tex: {'front','back','left','right','top','bottom'} → テクスチャ。
    front = +z 面（キャラの顔の向き）。M, t: ローカル→ワールドの回転と移動。origin: 箱の角のローカル位置。"""
    sx, sy, sz = size
    o = np.asarray(origin, float)
    defs = {
        "front": ((0, sy, sz), (sx, 0, 0), (0, -sy, 0)),
        "back": ((sx, sy, 0), (-sx, 0, 0), (0, -sy, 0)),
        "left": ((sx, sy, sz), (0, 0, -sz), (0, -sy, 0)),     # +x 側（キャラから見て左）
        "right": ((0, sy, 0), (0, 0, sz), (0, -sy, 0)),       # -x 側（キャラから見て右）
        "top": ((0, sy, 0), (sx, 0, 0), (0, 0, sz)),
        "bottom": ((0, 0, sz), (sx, 0, 0), (0, 0, -sz)),
    }
    faces = []
    for name, (p0, e1, e2) in defs.items():
        tx = tex.get(name)
        if tx is None:
            continue
        P0 = t + M @ (o + np.asarray(p0, float))
        faces.append(Face(P0, M @ np.asarray(e1, float), M @ np.asarray(e2, float), tx, **kw))
    return faces


# ---------------------------------------------------------------- ライト


@dataclass
class Lights:
    ambient: tuple = (0.25, 0.1, 0.08)
    dirs: list = field(default_factory=list)     # [(向き(光が来る方向), 色)]
    points: list = field(default_factory=list)   # [(位置, 色, 半径)]
    lava_y: float = -100.0
    lava_color: tuple = (0.0, 0.0, 0.0)
    lava_h: float = 3.0
    blobs: list = field(default_factory=list)    # 足元の丸い影 [(x, z, 半径, 濃さ)]


def shade(P, n, L: Lights):
    out = np.broadcast_to(np.asarray(L.ambient, float), (len(P), 3)).copy()
    for d, c in L.dirs:
        d = np.asarray(d, float)
        d = d / np.linalg.norm(d)
        out += max(0.0, float(n @ d)) * np.asarray(c, float)
    for p, c, rad in L.points:
        v = np.asarray(p, float) - P
        dist = np.linalg.norm(v, axis=1) + 1e-6
        ndl = np.clip((v @ n) / dist, 0, None)
        out += (ndl / (1 + (dist / rad) ** 2))[:, None] * np.asarray(c, float)
    if L.lava_color[0] > 0:
        hgt = np.clip(P[:, 1] - L.lava_y, 0, None)
        k = np.exp(-hgt / L.lava_h) * (0.3 + 0.7 * max(0.0, -float(n[1])) + 0.25 * (abs(float(n[1])) < 0.5))
        out += k[:, None] * np.asarray(L.lava_color, float)
    if L.blobs and n[1] > 0.9:
        for bx, bz, br, bk in L.blobs:
            d2 = (P[:, 0] - bx) ** 2 + (P[:, 2] - bz) ** 2
            out *= (1 - bk * np.exp(-d2 / (br * br)))[:, None]
    return out


# ---------------------------------------------------------------- 描画


def render(faces: list[Face], cam: Camera, L: Lights, near: float = 0.05):
    """色(HDR, float32)・深さ・ワールドの高さ を返す。"""
    W, H = cam.w, cam.h
    color = np.zeros((H, W, 3), np.float32)
    depth = np.full((H, W), np.inf, np.float32)
    wy = np.zeros((H, W), np.float32)
    xs = ((np.arange(W) + 0.5 - cam.cx) / cam.fpx).astype(np.float32)
    ys = (-(np.arange(H) + 0.5 - cam.cy) / cam.fpx).astype(np.float32)

    prepared = []
    for f in faces:
        P0c = cam.R @ (f.P0 - cam.pos)
        E1c, E2c, nc = cam.R @ f.E1, cam.R @ f.E2, cam.R @ f.n
        facing = float(nc @ -P0c)
        if facing <= 0 and not f.double:
            continue
        corners = np.array([P0c, P0c + E1c, P0c + E2c, P0c + E1c + E2c])
        zc = corners[:, 2]
        if zc.max() < near:
            continue
        if zc.min() > near:
            sx = cam.cx + cam.fpx * corners[:, 0] / zc
            sy = cam.cy - cam.fpx * corners[:, 1] / zc
            x0, x1 = int(max(0, math.floor(sx.min()))), int(min(W, math.ceil(sx.max()) + 1))
            y0, y1 = int(max(0, math.floor(sy.min()))), int(min(H, math.ceil(sy.max()) + 1))
        else:
            x0, x1, y0, y1 = 0, W, 0, H
        if x0 >= x1 or y0 >= y1:
            continue
        prepared.append((float(zc.min()), f, P0c, E1c, E2c, nc, (x0, x1, y0, y1)))
    prepared.sort(key=lambda p: p[0])  # 手前から描くと無駄な計算が減る

    for _, f, P0c, E1c, E2c, nc, (x0, x1, y0, y1) in prepared:
        if f.double and float(nc @ -P0c) < 0:
            nc = -nc
            n_world = -f.n
        else:
            n_world = f.n
        dx = xs[x0:x1][None, :]
        dy = ys[y0:y1][:, None]
        denom = nc[0] * dx + nc[1] * dy + nc[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (nc @ P0c) / denom
        valid = np.isfinite(t) & (t > near)
        rx, ry, rz = t * dx - P0c[0], t * dy - P0c[1], t - P0c[2]
        u = (rx * E1c[0] + ry * E1c[1] + rz * E1c[2]) / float(E1c @ E1c)
        v = (rx * E2c[0] + ry * E2c[1] + rz * E2c[2]) / float(E2c @ E2c)
        ins = valid & (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
        dsub = depth[y0:y1, x0:x1]
        ins &= t < dsub
        if not ins.any():
            continue
        iy, ix = np.nonzero(ins)
        uu, vv = u[iy, ix], v[iy, ix]
        th, tw = f.tex.shape[:2]
        tu = np.clip(((uu * f.rep[0]) % 1.0 * tw).astype(int), 0, tw - 1)
        tv = np.clip(((vv * f.rep[1]) % 1.0 * th).astype(int), 0, th - 1)
        texel = f.tex[tv, tu]
        keep = texel[:, 3] >= 128
        if not keep.all():
            iy, ix, uu, vv, tu, tv, texel = iy[keep], ix[keep], uu[keep], vv[keep], tu[keep], tv[keep], texel[keep]
            if len(iy) == 0:
                continue
        rgb = texel[:, :3].astype(np.float32) / 255.0 * np.asarray(f.tint, np.float32)
        P = f.P0[None, :] + uu[:, None] * f.E1[None, :] + vv[:, None] * f.E2[None, :]
        if f.lit:
            col = rgb * shade(P, n_world, f.lights or L).astype(np.float32)
        else:
            col = rgb.copy()
        if f.emis > 0:
            em = f.emis if f.emask is None else f.emis * f.emask[tv, tu]
            col = col + rgb * em[:, None] if np.ndim(em) else col + rgb * em
        csub = color[y0:y1, x0:x1]
        csub[iy, ix] = col
        dsub[iy, ix] = t[iy, ix]
        wy[y0:y1, x0:x1][iy, ix] = P[:, 1]
    return color, depth, wy
