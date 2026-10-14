"""マイクラの地形（ボクセル）を作って、見えている面だけを Face にする。"""
from __future__ import annotations

import numpy as np

from .voxel import Face

AIR, RACK, LAVA, GLOW, BRICK, MAGMA, QUARTZ, FALL = range(8)              # ネザー（#2）
GRASS, DIRT, STONE, LOG, LEAVES, PLANKS, OBSIDIAN, WATER = range(8, 16)  # オーバーワールド（#3）
DEEPSLATE, DTILES, SCULK = range(16, 19)                                  # ディープダーク（#1）
LIQUIDS = (LAVA, WATER)
EMISSIVE = {LAVA: 0.82, GLOW: 1.05, FALL: 1.15}   # ブロック全体が光る
SPECKLED = {MAGMA: 1.3, SCULK: 1.1}               # テクスチャの光る所（emask）だけ光る

# 6方向: (法線, 隣のずれ)
DIRS = {
    "px": (1, 0, 0), "nx": (-1, 0, 0), "py": (0, 1, 0), "ny": (0, -1, 0), "pz": (0, 0, 1), "nz": (0, 0, -1),
}


class VoxelWorld:
    def __init__(self, x0, y0, z0, nx, ny, nz):
        self.o = np.array([x0, y0, z0], int)
        self.v = np.zeros((nx, ny, nz), np.uint8)

    def box(self, xa, xb, ya, yb, za, zb, val):
        """[xa, xb) × [ya, yb) × [za, zb) をうめる（ワールド座標）。"""
        o = self.o
        s = self.v.shape
        i0, i1 = max(0, xa - o[0]), min(s[0], xb - o[0])
        j0, j1 = max(0, ya - o[1]), min(s[1], yb - o[1])
        k0, k1 = max(0, za - o[2]), min(s[2], zb - o[2])
        if i0 < i1 and j0 < j1 and k0 < k1:
            self.v[i0:i1, j0:j1, k0:k1] = val

    def get(self, x, y, z):
        i, j, k = x - self.o[0], y - self.o[1], z - self.o[2]
        if 0 <= i < self.v.shape[0] and 0 <= j < self.v.shape[1] and 0 <= k < self.v.shape[2]:
            return int(self.v[i, j, k])
        return AIR

    def faces(self, tex: dict, rng_seed=0, near_z: float = -6.0) -> list[Face]:
        """見えている面。z が near_z より奥の面は横につなげて数を減らす（手前は1ブロックずつ）。"""
        v = self.v
        out = []
        rng = np.random.default_rng(rng_seed)
        pad = np.pad(v, 1)
        for name, (dx, dy, dz) in DIRS.items():
            nb = pad[1 + dx:1 + dx + v.shape[0], 1 + dy:1 + dy + v.shape[1], 1 + dz:1 + dz + v.shape[2]]
            expose = (v != AIR) & (nb == AIR)
            # 溶岩や水は少し低いので、となりのブロックの側面も見せる
            if dy == 0:
                expose |= (v != AIR) & ~np.isin(v, LIQUIDS) & np.isin(nb, LIQUIDS)
            idx = np.argwhere(expose)
            if len(idx) == 0:
                continue
            far = (idx[:, 2] + self.o[2]) < near_z
            for (i, j, k), n_run, axis in _runs(idx[far], v, name) + [(tuple(c), 1, 0) for c in idx[~far]]:
                x, y, z = (int(a) for a in self.o + np.array([i, j, k]))
                out.append(_block_face(name, x, y, z, int(v[i, j, k]), tex, rng, n_run, axis))
        return out


def _runs(idx, v, name):
    """同じ行に並ぶ同種ブロックの面をまとめる。戻り値: [(開始セル, 長さ, 軸)]"""
    if len(idx) == 0:
        return []
    axis = 2 if name in ("px", "nx") else 0
    other = [a for a in range(3) if a != axis]
    idx = idx[np.lexsort((idx[:, axis], idx[:, other[1]], idx[:, other[0]]))]
    res = []
    start, prev, n = idx[0], idx[0], 1
    for cur in idx[1:]:
        same_line = cur[other[0]] == prev[other[0]] and cur[other[1]] == prev[other[1]]
        if same_line and cur[axis] == prev[axis] + 1 and v[tuple(cur)] == v[tuple(start)] and n < 24:
            n += 1
        else:
            res.append((tuple(start), n, axis))
            start, n = cur, 1
        prev = cur
    res.append((tuple(start), n, axis))
    return res


def _block_face(name, x, y, z, b, tex, rng, n_run, run_axis):
    t = tex[b]
    if isinstance(t, dict):  # 上・下・側面でちがう模様（草ブロックや原木）
        t = t["top"] if name == "py" else t["bottom"] if name == "ny" else t["side"]
    if isinstance(t, list):
        t = t[rng.integers(len(t))]
    emask = None
    if isinstance(t, tuple):
        t, emask = t
    top = 0.875 if b in LIQUIDS else 1.0
    L = float(n_run)
    ex = np.array([L, 0, 0]) if run_axis == 0 else np.array([1.0, 0, 0])
    ez = np.array([0, 0, L]) if run_axis == 2 else np.array([0, 0, 1.0])
    rep_x = (L, 1) if run_axis == 0 else (1, 1)
    rep_z = (L, 1) if run_axis == 2 else (1, 1)
    hx = L if run_axis == 0 else 1.0
    hz = L if run_axis == 2 else 1.0
    if name == "py":
        P0, E1, E2, rep = (x, y + top, z), ex, ez, (rep_x[0], rep_z[0])
    elif name == "ny":
        P0, E1, E2, rep = (x, y, z + hz), ex, -ez, (rep_x[0], rep_z[0])
    elif name == "pz":
        P0, E1, E2, rep = (x, y + top, z + 1), ex, np.array([0, -top, 0]), (rep_x[0], 1)
    elif name == "nz":
        P0, E1, E2, rep = (x + hx, y + top, z), -ex, np.array([0, -top, 0]), (rep_x[0], 1)
    elif name == "px":
        P0, E1, E2, rep = (x + 1, y + top, z + hz), -ez, np.array([0, -top, 0]), (rep_z[0], 1)
    else:  # nx
        P0, E1, E2, rep = (x, y + top, z), ez, np.array([0, -top, 0]), (rep_z[0], 1)
    varied = (RACK, QUARTZ, BRICK, GRASS, DIRT, STONE, LEAVES, PLANKS, DEEPSLATE, DTILES)
    tint = tuple(np.clip(1 + rng.normal(0, 0.05), 0.85, 1.12) * np.ones(3)) if b in varied else (1, 1, 1)
    return Face(np.array(P0, float), np.asarray(E1, float), np.asarray(E2, float), t,
                emis=EMISSIVE.get(b, SPECKLED.get(b, 0.0)), emask=emask, rep=rep, tint=tint,
                lit=b not in EMISSIVE)
