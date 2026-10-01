"""せくぬゆ（プレイヤー）とガストの 3D モデル。単位は1ブロック、PX = 1/16 ブロック。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .voxel import Face, box_faces, rot_x, rot_y, rot_z

PX = 1 / 16


@dataclass
class Pose:
    yaw: float = 0.0          # 体の向き（0 で +z = カメラ側を向く）
    lean: float = 0.0         # 前のめり（+ で前に倒れる）
    twist: float = 0.0        # 上半身のひねり
    head_yaw: float = 0.0
    head_pitch: float = 0.0   # + で上を向く
    head_roll: float = 0.0
    arm_r: tuple = (0.0, 0.0)  # (前後, 外に開く)  前に上げる = マイナス、真上 = -180
    arm_l: tuple = (0.0, 0.0)
    leg_r: float = 0.0         # 前に出す = マイナス
    leg_l: float = 0.0


def player(T, HS, sword_tex, pos, pose: Pose, scale: float = 1.0, sword_hand: str = "right",
           lights=None) -> tuple[list[Face], dict]:
    """プレイヤーの面のリストと、手や頭の位置（ワールド座標）を返す。"""
    pos = np.asarray(pos, float)
    Rr = rot_y(pose.yaw) @ rot_x(pose.lean)
    k = PX * scale
    faces: list[Face] = []
    info = {}

    def part(size, pivot, origin, Rl, tex, base=None, **kw):
        R0, t0 = base if base else (Rr, pos)
        M = R0 @ Rl
        t = t0 + R0 @ (np.asarray(pivot, float) * k)
        faces.extend(box_faces(np.asarray(size, float) * k, tex, M=M, t=t, origin=np.asarray(origin, float) * k, **kw))
        return M, t

    # 足
    part((4, 12, 4), (-2, 12, 0), (-2, -12, -2), rot_x(pose.leg_r), T["leg"])
    part((4, 12, 4), (2, 12, 0), (-2, -12, -2), rot_x(pose.leg_l), T["leg"])
    # 胴体（腰を中心にひねる）
    Rt = rot_y(pose.twist)
    Mb, tb = part((8, 12, 4), (0, 12, 0), (-4, 0, -2), Rt, T["body"])
    upper = (Mb, tb)
    # 腕（肩から）
    # 肩：まず外に開いてから前後に振る（右腕は -x 側なので開く向きが逆）
    Ra_r = rot_x(pose.arm_r[0]) @ rot_z(-pose.arm_r[1])
    Ra_l = rot_x(pose.arm_l[0]) @ rot_z(pose.arm_l[1])
    Mr, tr = part((4, 12, 4), (-6, 10, 0), (-2, -10, -2), Ra_r, T["arm"], base=upper)
    Ml, tl = part((4, 12, 4), (6, 10, 0), (-2, -10, -2), Ra_l, T["arm"], base=upper)
    # 頭（首から）
    Rh = rot_y(pose.head_yaw) @ rot_x(-pose.head_pitch) @ rot_z(pose.head_roll)
    Mh, th = part((8, 8, 8), (0, 12, 0), (-4, 0, -4), Rh, T["head"], base=upper)
    head = (Mh, th)
    part((9, 9, 9), (0, 0, 0), (-4.5, -0.5, -4.5), np.eye(3), T["hat"], base=head)
    # ヘッドセット
    part((2, 4.5, 4), (0, 0, 0), (-6.6, 1.0, -2), np.eye(3), HS["cup"], base=head)
    part((2, 4.5, 4), (0, 0, 0), (4.6, 1.0, -2), np.eye(3), HS["cup"], base=head)
    part((11.5, 1, 2), (0, 0, 0), (-5.75, 8.5, -1), np.eye(3), HS["band"], base=head)
    part((1, 4.0, 1.5), (0, 0, 0), (-5.75, 5.0, -0.75), np.eye(3), HS["band"], base=head)
    part((1, 4.0, 1.5), (0, 0, 0), (4.75, 5.0, -0.75), np.eye(3), HS["band"], base=head)
    part((0.8, 0.8, 3.4), (-5.6, 1.8, 1.8), (0, 0, 0), rot_y(22), HS["band"], base=head)
    part((1.6, 1.6, 1.6), (-5.2, 1.4, 4.6), (0, 0, 0), np.eye(3), HS["cup"], base=head)
    info["head_center"] = th + Mh @ (np.array([0, 4, 0]) * k)
    info["head_top"] = th + Mh @ (np.array([0, 9.5, 0]) * k)
    # 剣
    if sword_tex is not None:
        Ma, ta = (Mr, tr) if sword_hand == "right" else (Ml, tl)
        L = 0.8 * scale
        hand = ta + Ma @ (np.array([0, -10.5, 1.0]) * k)
        e1 = Ma @ (np.array([0, -1, 1]) / np.sqrt(2) * L)
        e2 = Ma @ (np.array([0, -1, -1]) / np.sqrt(2) * L)
        p0 = hand - e2 - e1 * 0.06
        faces.append(Face(p0, e1, e2, sword_tex, double=True))
    info["back"] = tb + Mb @ (np.array([0, 8, -3]) * k)
    if lights is not None:
        for f in faces:
            f.lights = lights
    return faces, info


def ghast(body_tex, emask, tent_tex, pos, yaw=0.0, pitch=0.0, size=4.0, tent_len=None, seed=1, lights=None):
    """ガスト。pos は胴体の中心。正面（顔）は +z 側。"""
    pos = np.asarray(pos, float)
    R = rot_y(yaw) @ rot_x(pitch)
    s = size
    faces = box_faces((s, s, s), body_tex, M=R, t=pos, origin=np.array([-s / 2, -s / 2, -s / 2]))
    # 顔の目だけ赤く光らせる
    for f in faces:
        if f.tex is body_tex["front"]:
            f.emis = 0.9
            f.emask = emask
    rng = np.random.default_rng(seed)
    tw = s * 0.11
    for i in range(3):
        for j in range(3):
            x = (-0.32 + 0.32 * i) * s + rng.uniform(-0.05, 0.05) * s
            z = (-0.32 + 0.32 * j) * s + rng.uniform(-0.05, 0.05) * s
            ln = (tent_len or rng.uniform(0.45, 0.85)) * s
            Rt = R @ rot_x(rng.uniform(-14, 10)) @ rot_z(rng.uniform(-10, 10))
            top = pos + R @ np.array([x, -s / 2, z])
            faces.extend(box_faces((tw, ln, tw), tent_tex, M=Rt, t=top, origin=np.array([-tw / 2, -ln, -tw / 2])))
    mouth = pos + R @ np.array([0, -s / 2 + s * (16 - 11.5) / 16, s / 2])
    if lights is not None:
        for f in faces:
            f.lights = lights
    return faces, mouth
