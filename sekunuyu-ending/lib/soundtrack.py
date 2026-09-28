"""BGM と効果音の配置（オリジナル曲）。"""
from __future__ import annotations

import numpy as np

from . import chiptune as ch
from . import timeline as T
from .chiptune import SR, freq

F = freq


def _notes(bar_str: str):
    """'A5 - G5 A5 C6 - A5 -' → [(開始8分, 長さ8分, 音名), ...]"""
    toks = bar_str.split()
    out = []
    for i, tk in enumerate(toks):
        if tk == "-":
            if out:
                s, d, n = out[-1]
                out[-1] = (s, d + 1, n)
        elif tk != ".":
            out.append((i, 1, tk))
    return out


MELODY = [
    "A5 - G5 A5 C6 - A5 -",
    "B5 - A5 B5 D6 - B5 -",
    "E6 - D6 B5 G5 - B5 -",
    "C6 - - - . A5 B5 C6",
    "D6 - C6 A5 F5 - A5 -",
    "D6 - E6 D6 B5 - G5 -",
    "E6 - D6 C6 G5 - C6 -",
    "D6 - - - E6 - D6 -",
    "C6 - A5 - B5 - D6 -",
]
# 小節ごとのコード（前半・後半）: (ベースの根音, 和音)
CHORDS = [
    (("F", ["F4", "A4", "C5"]),) * 2,
    (("G", ["G4", "B4", "D5"]),) * 2,
    (("E", ["E4", "G4", "B4"]),) * 2,
    (("A", ["E4", "A4", "C5"]),) * 2,
    (("F", ["F4", "A4", "C5"]),) * 2,
    (("G", ["G4", "B4", "D5"]),) * 2,
    (("C", ["E4", "G4", "C5"]),) * 2,
    (("C", ["E4", "G4", "C5"]), ("G", ["D4", "G4", "B4"])),
    (("F", ["F4", "A4", "C5"]), ("G", ["D4", "G4", "B4"])),
]


def logo_jingle(m: ch.Mixer, lead_bus: ch.Mixer) -> None:
    """シーンA：雲が開く→アバター→ロゴ ドンッ！→ファンファーレ。"""
    b = T.BEAT
    m.add(ch.whoosh(0.8, rise=True, seed=11), T.A_CURTAIN_OPEN[0], gain=0.32)
    m.add(ch.sparkle(notes=("C6", "E6", "G6", "C7", "E7", "G7"), step=0.06), 0.05, gain=0.13, pan=0.2)
    m.add(ch.boing(), T.A_AVATAR_POP, gain=0.2)
    m.add(ch.fall(), T.A_LOGO_DROP, gain=0.09)
    t0 = T.A_LOGO_HIT
    m.add(ch.kick(), t0, gain=0.75)
    m.add(ch.crash(dur=2.2), t0, gain=0.16)
    m.add(ch.stab([F("C4"), F("E4"), F("G4"), F("C5")], dur=0.34, duty=0.5), t0, gain=0.3)
    m.add(ch.pop(f0=700, f1=120), t0, gain=0.35)
    # ファンファーレ（拍, 長さ, 音）
    mel = [(1.0, 0.5, "G5"), (1.5, 0.5, "C6"), (2.0, 0.5, "E6"), (2.5, 0.25, "D6"), (2.75, 0.25, "C6"),
           (3.0, 0.5, "D6"), (3.5, 0.5, "G5"), (4.0, 1.75, "C6")]
    har = [(1.0, 0.5, "E5"), (1.5, 0.5, "G5"), (2.0, 0.5, "C6"), (2.5, 0.25, "B5"), (2.75, 0.25, "A5"),
           (3.0, 0.5, "B5"), (3.5, 0.5, "D5"), (4.0, 1.75, "E5")]
    for s, d, n in mel:
        lead_bus.add(ch.lead(F(n), d * b * 0.95), t0 + s * b, gain=0.2, pan=-0.1)
    for s, d, n in har:
        m.add(ch.lead(F(n), d * b * 0.95, duty=0.5, vib=False), t0 + s * b, gain=0.08, pan=0.25)
    m.add(ch.lead(F("G5"), 1.75 * b * 0.95, duty=0.5, vib=False), t0 + 4.0 * b, gain=0.07, pan=-0.25)
    bas = [(0, 1.0, "C2"), (1.0, 0.5, "C3"), (1.5, 0.5, "G2"), (2.0, 0.5, "C3"), (2.5, 0.5, "G2"),
           (3.0, 0.5, "B2"), (3.5, 0.5, "D3"), (4.0, 1.75, "C3")]
    for s, d, n in bas:
        m.add(ch.bass(F(n), d * b * 0.92), t0 + s * b, gain=0.27)
    for s, chord in ((1.5, ["E4", "G4", "C5"]), (2.5, ["D4", "G4", "B4"]), (3.5, ["D4", "G4", "B4"])):
        m.add(ch.stab([F(x) for x in chord]), t0 + s * b, gain=0.12, pan=0.3)
    for s in (2.0, 4.0):
        m.add(ch.kick(), t0 + s * b, gain=0.55)
    for s in (1.0, 3.0, 3.5, 3.75):
        m.add(ch.snare(seed=int(s * 4)), t0 + s * b, gain=0.2)
    for s in (1.0, 1.5, 2.0, 2.5, 3.0):
        m.add(ch.hat(), t0 + s * b, gain=0.06, pan=0.25)
    m.add(ch.crash(dur=1.8), t0 + 4.0 * b, gain=0.13)
    # リボン・キラーン・雲が閉じる
    m.add(ch.pop(), T.A_RIBBON, gain=0.22)
    m.add(ch.sparkle(), T.A_SHINE, gain=0.1, pan=0.3)
    m.add(ch.whoosh(0.55, rise=True, seed=12), T.A_CURTAIN_CLOSE[0], gain=0.3)


def endscreen_music(m: ch.Mixer, lead_bus: ch.Mixer) -> None:
    b = T.BEAT
    e = b / 2
    tb = T.B_MUSIC
    for bar, (mel, chords) in enumerate(zip(MELODY, CHORDS)):
        t_bar = tb + bar * T.BAR
        for s, d, n in _notes(mel):
            lead_bus.add(ch.lead(F(n), d * e * 0.92), t_bar + s * e, gain=0.2, pan=-0.1)
        for slot in range(8):
            root, triad = chords[0] if slot < 4 else chords[1]
            octv = 2 if slot % 2 == 0 else 3
            m.add(ch.bass(F(f"{root}{octv}"), e * 0.85), t_bar + slot * e, gain=0.24)
            if slot % 2 == 1:
                m.add(ch.stab([F(x) for x in triad]), t_bar + slot * e, gain=0.075, pan=0.3)
        if 4 <= bar <= 7:
            triad = chords[0][1]
            arp = [F(x) * 2 for x in triad] + [F(triad[1]) * 2]
            for k in range(16):
                m.add(ch.blip(arp[k % 4], dur=0.08, duty=0.125), t_bar + k * e / 2, gain=0.035, pan=-0.35)
        # ドラム
        kicks = [0, 2] + ([2.5] if bar in (3, 7) else [])
        for s in kicks:
            m.add(ch.kick(), t_bar + s * b, gain=0.5)
        fill = bar in (3, 7)
        snares = [1] + ([3, 3.25, 3.5, 3.75] if fill else [3])
        for s in snares:
            m.add(ch.snare(seed=bar * 10 + int(s * 4)), t_bar + s * b, gain=0.17 if s in (1, 3) else 0.12)
        for k in range(8):
            if fill and k >= 6:
                continue
            m.add(ch.hat(open_=(k == 7 and bar in (1, 5)), seed=k), t_bar + k * e, gain=0.05, pan=0.25)
        if bar in (0, 4, 8):
            m.add(ch.crash(dur=1.6), t_bar, gain=0.1)
    # 最後のジャーン
    tf = T.B_FINAL
    hold = T.ENDING_DUR - tf - 0.35
    lead_bus.add(ch.lead(F("C6"), hold), tf, gain=0.19, pan=-0.1)
    m.add(ch.lead(F("E5"), hold, duty=0.5, vib=False), tf, gain=0.07, pan=0.25)
    m.add(ch.lead(F("G5"), hold, duty=0.5, vib=False), tf, gain=0.06, pan=-0.25)
    m.add(ch.bass(F("C2"), hold), tf, gain=0.26)
    m.add(ch.stab([F("C4"), F("E4"), F("G4"), F("C5")], dur=0.4, duty=0.5), tf, gain=0.18)
    m.add(ch.kick(), tf, gain=0.6)
    m.add(ch.crash(dur=1.4), tf, gain=0.13)
    m.add(ch.sparkle(), tf + 0.55, gain=0.1, pan=0.3)


def endscreen_sfx(m: ch.Mixer, n_title: int, n_type: int, n_title2: int, n_type2: int) -> None:
    m.add(ch.whoosh(0.7, rise=False, seed=13), T.B_CURTAIN_OPEN[0] - 0.05, gain=0.3)
    scale = ["C6", "D6", "E6", "F6", "G6", "A6", "B6", "C7", "D7", "E7", "F7", "G7"]
    for i in range(n_title):
        m.add(ch.blip(F(scale[i % len(scale)]), dur=0.05, duty=0.25), T.B_TITLE + i * T.B_TITLE_STEP + 0.12,
              gain=0.07)
    m.add(ch.whoosh(0.35, rise=True, seed=14), T.B_FRAMES, gain=0.2, pan=-0.5)
    m.add(ch.whoosh(0.35, rise=True, seed=15), T.B_FRAMES + 0.08, gain=0.2, pan=0.5)
    m.add(ch.pop(), T.B_RING + 0.05, gain=0.25)
    m.add(ch.boing(), T.B_AVATAR, gain=0.16, pan=-0.4)
    m.add(ch.pop(f0=1200, f1=500), T.B_BUBBLE, gain=0.18, pan=-0.3)
    for i in range(n_type):
        m.add(ch.blip(1320 if i % 2 else 1480, dur=0.028), T.B_TYPE + i * T.B_TYPE_STEP, gain=0.045, pan=-0.3)
    m.add(ch.pikon(), T.B_THUMB, gain=0.1, pan=0.4)
    for tc in T.B_CLICKS:
        m.add(ch.click(), tc, gain=0.3, pan=0.2)
        m.add(ch.sparkle(notes=("G6", "C7", "E7", "G7")), tc + 0.03, gain=0.09, pan=0.2)
    m.add(ch.jump(), T.B_SWAP, gain=0.14)
    for i in range(n_title2):
        m.add(ch.blip(F(scale[(i + 2) % len(scale)]), dur=0.05, duty=0.25),
              T.B_SWAP + 0.1 + i * T.B_TITLE_STEP + 0.12, gain=0.07)
    for i in range(n_type2):
        m.add(ch.blip(1320 if i % 2 else 1480, dur=0.028), T.B_SWAP_TYPE + i * T.B_TYPE_STEP, gain=0.045,
              pan=-0.3)


def _finish(m: ch.Mixer, lead_bus: ch.Mixer, fade: float, peak_db: float) -> np.ndarray:
    lead_bus.echo(delay=3 * T.BEAT / 4, fb=0.35, mix=0.25)
    m.buf += lead_bus.buf
    n = int(fade * SR)
    m.buf[-n:] *= np.linspace(1, 0, n)[:, None] ** 2
    return m.mixdown(peak_db=peak_db)


def build_ending(n_title: int, n_type: int, n_title2: int, n_type2: int) -> np.ndarray:
    m = ch.Mixer(T.ENDING_DUR)
    lead_bus = ch.Mixer(T.ENDING_DUR)
    logo_jingle(m, lead_bus)
    endscreen_music(m, lead_bus)
    endscreen_sfx(m, n_title, n_type, n_title2, n_type2)
    return _finish(m, lead_bus, fade=0.35, peak_db=-4.0)


def build_opening() -> np.ndarray:
    m = ch.Mixer(T.OPENING_DUR)
    lead_bus = ch.Mixer(T.OPENING_DUR)
    logo_jingle(m, lead_bus)
    return _finish(m, lead_bus, fade=0.3, peak_db=-4.0)
