"""ファミコン風（チップチューン）の音源と効果音を numpy だけで合成する。"""
from __future__ import annotations

import wave

import numpy as np
from scipy.signal import butter, lfilter

SR = 48000

_NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def freq(name: str) -> float:
    """'C5', 'F#4', 'Bb3' → 周波数(Hz)。A4 = 440Hz。"""
    n = _NOTE[name[0]]
    i = 1
    if name[i] == "#":
        n += 1
        i += 1
    elif name[i] == "b":
        n -= 1
        i += 1
    octave = int(name[i:])
    midi = 12 * (octave + 1) + n
    return 440.0 * 2 ** ((midi - 69) / 12)


# ---------------------------------------------------------------- 発振器


def _polyblep(t, dt):
    y = np.zeros_like(t)
    m = t < dt
    x = t[m] / dt[m]
    y[m] = x + x - x * x - 1
    m = t > 1 - dt
    x = (t[m] - 1) / dt[m]
    y[m] = x * x + x + x + 1
    return y


def _phase(f_arr):
    dt = f_arr / SR
    ph = np.concatenate([[0.0], np.cumsum(dt)[:-1]]) % 1.0
    return ph, dt


def pulse(f_arr: np.ndarray, duty: float = 0.5) -> np.ndarray:
    """帯域制限（PolyBLEP）した矩形波。f_arr はサンプルごとの周波数。"""
    ph, dt = _phase(f_arr)
    y = np.where(ph < duty, 1.0, -1.0)
    y += _polyblep(ph, dt) - _polyblep((ph - duty) % 1.0, dt)
    return y - (2 * duty - 1)


def triangle(f_arr: np.ndarray, steps: int = 16) -> np.ndarray:
    """ファミコンの三角波っぽく 16 段階に量子化。"""
    ph, _ = _phase(f_arr)
    y = 4 * np.abs(ph - 0.5) - 1
    if steps:
        y = np.round((y + 1) * (steps - 1) / 2) / ((steps - 1) / 2) - 1
    return y - y.mean()


def noise(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1, 1, n)


def lowpass(x, cutoff, order=2):
    b, a = butter(order, min(cutoff, SR / 2 - 100) / (SR / 2), "low")
    return lfilter(b, a, x)


def highpass(x, cutoff, order=2):
    b, a = butter(order, cutoff / (SR / 2), "high")
    return lfilter(b, a, x)


def bandpass(x, lo, hi, order=2):
    b, a = butter(order, [lo / (SR / 2), hi / (SR / 2)], "band")
    return lfilter(b, a, x)


def adsr(dur: float, a=0.004, d=0.08, s=0.7, r=0.05) -> np.ndarray:
    """dur 秒押して離したときのエンベロープ（リリース分だけ長くなる）。"""
    n_on = max(1, int(dur * SR))
    na, nd, nr = max(1, int(a * SR)), max(1, int(d * SR)), max(1, int(r * SR))
    e = np.empty(n_on + nr)
    t = np.arange(n_on)
    on = np.where(t < na, t / na, s + (1 - s) * np.exp(-(t - na) / (nd / 3)))
    e[:n_on] = on
    last = on[-1]
    e[n_on:] = last * np.exp(-np.arange(nr) / (nr / 5))
    return e


def glide(f0: float, f1: float, n: int, curve: float = 1.0) -> np.ndarray:
    x = np.linspace(0, 1, n) ** curve
    return f0 * (f1 / f0) ** x


# ---------------------------------------------------------------- 楽器


def lead(f: float, dur: float, duty=0.25, vib=True, gain=1.0) -> np.ndarray:
    env = adsr(dur, a=0.004, d=0.12, s=0.62, r=0.07)
    n = len(env)
    t = np.arange(n) / SR
    f_arr = np.full(n, f)
    if vib and dur > 0.28:
        ramp = np.clip((t - 0.14) / 0.2, 0, 1)
        f_arr = f * (1 + 0.009 * np.sin(2 * np.pi * 5.8 * t) * ramp)
    return pulse(f_arr, duty) * env * gain


def stab(fs: list[float], dur=0.09, duty=0.125, gain=1.0) -> np.ndarray:
    env = adsr(dur, a=0.002, d=0.05, s=0.35, r=0.03)
    n = len(env)
    out = np.zeros(n)
    for f in fs:
        out += pulse(np.full(n, f), duty)
    return out / max(1, len(fs)) * env * gain


def bass(f: float, dur: float, gain=1.0) -> np.ndarray:
    env = adsr(dur, a=0.003, d=0.1, s=0.85, r=0.03)
    return triangle(np.full(len(env), f)) * env * gain


def kick(gain=1.0) -> np.ndarray:
    n = int(0.22 * SR)
    t = np.arange(n) / SR
    f = glide(170, 42, n, curve=0.35)
    ph = np.cumsum(f) / SR
    body = np.sin(2 * np.pi * ph) * np.exp(-t / 0.09)
    click = noise(n, 1) * np.exp(-t / 0.004) * 0.35
    return (body + click) * gain


def snare(gain=1.0, seed=2) -> np.ndarray:
    n = int(0.2 * SR)
    t = np.arange(n) / SR
    nz = bandpass(noise(n, seed), 1200, 9000) * np.exp(-t / 0.06)
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.035) * 0.5
    return (nz * 1.6 + tone) * gain


def hat(open_=False, gain=1.0, seed=3) -> np.ndarray:
    dur = 0.18 if open_ else 0.045
    n = int(dur * SR)
    t = np.arange(n) / SR
    return highpass(noise(n, seed), 7000) * np.exp(-t / (dur / 4)) * gain


def crash(gain=1.0, dur=1.6, seed=4) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    nz = highpass(noise(n, seed), 4000) * np.exp(-t / (dur / 4.5))
    return nz * gain


# ---------------------------------------------------------------- 効果音


def whoosh(dur=0.6, rise=True, gain=1.0, seed=5) -> np.ndarray:
    """雲がシュワッと動く音。低い帯域→高い帯域へ（rise=False で逆）。"""
    n = int(dur * SR)
    x = np.linspace(0, 1, n)
    nz = noise(n, seed)
    bands = [lowpass(nz, 500), bandpass(nz, 500, 2000), bandpass(nz, 2000, 6000), highpass(nz, 6000)]
    pos = x if rise else 1 - x
    out = np.zeros(n)
    for i, b in enumerate(bands):
        w = np.clip(1 - np.abs(pos * 3 - i), 0, 1)
        out += b * w * (1.0 if i < 3 else 0.6)
    env = np.sin(np.pi * np.clip(x, 0, 1)) ** 1.5
    return out * env * 1.6 * gain


def boing(gain=1.0) -> np.ndarray:
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    f = np.concatenate([glide(260, 820, int(0.07 * SR), 0.7), np.full(n - int(0.07 * SR), 820.0)])
    f = f * (1 + 0.06 * np.sin(2 * np.pi * 17 * t) * np.exp(-t / 0.1))
    return pulse(f, 0.25) * np.exp(-t / 0.12) * gain


def pop(gain=1.0, f0=900, f1=260) -> np.ndarray:
    n = int(0.09 * SR)
    t = np.arange(n) / SR
    f = glide(f0, f1, n, 0.5)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.03) * gain


def blip(f: float, dur=0.04, duty=0.5, gain=1.0) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    return pulse(np.full(n, f), duty) * np.exp(-t / (dur / 2.5)) * gain


def sparkle(gain=1.0, notes=("C7", "E7", "G7", "C8", "E8"), step=0.045) -> np.ndarray:
    n = int((step * len(notes) + 0.25) * SR)
    out = np.zeros(n)
    for i, nm in enumerate(notes):
        b = blip(freq(nm), dur=0.12, duty=0.125, gain=0.8)
        s = int(i * step * SR)
        out[s:s + len(b)] += b[: n - s]
    # 小さなやまびこ
    d = int(0.09 * SR)
    out[d:] += out[:-d] * 0.35
    return out * gain


def fall(gain=1.0) -> np.ndarray:
    n = int(0.2 * SR)
    t = np.arange(n) / SR
    f = glide(1800, 420, n, 1.4)
    return pulse(f, 0.5) * (0.4 + 0.6 * t / t[-1]) * gain


def click(gain=1.0) -> np.ndarray:
    n = int(0.03 * SR)
    t = np.arange(n) / SR
    return (pulse(np.full(n, 2100.0), 0.5) * 0.5 + noise(n, 7) * 0.6) * np.exp(-t / 0.006) * gain


def jump(gain=1.0) -> np.ndarray:
    n = int(0.2 * SR)
    t = np.arange(n) / SR
    f = glide(380, 1300, n, 0.6)
    return pulse(f, 0.25) * np.exp(-t / 0.12) * gain


def pikon(gain=1.0) -> np.ndarray:
    a = blip(freq("A6"), dur=0.06, duty=0.25)
    b = blip(freq("A7"), dur=0.22, duty=0.25)
    out = np.zeros(len(a) + len(b))
    out[:len(a)] += a
    out[int(0.055 * SR):int(0.055 * SR) + len(b)] += b
    return out * gain


# ---------------------------------------------------------------- ミキサー


class Mixer:
    def __init__(self, dur: float):
        self.n = int(dur * SR)
        self.buf = np.zeros((self.n, 2))

    def add(self, sig: np.ndarray, t: float, gain: float = 1.0, pan: float = 0.0) -> None:
        s = int(round(t * SR))
        if s >= self.n or s + len(sig) <= 0:
            return
        if s < 0:
            sig = sig[-s:]
            s = 0
        e = min(self.n, s + len(sig))
        sig = sig[: e - s] * gain
        ang = (pan + 1) * np.pi / 4
        self.buf[s:e, 0] += sig * np.cos(ang)
        self.buf[s:e, 1] += sig * np.sin(ang)

    def echo(self, delay: float, fb: float, mix: float) -> None:
        d = int(delay * SR)
        wet = np.zeros_like(self.buf)
        src = self.buf.copy()
        g = mix
        for k in range(1, 5):
            wet[d * k:] += src[: self.n - d * k][:, ::-1 if k % 2 else 1] * g
            g *= fb
        self.buf += wet

    def mixdown(self, peak_db: float = -1.5) -> np.ndarray:
        x = self.buf
        x = np.tanh(x * 1.1) / 1.1
        pk = np.max(np.abs(x))
        if pk > 0:
            x = x / pk * 10 ** (peak_db / 20)
        return x


def write_wav(path: str, x: np.ndarray) -> None:
    pcm = np.clip(x * 32767, -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
