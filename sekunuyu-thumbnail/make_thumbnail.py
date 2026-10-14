#!/usr/bin/env python3
"""「セククラ ハードコア」シリーズのサムネイルを作る（1280x720）。

    python3 make_thumbnail.py               # 全エピソードを output/ に書き出し
    python3 make_thumbnail.py --episode 3   # #3 だけ
    python3 make_thumbnail.py --episode 3 --preview   # 等倍ですばやく確認
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lib.fonts import ensure_fonts  # noqa: E402

W, H = 1280, 720
EPISODES = {
    1: "episodes.ep01_deepdark",
    2: "episodes.ep02_nether",
    3: "episodes.ep03_portal",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, choices=sorted(EPISODES), action="append",
                    help="作るエピソード（省略すると全部）")
    ap.add_argument("--preview", action="store_true", help="等倍ですばやく確認")
    ap.add_argument("--out", default=os.path.join(HERE, "output"))
    args = ap.parse_args()
    ensure_fonts()
    os.makedirs(args.out, exist_ok=True)
    S = 1 if args.preview else 2
    for ep in args.episode or sorted(EPISODES):
        mod = importlib.import_module(EPISODES[ep])
        thumb = mod.compose(mod.render_art(ss=S), S).convert("RGB")
        if S != 1:
            thumb = thumb.resize((W, H), Image.LANCZOS)
        name = f"preview_{ep:02d}" if args.preview else mod.NAME
        thumb.save(os.path.join(args.out, f"{name}.png"))
        if not args.preview:
            thumb.save(os.path.join(args.out, f"{name}.jpg"), quality=92, optimize=True)
        print(f"#{ep}: {name}.png")


if __name__ == "__main__":
    main()
