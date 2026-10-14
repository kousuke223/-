"""サムネで使うフォント（Google Fonts / SIL Open Font License）。初回に自動でダウンロードする。"""
from __future__ import annotations

import os
import re
import urllib.request

FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")

FONTS = {
    "NotoSansJP-Black.ttf": "Noto+Sans+JP:wght@900",
    "DelaGothicOne-Regular.ttf": "Dela+Gothic+One",
    "PressStart2P-Regular.ttf": "Press+Start+2P",
}

BLACK = os.path.join(FONT_DIR, "NotoSansJP-Black.ttf")
DELA = os.path.join(FONT_DIR, "DelaGothicOne-Regular.ttf")
PIXEL = os.path.join(FONT_DIR, "PressStart2P-Regular.ttf")   # 英字のブロック文字の元にするドットフォント


def ensure_fonts() -> None:
    os.makedirs(FONT_DIR, exist_ok=True)
    for fname, family in FONTS.items():
        path = os.path.join(FONT_DIR, fname)
        if os.path.exists(path):
            continue
        print(f"フォントをダウンロード中: {fname}")
        req = urllib.request.Request(f"https://fonts.googleapis.com/css2?family={family}",
                                     headers={"User-Agent": "Wget/1.12"})
        css = urllib.request.urlopen(req).read().decode()
        url = re.search(r"url\((https://[^)]+\.ttf)\)", css).group(1)
        urllib.request.urlretrieve(url, path)
