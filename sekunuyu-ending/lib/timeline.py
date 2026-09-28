"""映像と音で共有するタイムライン（秒）。

テンポ 150BPM（1拍 = 0.4秒、1小節 = 1.6秒）に合わせて演出を置いている。
"""

BPM = 150
BEAT = 60 / BPM
BAR = 4 * BEAT

# ---- シーンA：ロゴ（参考動画の「ワールドマップ＋アバター＋ロゴ」風）
A_CURTAIN_OPEN = (0.0, 0.75)    # 雲が左右に開く
A_AVATAR_POP = 0.35             # アバターがポンッと出る
A_LOGO_DROP = 0.90              # ロゴが落ちてくる
A_LOGO_HIT = 1.08               # ロゴ着地（ドンッ！）
A_RIBBON = 1.32                 # 「GAME CHANNEL」リボン
A_SHINE = 1.95                  # ロゴがキラーン
A_CURTAIN_CLOSE = (3.55, 4.0)   # 雲が閉じる

# ---- シーンB：終了画面
B_START = 4.0
B_CURTAIN_OPEN = (4.05, 4.75)
B_MUSIC = 4.25                  # BGM 1小節目
B_TITLE = 4.35                  # 「ご視聴ありがとう！」が1文字ずつ落ちてくる
B_TITLE_STEP = 0.06
B_FRAMES = 4.70                 # 動画枠がスライドイン
B_RING = 5.10                   # 登録ボタンの枠
B_AVATAR = 5.30                 # アバターがひょこっと出る
B_BUBBLE = 5.80                 # 吹き出し
B_TYPE = 6.00                   # 吹き出しの文字（タイプライター）
B_TYPE_STEP = 0.06
B_THUMB = 6.30                  # 高評価
B_CLICKS = (8.0, 13.2)          # カーソルが登録ボタンをポチッ
B_SWAP = B_MUSIC + 8 * BAR      # 「また見てね！」に切り替え（BGM 9小節目）
B_FINAL = B_MUSIC + 9 * BAR     # 最後のジャーン
B_SWAP_TYPE = B_SWAP + 0.25

ENDING_DUR = 20.0
OPENING_DUR = 4.1

# YouTube の終了画面要素を出す目安（エンディング開始からの秒数）
ENDSCREEN_FROM = 5.0
