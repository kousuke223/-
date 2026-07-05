# Web ミニゲーム集

ブラウザだけで遊べる簡単なゲーム集です。

## ページ一覧

| ページ | ファイル | 内容 |
| --- | --- | --- |
| 🎬 ツツミーズ ポートフォリオ | [`profile.html`](profile.html) | 映像クリエイター「ツツミーズ」のポートフォリオ兼コーポレートサイト。名刺のQRコードの遷移先。 |
| 💰 お金クリッカー | [`money.html`](money.html) | クリックでお金を稼ぎ、施設を買って自動収入を増やす放置ゲーム。セーブ＆放置収入あり。 |
| ✊ じゃんけん | [`index.html`](index.html) | 5点先取のじゃんけんゲーム。 |

## 名刺用QRコード

`assets/qr-code.png` は GitHub Pages で公開した `profile.html`
（`https://kousuke223.github.io/-/profile.html`）へのQRコードです。
名刺デザインのQR部分にこの画像を貼り付けてください。

GitHub Pages が別のURLで公開されている場合は、
`assets/qr-code.png` を再生成するか、`profile.html` 内の
リンク先を差し替えてください。

`assets/tailwind.css` は `profile.html` 内で使用しているクラスから
Tailwind CLI でビルドした静的CSSです（CDN不使用）。
`profile.html` のクラスを変更した場合は、Tailwind CLI で再ビルドしてください。

## 遊び方

`money.html`（または `index.html`）をブラウザで開くだけ。サーバーは不要です。

### お金クリッカーの特徴
- 🪙 コインをクリックしてお金を稼ぐ
- 🏪 ショップで施設を買うと自動でお金が増える（放置で増加）
- 💾 ブラウザに自動セーブ（10秒ごと＆閉じる時）
- 🎉 放置していた間の収入も受け取れる（最大8時間分）
