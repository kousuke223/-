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

## Google検索対策（SEO）

`robots.txt` / `sitemap.xml` / `profile.html` 内のメタタグ・構造化データ
（JSON-LD）で、検索エンジンに「ツツミーズ」の情報を伝える基本対応を
行っています。ただし、これだけでは検索結果に出るまで時間がかかるため、
早く反映させたい場合は以下も行ってください。

1. [Google Search Console](https://search.google.com/search-console) に
   `https://kousuke223.github.io/-/` を登録（所有権確認が必要）
2. サイトマップとして `https://kousuke223.github.io/-/sitemap.xml` を送信
3. 「URL検査」ツールで `profile.html` のURLを検査し、インデックス登録をリクエスト

なお「動画編集」のような一般的な単語単体での上位表示は、競合が非常に
多いため、個人のポートフォリオサイト単体では現実的に難しいです。
「ツツミーズ」「堤康亮」のような固有名詞であれば、上記対応後、比較的
出やすくなります。

## 遊び方

`money.html`（または `index.html`）をブラウザで開くだけ。サーバーは不要です。

### お金クリッカーの特徴
- 🪙 コインをクリックしてお金を稼ぐ
- 🏪 ショップで施設を買うと自動でお金が増える（放置で増加）
- 💾 ブラウザに自動セーブ（10秒ごと＆閉じる時）
- 🎉 放置していた間の収入も受け取れる（最大8時間分）
