# local-livestream

`core/` に `qwen3-ASR` ベースのリアルタイム文字起こし処理を実装しています。

## 主要ファイル

- `core/audio.py`: マイク入力 + 無音区間での発話切り出し
- `core/asr.py`: Qwen3-ASR のロードとストリーミング推論ラッパ
- `core/realtime_transcriber.py`: 録音スレッドと文字起こしスレッドの統合
- `main.py`: エントリポイント

## 実行

```bash
uv sync
uv run python main.py
```

発話が区切られるたびに、文字起こし結果を標準出力に表示します。
