# local-livestream

`core/` に `qwen3-ASR` ベースのリアルタイム文字起こし処理を実装しています。
WSL では `sounddevice + PortAudio` が扱いづらいため、ブラウザマイクを使う Gradio サーバーモードを標準起動にしています。

## 主要ファイル

- `core/audio.py`: マイク入力 + 無音区間での発話切り出し
- `core/asr.py`: Qwen3-ASR のロードとストリーミング推論ラッパ
- `core/realtime_transcriber.py`: 録音スレッドと文字起こしスレッドの統合
- `core/gradio_server.py`: ブラウザ音声を受ける WSL 向けサーバー
- `main.py`: エントリポイント

## 実行

```bash
uv sync
uv run python main.py
```

起動後にブラウザで `http://localhost:7860` を開き、マイク許可をすると発話ごとに文字起こしが追記されます。

## WSL メモ

- WSL 上で `uv run python main.py` を起動
- Windows 側ブラウザで `http://localhost:7860` へアクセス
- 音声はブラウザ -> Gradio -> WSL の Qwen3-ASR に送られます
