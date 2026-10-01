"""Cover Studio のサーバを起動する。

python -m cover_studio
python -m cover_studio --root D:/covers --port 8080 --no-browser --device cpu
python -m cover_studio --hub-repo anime-song/tsumugi-piano-cover --checkpoints-dir ""   # 公開した重みだけを使う

画面は web/ を npm run build したもの (cover_studio/static)。ないときは API の説明 (/docs) を開く。
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import webbrowser
from pathlib import Path

from piano_ar.hub import HF_REPO

from .app import STATIC_DIR, create_app
from .engine import REPO_ROOT, Engine, TsumugiSetup, find_models


def _optional_dir(value: str) -> Path | None:
    return Path(value).resolve() if value else None


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m cover_studio", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "projects", help="プロジェクトを置くフォルダ")
    parser.add_argument(
        "--pretrained-dir",
        type=_optional_dir,
        default=REPO_ROOT / "pretrained",
        help="piano_cover.export の出力を探すフォルダ (空文字で探さない)",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=_optional_dir,
        default=REPO_ROOT / "checkpoints",
        help="学習のチェックポイント (*/best.pt) を探すフォルダ (空文字で探さない)",
    )
    parser.add_argument("--hub-repo", default=HF_REPO, help="Hugging Face の repository (空文字で使わない)")
    parser.add_argument("--device", default="auto", help="auto / cuda / cpu")
    parser.add_argument("--tsumugi-dir", type=Path, default=None, help="tsumugi のリポジトリ (既定 .tsumugi/tsumugi)")
    parser.add_argument("--tsumugi-python", type=Path, default=None, help="tsumugi の環境の Python (既定 その .venv)")
    parser.add_argument("--ffmpeg-dir", type=Path, default=None, help="FFmpeg の共有ライブラリ版の bin (Windows)")
    parser.add_argument("--tsumugi-compile", action="store_true", help="採譜で torch.compile を使う")
    parser.add_argument("--tsumugi-device", default="auto", help="採譜のデバイス (auto / cuda / cpu)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    import uvicorn

    models = find_models(args.pretrained_dir, args.checkpoints_dir, args.hub_repo or None)
    tsumugi = TsumugiSetup.detect(args.tsumugi_dir, args.tsumugi_python, args.ffmpeg_dir)
    if tsumugi is not None:
        tsumugi.compile = args.tsumugi_compile
        tsumugi.device = args.tsumugi_device
        print(f"tsumugi: {tsumugi.dir} ({tsumugi.python})")
    else:
        print("tsumugi が見つかりません。採譜は使えません (採譜済みの MIDI を音源と一緒に入れれば生成はできます)")
    print("モデル: " + (", ".join(m.label for m in models) or "なし"))
    if not (STATIC_DIR / "index.html").exists():
        print("画面がまだビルドされていません: cd web && npm install && npm run build")

    app = create_app(args.root, Engine(models, args.device, tsumugi))
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, [f"http://{args.host}:{args.port}/"]).start()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
