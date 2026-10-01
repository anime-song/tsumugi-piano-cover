"""ビルド済みの画面 (web/ を npm run build したもの) の置き場と、GitHub の Release からの取得。

画面は GitHub Actions (.github/workflows/web.yml) が main への push ごとにビルドし、Release「web-latest」に
web-dist.zip として置く。Node.js の無い環境 (Colab など) では python -m cover_studio がそれを取ってくる。

    python -m cover_studio.web            # 無ければ取ってくる
    python -m cover_studio.web --force    # 取り直す
"""

from __future__ import annotations

import argparse
import io
import logging
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
WEB_DIST_URL = "https://github.com/anime-song/tsumugi-piano-cover/releases/download/web-latest/web-dist.zip"


def has_web(dest: Path = STATIC_DIR) -> bool:
    return (dest / "index.html").is_file()


def download_web(url: str = WEB_DIST_URL, dest: Path = STATIC_DIR) -> Path:
    """web-dist.zip を取ってきて dest に展開する (中身を入れ替える)"""
    log.info("画面を取得しています: %s", url)
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        if "index.html" not in z.namelist():
            raise ValueError(f"画面の zip に index.html がありません: {url}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".static-", dir=dest.parent))
        try:
            z.extractall(tmp)
            shutil.rmtree(dest, ignore_errors=True)
            tmp.rename(dest)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    return dest


def ensure_web(url: str = WEB_DIST_URL, dest: Path = STATIC_DIR) -> bool:
    """画面が無ければ取ってくる。取れなければ False (API だけで動く)"""
    if has_web(dest):
        return True
    try:
        download_web(url, dest)
        return True
    except Exception as e:  # noqa: BLE001  ネットが無いなど。画面なしでも API は使える
        log.warning("画面を取得できませんでした (%s)。web/ で npm run build してください", e)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m cover_studio.web", description=__doc__)
    parser.add_argument("--force", action="store_true", help="あっても取り直す")
    parser.add_argument("--url", default=WEB_DIST_URL)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.force or not has_web():
        print(download_web(args.url))
    else:
        print(f"画面はあります: {STATIC_DIR}")
