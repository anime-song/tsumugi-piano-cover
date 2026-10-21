"""好みのベクトル (piano_cover.style) の置き場所。プロジェクトの集まり (root) の .styles/<id>.json に 1 つずつ置く。

  id / name / created
  state       queued | running | done | error | cancelled
  model       学習したモデル ({id, label})。好みのベクトルはそのモデルの演奏者の埋め込みの中のものなので、同じモデルでだけ使う
  takes       学習に使ったお気に入り ([{project, take}]、学習を始めたときのもの)
  init        初期値にした演奏者の番号 (お気に入りで使った演奏者の埋め込みの平均)
  used        実際に使えたテイクの数 / loss_before / loss_after / distance (piano_cover.style.StyleResult)
  vector      学習したベクトル (done のときだけ。一覧の API では返さない)
"""

from __future__ import annotations

import re
import threading
import uuid
from pathlib import Path

from .project import now, read_json, write_json

STYLE_STATES = ("queued", "running", "done", "error", "cancelled")


class StyleStore:
    def __init__(self, root: Path) -> None:
        self.dir = Path(root) / ".styles"
        self._lock = threading.RLock()

    def path(self, style_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{12}", style_id):
            raise KeyError(style_id)
        return self.dir / f"{style_id}.json"

    def create(self, name: str, model: dict, takes: list[dict], init: list[int]) -> dict:
        style = {
            "id": uuid.uuid4().hex[:12],
            "name": name,
            "created": now(),
            "state": "queued",
            "error": None,
            "model": model,
            "takes": takes,
            "init": init,
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            write_json(self.path(style["id"]), style)
        return style

    def get(self, style_id: str) -> dict:
        path = self.path(style_id)
        with self._lock:
            if not path.is_file():
                raise KeyError(style_id)
            return read_json(path)

    def list(self) -> list[dict]:
        """新しい順"""
        if not self.dir.is_dir():
            return []
        with self._lock:
            styles = [read_json(p) for p in self.dir.glob("*.json")]
        return sorted(styles, key=lambda s: s["created"], reverse=True)

    def update(self, style_id: str, **values) -> dict:
        with self._lock:
            style = self.get(style_id)
            style.update(values)
            write_json(self.path(style_id), style)
            return style

    def delete(self, style_id: str) -> None:
        with self._lock:
            self.path(style_id).unlink(missing_ok=True)

    def recover(self) -> None:
        """サーバを止めたときに待ち・学習中だったもの (ジョブはメモリにしかない) を中断にする"""
        for style in self.list():
            if style["state"] in ("queued", "running"):
                self.update(style["id"], state="error", error="サーバを止めたので中断しました")
