"""1 曲 = 1 フォルダのプロジェクトと、そこから生成したテイク。

projects/<曲名>/
  project.json            曲名・音源・原曲の MIDI の出所・採譜の状態
  audio/<元のファイル名>  音源 (なくてもよい。MIDI だけのプロジェクト)
  source.mid              原曲の MIDI (tsumugi の出力かアップロード)
  takes/<id>/take.json    テイクの設定 (生成の設定・seed・モデル)・状態・名前・お気に入り・メモ
  takes/<id>/cover.mid    生成したカバー
  takes/<id>/patches.json パッチごとのトークン列 (途中から作り直すときに先頭をそのまま使う)

json は読むたびにファイルから読み、書くときは読み直してから変える (生成のスレッドと UI が同時に書いても消えない)。
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

_LOCKS: dict[Path, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()

TAKE_STATES = ("queued", "running", "done", "error", "cancelled")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _lock(path: Path) -> threading.RLock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(path, threading.RLock())


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name).strip(" .")
    return name[:80] or "song"


class Project:
    def __init__(self, path: str | Path) -> None:
        # 絶対パスにしておく (tsumugi のプロセスは別の作業フォルダで動く)
        self.dir = Path(path).resolve()
        if not (self.dir / "project.json").is_file():
            raise FileNotFoundError(f"プロジェクトではありません: {self.dir}")

    @property
    def id(self) -> str:
        return self.dir.name

    @property
    def data(self) -> dict:
        with _lock(self.dir):
            return read_json(self.dir / "project.json")

    def update(self, **values) -> None:
        with _lock(self.dir):
            data = read_json(self.dir / "project.json")
            data.update(values)
            write_json(self.dir / "project.json", data)

    # ------------------------------------------------------------ 作る・消す
    @classmethod
    def create(cls, root: Path, title: str, audio: Path | None = None, midi: Path | None = None) -> Project:
        root = Path(root).resolve()
        base = safe_name(title)
        d, n = root / base, 2
        while d.exists():
            d, n = root / f"{base} ({n})", n + 1
        d.mkdir(parents=True)
        data = {
            "version": 1,
            "title": title,
            "created": now(),
            "audio": None,
            "audio_sha1": None,
            "source": None,  # {"origin": "tsumugi" | "upload", "created": ...}
            "transcribe": None,  # {"state": "queued" | "running" | "done" | "error", "error": ...}
            "next_take": 1,
            "next_batch": 1,
        }
        if audio is not None:
            (d / "audio").mkdir()
            shutil.copy2(audio, d / "audio" / audio.name)
            data["audio"] = f"audio/{audio.name}"
            data["audio_sha1"] = file_sha1(audio)
        write_json(d / "project.json", data)
        p = cls(d)
        if midi is not None:
            p.set_source(midi, "upload")
        return p

    def delete(self) -> None:
        shutil.rmtree(self.dir)

    # ------------------------------------------------------------ ファイル
    @property
    def audio_path(self) -> Path | None:
        audio = self.data.get("audio")
        return self.dir / audio if audio else None

    @property
    def source_path(self) -> Path:
        return self.dir / "source.mid"

    @property
    def has_source(self) -> bool:
        return self.source_path.is_file()

    def set_source(self, midi: Path, origin: str) -> None:
        tmp = self.source_path.with_suffix(".mid.tmp")
        shutil.copy2(midi, tmp)
        tmp.replace(self.source_path)
        self.update(source={"origin": origin, "created": now()})

    # ------------------------------------------------------------ テイク
    @property
    def takes_dir(self) -> Path:
        return self.dir / "takes"

    def take_dir(self, take_id: str) -> Path:
        if not re.fullmatch(r"t\d+", take_id):
            raise KeyError(take_id)
        return self.takes_dir / take_id

    def new_takes(self, count: int, record: dict) -> list[dict]:
        """count 本のテイクを「待ち」で作る (同じ設定で一度に生成する 1 組。batch で組を見分ける)"""
        with _lock(self.dir):
            data = read_json(self.dir / "project.json")
            first, batch = data.get("next_take", 1), data.get("next_batch", 1)
            data["next_take"], data["next_batch"] = first + count, batch + 1
            write_json(self.dir / "project.json", data)
        takes = []
        for i in range(count):
            number = first + i
            take = {
                "id": f"t{number:04d}",
                "number": number,
                "name": "",
                "created": now(),
                "state": "queued",
                "error": None,
                "batch": batch,
                "batch_index": i,
                "favorite": False,
                "memo": "",
                "duration": None,
                "notes": None,
                **record,
            }
            d = self.takes_dir / take["id"]
            d.mkdir(parents=True)
            write_json(d / "take.json", take)
            takes.append(take)
        return takes

    def take(self, take_id: str) -> dict:
        path = self.take_dir(take_id) / "take.json"
        if not path.is_file():
            raise KeyError(take_id)
        with _lock(self.dir):
            return read_json(path)

    def takes(self) -> list[dict]:
        """新しい順"""
        if not self.takes_dir.is_dir():
            return []
        out = []
        with _lock(self.dir):
            for d in self.takes_dir.iterdir():
                if (d / "take.json").is_file():
                    out.append(read_json(d / "take.json"))
        return sorted(out, key=lambda t: t["number"], reverse=True)

    def update_take(self, take_id: str, **values) -> dict:
        path = self.take_dir(take_id) / "take.json"
        with _lock(self.dir):
            take = read_json(path)
            take.update(values)
            write_json(path, take)
            return take

    def delete_take(self, take_id: str) -> None:
        shutil.rmtree(self.take_dir(take_id))

    def take_midi(self, take_id: str) -> Path:
        return self.take_dir(take_id) / "cover.mid"

    def take_patches(self, take_id: str) -> list[list[int]]:
        return json.loads((self.take_dir(take_id) / "patches.json").read_text(encoding="utf-8"))

    def recover(self) -> None:
        """サーバを止めたときに待ち・実行中だったもの (ジョブはメモリにしかない) を中断にする"""
        transcribe = self.data.get("transcribe")
        if transcribe and transcribe.get("state") in ("queued", "running"):
            self.update(transcribe={**transcribe, "state": "error", "error": "サーバを止めたので中断しました"})
        for take in self.takes():
            if take["state"] in ("queued", "running"):
                self.update_take(take["id"], state="cancelled", error="サーバを止めたので中断しました")


def projects(root: Path) -> list[Project]:
    """新しい順"""
    out = []
    if Path(root).is_dir():
        for d in Path(root).iterdir():
            if (d / "project.json").is_file():
                out.append(Project(d))
    return sorted(out, key=lambda p: p.data.get("created", ""), reverse=True)
