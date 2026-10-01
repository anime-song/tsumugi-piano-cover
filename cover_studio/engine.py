"""生成と採譜の中身。カバーモデルは一度読んだら次の生成でも使い回し、別のモデルを選んだときだけ読み直す。

モデルの候補 (find_models):
  export     piano_cover.export の出力 (既定 pretrained/ の下)
  checkpoint 学習のチェックポイント (既定 checkpoints/*/best.pt。隣に planner*.pt があれば差し替えた版も)
  hub        Hugging Face に公開した重み (piano_ar.hub.HF_REPO。初回にダウンロードする)
"""

from __future__ import annotations

import gc
import json
import os
import random
import subprocess
import sys
import threading
import zipfile
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .project import Project

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKER = Path(__file__).resolve().parent / "tsumugi_worker.py"


class Cancelled(Exception):
    pass


# ================================================================== モデルの候補
@dataclass
class ModelEntry:
    id: str
    label: str
    kind: str  # export | checkpoint | hub
    spec: str  # load_cover に渡すもの (フォルダ・.pt・repository)
    planner: str | None = None
    num_channels: int | None = None


def _is_cover_checkpoint(path: Path) -> bool:
    """torch.save の zip の中の pickle に cover_config の名前があるか (重みを読まずに見分ける)"""
    try:
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith("/data.pkl"))
            with z.open(name) as f:
                head = f.read(1 << 22)
        return b"cover_config" in head
    except (zipfile.BadZipFile, StopIteration, OSError):
        return False


def find_models(pretrained_dir: Path | None, checkpoints_dir: Path | None, hub_repo: str | None) -> list[ModelEntry]:
    entries: list[ModelEntry] = []
    if pretrained_dir is not None and pretrained_dir.is_dir():
        for config_path in sorted(pretrained_dir.rglob("config.json")):
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if config.get("model_type") != "piano_cover":
                continue
            folder = config_path.parent
            rel = folder.relative_to(pretrained_dir.parent).as_posix()
            entries.append(
                ModelEntry(
                    f"export:{rel}", rel, "export", str(folder), None, config["model_config"].get("num_channels")
                )
            )
    if checkpoints_dir is not None and checkpoints_dir.is_dir():
        found = [p for p in checkpoints_dir.glob("*/best.pt") if _is_cover_checkpoint(p)]
        for path in sorted(found, key=lambda p: p.stat().st_mtime, reverse=True):
            rel = path.relative_to(checkpoints_dir).as_posix()
            entries.append(ModelEntry(f"checkpoint:{rel}", rel, "checkpoint", str(path)))
            for planner in sorted(path.parent.glob("planner*.pt")):
                if "features" in planner.name:
                    continue
                entries.append(
                    ModelEntry(
                        f"checkpoint:{rel}+{planner.name}",
                        f"{rel} + {planner.name}",
                        "checkpoint",
                        str(path),
                        str(planner),
                    )
                )
    if hub_repo:
        entries.append(ModelEntry(f"hub:{hub_repo}", f"{hub_repo} (Hugging Face)", "hub", hub_repo))
    return entries


# ================================================================== tsumugi
@dataclass
class TsumugiSetup:
    dir: Path
    python: Path
    ffmpeg_dir: Path | None = None
    compile: bool = False
    device: str = "auto"

    @classmethod
    def detect(cls, tsumugi_dir: Path | None, python: Path | None, ffmpeg_dir: Path | None) -> TsumugiSetup | None:
        d = (tsumugi_dir or REPO_ROOT / ".tsumugi" / "tsumugi").resolve()
        if not (d / "instrument_agnostic_amt").is_dir():
            return None
        if python is None:
            for candidate in (d / ".venv" / "Scripts" / "python.exe", d / ".venv" / "bin" / "python"):
                if candidate.is_file():
                    python = candidate
                    break
            else:
                python = Path(sys.executable)
        if ffmpeg_dir is None and (d.parent / "ffmpeg-shared" / "bin").is_dir():
            ffmpeg_dir = d.parent / "ffmpeg-shared" / "bin"
        return cls(d, python, ffmpeg_dir)


# ================================================================== 本体
class Engine:
    def __init__(
        self,
        models: list[ModelEntry],
        device: str = "auto",
        tsumugi: TsumugiSetup | None = None,
    ) -> None:
        self.models = models
        self.device_name = device
        self.tsumugi = tsumugi
        self._model = None
        self._model_id: str | None = None
        self._sources: OrderedDict[tuple, object] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------ モデル
    @property
    def device(self):
        import torch

        if self.device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_name)

    @property
    def loaded(self) -> str | None:
        return self._model_id

    def entry(self, model_id: str | None) -> ModelEntry:
        if not self.models:
            raise ValueError(
                "使えるモデルがありません (--pretrained-dir / --checkpoints-dir / --hub-repo を確かめてください)"
            )
        if model_id is None:
            return self.models[0]
        for entry in self.models:
            if entry.id == model_id:
                return entry
        raise ValueError(f"モデルがありません: {model_id}")

    def model(self, entry: ModelEntry, log: Callable[[str], None] = print):
        if self._model_id == entry.id:
            return self._model
        self.unload()
        from piano_cover.hub import load_cover

        log(f"モデルを読んでいます: {entry.label}")
        self._model = load_cover(entry.spec, planner=entry.planner, device=self.device)
        self._model_id = entry.id
        entry.num_channels = self._model.decoder.config.num_channels
        return self._model

    def unload(self) -> None:
        """モデルを手放して VRAM を空ける (採譜の前・UI のボタン)"""
        if self._model is None:
            return
        self._model = None
        self._model_id = None
        self._sources.clear()
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _source(self, project: Project, model):
        from piano_cover.generate import prepare_source

        path = project.source_path
        key = (str(path), path.stat().st_mtime_ns, self._model_id)
        if key not in self._sources:
            self._sources[key] = prepare_source(path, model.tokenizer, model.config)
            while len(self._sources) > 8:
                self._sources.popitem(last=False)
        self._sources.move_to_end(key)
        return self._sources[key]

    # ------------------------------------------------------------ 生成
    def generate(
        self,
        project: Project,
        take_ids: list[str],
        progress: Callable[[float, str], None],
        should_stop: Callable[[], bool],
        log: Callable[[str], None],
    ) -> None:
        """take.json (Project.new_takes で作ったもの) の設定で、take_ids の本数を一度に生成して書き込む"""
        import torch

        from piano_ar.evaluation import events_end_frame
        from piano_ar.tokenizer import KIND, KIND_NOTE
        from piano_cover.generate import CoverParams, generate_covers

        with self._lock:
            first = project.take(take_ids[0])
            entry = self.entry(first["model"]["id"])
            progress(0.0, "loading")  # 画面が言語に合わせて訳す (web/src/i18n.ts の progressText)
            model = self.model(entry, log)
            params = CoverParams.from_dict(first["params"])
            if params.channel and entry.num_channels and params.channel >= entry.num_channels:
                raise ValueError(f"演奏者の番号は 1〜{entry.num_channels - 1} です")
            source = self._source(project, model)
            tokenizer = model.tokenizer
            frame_rate = tokenizer.config.frame_rate

            prompt = None
            keep = first.get("continue_from")
            if keep:
                # 指定した時刻を含まないパッチまで (生成は原曲の最初の音から始まるので lead を引く)
                patches = project.take_patches(keep["take"])
                count = int((keep["seconds"] * frame_rate - source.lead) // tokenizer.patch_frames)
                prompt = patches[: max(0, min(count, len(patches)))] or None

            for take_id in take_ids:
                project.update_take(take_id, state="running")

            def on_patch(done: int, total: int) -> None:
                if should_stop():
                    raise Cancelled()
                seconds = tokenizer.config.patch_seconds
                progress(done / total, f"{done * seconds:.0f}/{total * seconds:.0f}")  # 生成した秒数/全体の秒数

            seed = first["seed"]
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            covers = generate_covers(model, source, params, num_samples=len(take_ids), prompt=prompt, progress=on_patch)

        for take_id, cover in zip(take_ids, covers):
            d = project.take_dir(take_id)
            tokenizer.events_to_midi(cover.events, d / "cover.mid")
            (d / "patches.json").write_text(json.dumps(cover.patches), encoding="utf-8")
            notes = int((cover.events[:, KIND] == KIND_NOTE).sum()) if len(cover.events) else 0
            project.update_take(
                take_id,
                state="done",
                duration=events_end_frame(cover.events) / frame_rate if len(cover.events) else 0.0,
                notes=notes,
            )

    # ------------------------------------------------------------ 採譜
    def transcribe(
        self,
        project: Project,
        progress: Callable[[float | None, str], None],
        should_stop: Callable[[], bool],
        log: Callable[[str], None],
    ) -> None:
        if self.tsumugi is None:
            raise RuntimeError("tsumugi が見つかりません (--tsumugi-dir で場所を指定してください)")
        audio = project.audio_path
        if audio is None:
            raise RuntimeError("音源のないプロジェクトは採譜できません")
        # tsumugi は VRAM を多く使うので、カバーモデルを手放してから走らせる
        self.unload()
        out = project.dir / "source.tsumugi.mid"
        command = [
            str(self.tsumugi.python),
            "-X",
            "utf8",
            str(WORKER),
            "--tsumugi-dir",
            str(self.tsumugi.dir),
            "--audio",
            str(audio),
            "--out",
            str(out),
            "--work",
            str(project.dir / "_tsumugi_work"),
        ]
        if self.tsumugi.ffmpeg_dir:
            command += ["--ffmpeg-dir", str(self.tsumugi.ffmpeg_dir)]
        if self.tsumugi.compile:
            command.append("--compile")
        command += ["--device", self.tsumugi.device]
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        env.pop("VIRTUAL_ENV", None)
        progress(None, "starting")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=str(self.tsumugi.dir),
        )
        stopped = threading.Event()

        def watch() -> None:
            while process.poll() is None:
                if should_stop():
                    stopped.set()
                    process.kill()
                    return
                stopped.wait(0.5)

        threading.Thread(target=watch, daemon=True).start()
        assert process.stdout is not None
        for line in process.stdout:  # tqdm の \r も行の区切りになる
            line = line.rstrip()
            if line:
                log(line)
                progress(None, line[-120:])
        code = process.wait()
        stopped.set()
        if should_stop():
            raise Cancelled()
        if code != 0 or not out.is_file():
            raise RuntimeError(f"tsumugi が失敗しました (終了コード {code})。ログを確かめてください")
        project.set_source(out, "tsumugi")
        out.unlink()

    def describe(self) -> dict:
        return {
            "models": [asdict(m) for m in self.models],
            "loaded": self._model_id,
            "device": str(self.device),
            "tsumugi": self.tsumugi is not None,
        }
