"""生成と採譜の中身。カバーモデルは一度読んだら次の生成でも使い回し、別のモデルを選んだときだけ読み直す。

モデルの候補 (find_models):
  export     piano_cover.export の出力 (既定 pretrained/ の下)
  checkpoint 学習のチェックポイント (既定 checkpoints/**/best.pt。隣に planner*.pt があれば差し替えた版も)
  hub        Hugging Face に公開した重み (piano_ar.hub.HF_REPOS の V2・V1。初回にダウンロードする)。一覧の先頭 (既定) は V2
"""

from __future__ import annotations

import gc
import json
import os
import random
import subprocess
import sys
import tempfile
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


def find_models(
    pretrained_dir: Path | None, checkpoints_dir: Path | None, hub_repos: list[str] | tuple[str, ...]
) -> list[ModelEntry]:
    """画面の一覧の先頭が既定のモデルになるので、Hugging Face の公開した重み (hub_repos の順) を先に並べる"""
    from piano_ar.hub import HF_REPO_V1, HF_REPO_V2

    names = {HF_REPO_V2: "V2", HF_REPO_V1: "V1"}
    entries: list[ModelEntry] = [
        ModelEntry(f"hub:{repo}", f"{names[repo]} ({repo})" if repo in names else f"{repo} (Hugging Face)", "hub", repo)
        for repo in hub_repos
    ]
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
        found = [p for p in checkpoints_dir.rglob("best.pt") if _is_cover_checkpoint(p)]
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
        sheetsage_model: str | None = None,
        frames_pca: Path | None = None,
    ) -> None:
        self.models = models
        self.device_name = device
        self.tsumugi = tsumugi
        self.sheetsage_model = sheetsage_model  # None なら sheetsage2_refined の既定 (Hugging Face)
        # 原曲の音声の特徴を使うモデル (frame_dims) の、特徴を作る PCA (piano_cover.audio_frames fit の出力)
        self.frames_pca = frames_pca or REPO_ROOT / "data" / "audio_frames" / "original" / "pca.npz"
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

    def _source(self, project: Project, model, progress: Callable[[float, str], None], log: Callable[[str], None]):
        from piano_ar.beat_tokenizer import BeatTokenizer
        from piano_cover.generate import prepare_source

        self._source_beats(project, progress, log)
        path = project.source_path
        annotation = None
        c = model.config
        if c.sheetsage_beats or c.sheetsage_melody or isinstance(model.tokenizer, BeatTokenizer):
            annotation = self._sheetsage(project, progress, log)
        key = (
            str(path),
            path.stat().st_mtime_ns,
            self._model_id,
            annotation and (str(annotation), annotation.stat().st_mtime_ns),
        )
        frames = self._frames(project, progress, log) if c.frame_dims else None
        if frames is not None:
            key += (str(frames), frames.stat().st_mtime_ns)
        if key not in self._sources:
            self._sources[key] = prepare_source(
                path, model.tokenizer, model.config, annotation, np.load(frames)["frames"] if frames else None
            )
            while len(self._sources) > 8:
                self._sources.popitem(last=False)
        self._sources.move_to_end(key)
        return self._sources[key]

    def _source_beats(
        self, project: Project, progress: Callable[[float, str], None], log: Callable[[str], None]
    ) -> None:
        """原曲の MIDI の拍 (テンポと拍子) を、音源から予測した SheetSage2-Refined の拍にする (音などの秒の位置は変えない)。
        画面の拍の線・カバーの MIDI に写すテンポと拍子・V1 のモデルが読む拍が、どれも SheetSage の拍になる。
        音源のないプロジェクトは tsumugi の拍のまま。音源ごとに 1 回だけ (source の beats に印を付ける)"""
        from piano_cover.metadata import apply_sheetsage_beats

        audio = project.audio_path
        if audio is None or not audio.is_file() or not project.has_source:
            return
        source = project.data.get("source") or {}
        sha1 = project.data.get("audio_sha1")
        if source.get("beats") == "sheetsage" and source.get("beats_audio_sha1") == sha1:
            return
        annotation = json.loads(self._sheetsage(project, progress, log).read_text(encoding="utf-8"))
        if apply_sheetsage_beats(project.source_path, annotation):
            project.update(source={**source, "beats": "sheetsage", "beats_audio_sha1": sha1})
            log("原曲の MIDI の拍を SheetSage2-Refined の拍にしました")

    def _sheetsage(self, project: Project, progress: Callable[[float, str], None], log: Callable[[str], None]) -> Path:
        """原曲の音声の拍とメロディを SheetSage2-Refined (HeadBeats / HeadMelody) で予測し、project/sheetsage.json に置く。
        音声が同じ (audio_sha1) なら前回の結果を使う"""
        out = project.dir / "sheetsage.json"
        audio = project.audio_path
        sha1 = project.data.get("audio_sha1")
        if out.is_file() and (audio is None or json.loads(out.read_text(encoding="utf-8")).get("audio_sha1") == sha1):
            return out
        if audio is None or not audio.is_file():
            raise FileNotFoundError(
                "このモデルは原曲の音声から拍とメロディを予測して使います。音源のあるプロジェクトで生成してください"
            )
        import torch

        from piano_ar.sheetsage import SheetSageAnnotator

        progress(0.0, "beats")  # 画面が言語に合わせて訳す (web/src/i18n.ts の progressText)
        log(f"原曲の拍とメロディを予測しています: {audio.name}")
        annotator = SheetSageAnnotator(self.sheetsage_model, device=str(self.device))
        try:
            result = annotator.annotate(audio)
        finally:
            del annotator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        result["audio_sha1"] = sha1
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        tmp.replace(out)
        log(f"拍 {len(result['beats'])}・メロディ {len(result['melody'])} 音")
        return out

    def _frames(self, project: Project, progress: Callable[[float, str], None], log: Callable[[str], None]) -> Path:
        """原曲の音声の特徴 (piano_cover.audio_frames) を作り、project/frames.npz に置く。
        音声 (audio_sha1) と PCA が同じなら前回の結果を使う"""
        import hashlib

        out = project.dir / "frames.npz"
        audio = project.audio_path
        # 公開した重みは重みの隣の PCA、学習のチェックポイントは --frames-pca (既定は手元で作ったもの)
        pca = Path(getattr(self._model, "frames_pca_path", None) or self.frames_pca)
        if not pca.is_file():
            raise FileNotFoundError(f"原曲の音声の特徴を作る PCA がありません: {pca} (--frames-pca で指定)")
        sha1 = project.data.get("audio_sha1")
        pca_sha1 = hashlib.sha1(pca.read_bytes()).hexdigest()
        if out.is_file():
            saved = np.load(out)
            if str(saved["audio_sha1"]) == str(sha1) and str(saved["pca_sha1"]) == pca_sha1:
                return out
        if audio is None or not audio.is_file():
            raise FileNotFoundError("このモデルは原曲の音声の特徴を使います。音源のあるプロジェクトで生成してください")
        import torch

        from piano_cover.audio_frames import FrameExtractor, FrameProjector

        progress(0.0, "beats")
        log(f"原曲の音声の特徴を作っています: {audio.name}")
        extractor = FrameExtractor(self.sheetsage_model, device=str(self.device))
        try:
            frames = FrameProjector(extractor, pca)(audio)
        finally:
            del extractor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, frames=frames, audio_sha1=np.str_(sha1), pca_sha1=np.str_(pca_sha1))
        tmp.replace(out)
        log(f"特徴 {len(frames)} フレーム")
        return out

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
        from piano_cover.metadata import score_for_cover

        with self._lock:
            first = project.take(take_ids[0])
            entry = self.entry(first["model"]["id"])
            progress(0.0, "loading")  # 画面が言語に合わせて訳す (web/src/i18n.ts の progressText)
            model = self.model(entry, log)
            params = CoverParams.from_dict(first["params"])
            if params.channel and entry.num_channels and params.channel >= entry.num_channels:
                raise ValueError(f"演奏者の番号は 1〜{entry.num_channels - 1} です")
            source = self._source(project, model, progress, log)
            tokenizer = model.tokenizer
            frame_rate = tokenizer.config.frame_rate

            prompt = None
            keep = first.get("continue_from")
            if keep:
                # 指定した時刻を含まないパッチまで (生成は原曲の最初の音から始まるので lead を引く)
                patches = project.take_patches(keep["take"])
                if source.grid is not None:
                    starts = source.grid.beats[source.grid.bar_starts]
                    count = int(np.searchsorted(starts, keep["seconds"] * frame_rate, side="right") - 1)
                else:
                    count = int((keep["seconds"] * frame_rate - source.lead) // tokenizer.patch_frames)
                prompt = patches[: max(0, min(count, len(patches)))] or None

            for take_id in take_ids:
                project.update_take(take_id, state="running")

            def on_patch(done: int, total: int) -> None:
                if should_stop():
                    raise Cancelled()
                if source.grid is not None:
                    starts = source.grid.beats[source.grid.bar_starts]
                    elapsed = starts[min(done, len(starts) - 1)] / frame_rate
                    duration = starts[min(total, len(starts) - 1)] / frame_rate
                else:
                    elapsed = done * tokenizer.config.patch_seconds
                    duration = total * tokenizer.config.patch_seconds
                progress(done / total, f"{elapsed:.0f}/{duration:.0f}")

            seed = first["seed"]
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            covers = generate_covers(model, source, params, num_samples=len(take_ids), prompt=prompt, progress=on_patch)

        # 原曲のメタデータ (テンポ・拍子・調・コードマーカー) は、書き出す前に Score に入れて
        # 1 回だけ書き出す。書き出してから写すと秒 -> tick の丸めが 2 回起きて、音が少しずつずれる
        source_score = None
        if project.has_source:
            try:
                from symusic import Score

                source_score = Score(str(project.source_path)).to("second")
            except Exception:
                log("原曲のメタデータを読めなかったので、メタデータなしで書き出します")
        source_created = (project.data.get("source") or {}).get("created")

        for take_id, cover in zip(take_ids, covers):
            d = project.take_dir(take_id)
            try:
                score_for_cover(tokenizer, cover.events, source_score).dump_midi(str(d / "cover.mid"))
                metadata_copied = source_score is not None
            except Exception:  # メタデータを写せなくても、テイクは落とさない
                log("原曲のメタデータを写せなかったので、メタデータなしで書き出します")
                tokenizer.events_to_midi(cover.events, d / "cover.mid")
                metadata_copied = False
            (d / "patches.json").write_text(json.dumps(cover.patches), encoding="utf-8")
            notes = int((cover.events[:, KIND] == KIND_NOTE).sum()) if len(cover.events) else 0
            project.update_take(
                take_id,
                state="done",
                duration=events_end_frame(cover.events) / frame_rate if len(cover.events) else 0.0,
                notes=notes,
                # どの原曲から作ったかと、メタデータを入れ終えた印 (古いテイクを後から直すときに使う)
                source_created=source_created,
                metadata_copied=metadata_copied,
            )

    # ------------------------------------------------------------ 採譜
    def transcribe(
        self,
        project: Project,
        progress: Callable[[float | None, str], None],
        should_stop: Callable[[], bool],
        log: Callable[[str], None],
        live: Callable[[str, dict], None] | None = None,
    ) -> None:
        """live(種類, 中身) には worker が流す途中経過 (確定したノートなど。tsumugi_worker.emit) を渡す"""
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
            # ステムの音声など大きい作業ファイル (1 曲数百 MB) は、プロジェクト (Google ドライブのこともある) に置かない
            str(Path(tempfile.gettempdir()) / "cover_studio_tsumugi" / project.data["created"].replace(":", "")),
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
            if line.startswith("@@"):
                kind, _, payload = line[2:].partition(" ")
                if live is not None:
                    try:
                        live(kind, json.loads(payload))
                    except json.JSONDecodeError:
                        pass
            elif line:
                log(line)
                progress(None, line[-120:])
        code = process.wait()
        stopped.set()
        if should_stop():
            raise Cancelled()
        if code != 0 or not out.is_file():
            raise RuntimeError(f"tsumugi が失敗しました (終了コード {code})。ログを確かめてください")
        project.set_source(out, "tsumugi")
        self._source_beats(project, progress, log)
        out.unlink()

    def describe(self) -> dict:
        return {
            "models": [asdict(m) for m in self.models],
            "loaded": self._model_id,
            "device": str(self.device),
            "tsumugi": self.tsumugi is not None,
        }
