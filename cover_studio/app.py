"""Cover Studio の API (FastAPI)。重い処理 (採譜・生成) は Runner の列に流し、画面は状態を問い合わせて進み具合を出す。

  設定         : GET /api/config、POST /api/engine/unload
  プロジェクト : GET/POST /api/projects、GET/PATCH/DELETE /api/projects/{pid}
  原曲         : POST …/transcribe、PUT …/source (手元の MIDI)、GET …/audio、…/source.mid、…/source/view
  テイク       : POST …/takes (生成)、PATCH/DELETE …/takes/{tid}、GET …/takes/{tid}/cover.mid、…/view
  ジョブ       : GET /api/jobs、DELETE /api/jobs/{id} (取り消し・停止)

{pid} はプロジェクトのフォルダ名。web/ のビルド (cover_studio/static) があれば / で配る。
"""

from __future__ import annotations

import random
import shutil
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from .engine import Engine
from .jobs import Job, Runner
from .project import Project, projects, safe_name

STATIC_DIR = Path(__file__).parent / "static"
AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".aac")
MIDI_EXTS = (".mid", ".midi")


class ContinueFrom(BaseModel):
    take: str
    seconds: float = Field(ge=0)


class GenerateIn(BaseModel):
    params: dict = {}
    model: str | None = None
    count: int = Field(default=2, ge=1, le=4)
    seed: int | None = None
    continue_from: ContinueFrom | None = None


class ProjectPatch(BaseModel):
    title: str | None = None


class TakePatch(BaseModel):
    name: str | None = None
    favorite: bool | None = None
    memo: str | None = None


def create_app(root: str | Path, engine: Engine) -> FastAPI:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for p in projects(root):
        p.recover()

    def open_project(pid: str) -> Project:
        d = root / pid
        if pid in ("", ".", "..") or Path(pid).name != pid or not (d / "project.json").is_file():
            raise HTTPException(404, f"プロジェクトがありません: {pid}")
        return Project(d)

    # ------------------------------------------------------------ ジョブの中身
    def handle(job: Job) -> None:
        p = open_project(job.project)

        def progress(value: float | None, message: str) -> None:
            job.progress, job.message = value, message

        if job.kind == "transcribe":
            p.update(transcribe={"state": "running", "error": None})
            engine.transcribe(p, progress, job._stop.is_set, job.add_log)
        else:
            engine.generate(p, job.takes, progress, job._stop.is_set, job.add_log)

    def on_end(job: Job) -> None:
        try:
            p = open_project(job.project)
        except HTTPException:
            return  # プロジェクトごと消された
        if job.kind == "transcribe":
            state = {"done": "done", "cancelled": "cancelled"}.get(job.state, "error")
            p.update(transcribe={"state": state, "error": job.error})
            return
        for take_id in job.takes:
            try:
                take = p.take(take_id)
            except KeyError:
                continue
            if take["state"] in ("queued", "running"):
                state = "cancelled" if job.state == "cancelled" else "error"
                p.update_take(take_id, state=state, error=job.error)

    runner = Runner(handle, on_end)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        runner.close()

    app = FastAPI(title="Cover Studio", lifespan=lifespan)
    app.state.runner = runner
    app.state.engine = engine

    def detail(p: Project) -> dict:
        data = p.data
        jobs = runner.jobs(p.id)
        live = {}
        for job in jobs:
            if job.active and job.kind == "generate":
                for take_id in job.takes:
                    live[take_id] = job
        takes = []
        for take in p.takes():
            job = live.get(take["id"])
            if job is not None:
                take = {**take, "job": job.id, "progress": job.progress, "message": job.message}
            takes.append(take)
        transcribe = data.get("transcribe")
        job = next((j for j in jobs if j.kind == "transcribe"), None)
        if transcribe is not None and job is not None:
            transcribe = {**transcribe, "job": job.id, "message": job.message, "log": job.log[-80:]}
            if job.active:
                transcribe["state"] = job.state
        return {
            "id": p.id,
            "title": data["title"],
            "created": data["created"],
            "audio": p.audio_path.name if p.audio_path else None,
            "source": data.get("source") if p.has_source else None,
            "transcribe": transcribe,
            "takes": takes,
            "busy": any(j.active for j in jobs),
        }

    def summary(p: Project) -> dict:
        data = p.data
        takes = p.takes()
        return {
            "id": p.id,
            "title": data["title"],
            "created": data["created"],
            "audio": p.audio_path.name if p.audio_path else None,
            "has_source": p.has_source,
            "transcribe": data.get("transcribe"),
            "takes": len(takes),
            "favorites": sum(t.get("favorite", False) for t in takes),
            "busy": bool(runner.active_for(p.id)),
        }

    def save_upload(upload: UploadFile, exts: tuple[str, ...]) -> Path:
        name = Path(upload.filename or "").name
        if Path(name).suffix.lower() not in exts:
            raise HTTPException(400, f"対応していないファイルです ({' / '.join(exts)}): {name}")
        path = root / ".uploads" / uuid.uuid4().hex / name
        path.parent.mkdir(parents=True)
        with open(path, "wb") as f:
            shutil.copyfileobj(upload.file, f)
        return path

    def submit_transcribe(p: Project) -> Job:
        if engine.tsumugi is None:
            raise HTTPException(400, "tsumugi が見つからないので採譜できません (MIDI を一緒に入れてください)")
        p.update(transcribe={"state": "queued", "error": None})
        return runner.submit("transcribe", p.id)

    # ------------------------------------------------------------ 設定
    @app.get("/api/config")
    def config():
        from piano_cover.generate import CoverParams

        return {
            **engine.describe(),
            "defaults": asdict(CoverParams()),
            "busy": any(j.active for j in runner.jobs()),
        }

    @app.post("/api/engine/unload")
    def unload():
        if any(j.state == "running" for j in runner.jobs()):
            raise HTTPException(409, "実行中のジョブがあります")
        engine.unload()
        return engine.describe()

    # ------------------------------------------------------------ プロジェクト
    @app.get("/api/projects")
    def list_projects():
        return [summary(p) for p in projects(root)]

    @app.post("/api/projects")
    def create_project(
        audio: Annotated[UploadFile | None, File()] = None,
        midi: Annotated[UploadFile | None, File()] = None,
        title: Annotated[str, Form()] = "",
    ):
        """音源 (と、あれば採譜済みの MIDI) からプロジェクトを作る。MIDI がなければ tsumugi の採譜を始める"""
        if audio is None and midi is None:
            raise HTTPException(400, "音源か MIDI のどちらかが要ります")
        audio_path = save_upload(audio, AUDIO_EXTS) if audio is not None else None
        midi_path = save_upload(midi, MIDI_EXTS) if midi is not None else None
        try:
            name = title.strip() or Path((audio_path or midi_path).name).stem
            p = Project.create(root, name, audio_path, midi_path)
        finally:
            for path in (audio_path, midi_path):
                if path is not None:
                    shutil.rmtree(path.parent, ignore_errors=True)
        if midi_path is None:
            submit_transcribe(p)
        return detail(p)

    @app.get("/api/projects/{pid}")
    def get_project(pid: str):
        return detail(open_project(pid))

    @app.patch("/api/projects/{pid}")
    def update_project(pid: str, body: ProjectPatch):
        p = open_project(pid)
        if body.title is not None and body.title.strip():
            p.update(title=body.title.strip())
        return detail(p)

    @app.delete("/api/projects/{pid}", status_code=204)
    def delete_project(pid: str):
        p = open_project(pid)
        if runner.active_for(pid):
            raise HTTPException(409, "実行中・待ちのジョブがあります")
        p.delete()
        runner.forget(pid)

    # ------------------------------------------------------------ 原曲
    @app.post("/api/projects/{pid}/transcribe")
    def transcribe(pid: str):
        p = open_project(pid)
        if any(j.kind == "transcribe" for j in runner.active_for(pid)):
            raise HTTPException(409, "採譜は実行中です")
        submit_transcribe(p)
        return detail(p)

    @app.put("/api/projects/{pid}/source")
    def upload_source(pid: str, midi: Annotated[UploadFile, File()]):
        p = open_project(pid)
        if any(j.kind == "transcribe" for j in runner.active_for(pid)):
            raise HTTPException(409, "採譜の実行中は差し替えられません")
        path = save_upload(midi, MIDI_EXTS)
        try:
            p.set_source(path, "upload")
        finally:
            shutil.rmtree(path.parent, ignore_errors=True)
        return detail(p)

    @app.get("/api/projects/{pid}/audio")
    def get_audio(pid: str):
        p = open_project(pid)
        if p.audio_path is None or not p.audio_path.is_file():
            raise HTTPException(404, "音源がありません")
        return FileResponse(p.audio_path)

    @app.get("/api/projects/{pid}/source.mid")
    def get_source_midi(pid: str):
        p = open_project(pid)
        if not p.has_source:
            raise HTTPException(404, "原曲の MIDI はまだありません")
        return FileResponse(p.source_path, media_type="audio/midi", filename=f"{safe_name(p.data['title'])}_source.mid")

    @app.get("/api/projects/{pid}/source/view")
    def get_source_view(pid: str):
        from .midi_view import source_view

        p = open_project(pid)
        if not p.has_source:
            raise HTTPException(404, "原曲の MIDI はまだありません")
        return source_view(p.source_path)

    # ------------------------------------------------------------ テイク
    @app.post("/api/projects/{pid}/takes")
    def generate(pid: str, body: GenerateIn):
        from piano_cover.generate import CoverParams

        p = open_project(pid)
        if not p.has_source:
            raise HTTPException(400, "原曲の MIDI がまだありません")
        try:
            entry = engine.entry(body.model)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        if body.continue_from is not None:
            try:
                source_take = p.take(body.continue_from.take)
            except KeyError as e:
                raise HTTPException(404, f"テイクがありません: {body.continue_from.take}") from e
            if source_take["state"] != "done":
                raise HTTPException(400, "生成が終わったテイクからだけ続きを作れます")
        params = asdict(CoverParams.from_dict(body.params))
        record = {
            "params": params,
            "model": {"id": entry.id, "label": entry.label},
            "seed": body.seed if body.seed is not None else random.randrange(2**31),
            "count": body.count,
            "continue_from": body.continue_from.model_dump() if body.continue_from else None,
        }
        takes = p.new_takes(body.count, record)
        runner.submit("generate", pid, [t["id"] for t in takes])
        return detail(p)

    def find_take(p: Project, tid: str) -> dict:
        try:
            return p.take(tid)
        except KeyError as e:
            raise HTTPException(404, f"テイクがありません: {tid}") from e

    @app.patch("/api/projects/{pid}/takes/{tid}")
    def update_take(pid: str, tid: str, body: TakePatch):
        p = open_project(pid)
        find_take(p, tid)
        return p.update_take(tid, **body.model_dump(exclude_none=True))

    @app.delete("/api/projects/{pid}/takes/{tid}", status_code=204)
    def delete_take(pid: str, tid: str):
        p = open_project(pid)
        find_take(p, tid)
        if any(tid in j.takes for j in runner.active_for(pid)):
            raise HTTPException(409, "生成中・待ちのテイクは先に止めてください")
        p.delete_take(tid)

    @app.get("/api/projects/{pid}/takes/{tid}/cover.mid")
    def get_take_midi(pid: str, tid: str):
        p = open_project(pid)
        take = find_take(p, tid)
        path = p.take_midi(tid)
        if not path.is_file():
            raise HTTPException(404, "まだ生成していません")
        name = f"{safe_name(p.data['title'])}_{take['name'] or tid}.mid"
        return FileResponse(path, media_type="audio/midi", headers={"Content-Disposition": _attachment(name)})

    @app.get("/api/projects/{pid}/takes/{tid}/view")
    def get_take_view(pid: str, tid: str):
        from .midi_view import cover_view

        p = open_project(pid)
        find_take(p, tid)
        path = p.take_midi(tid)
        if not path.is_file():
            raise HTTPException(404, "まだ生成していません")
        return cover_view(path)

    # ------------------------------------------------------------ ジョブ
    @app.get("/api/jobs")
    def list_jobs(project: str | None = None):
        return [j.to_dict() for j in runner.jobs(project)]

    @app.delete("/api/jobs/{job_id}")
    def cancel_job(job_id: str):
        job = runner.cancel(job_id)
        if job is None:
            raise HTTPException(404, f"ジョブがありません: {job_id}")
        return job.to_dict()

    # ------------------------------------------------------------ 画面
    @app.exception_handler(ValueError)
    def value_error(_, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/{path:path}", include_in_schema=False)
    def web(path: str):
        """web/ のビルドを配る。ファイルでないパス (/p/<id> など画面の中の行き先) には index.html を返す"""
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404)
        if not (STATIC_DIR / "index.html").exists():
            return RedirectResponse("/docs")
        file = (STATIC_DIR / path).resolve()
        if path and file.is_file() and file.is_relative_to(STATIC_DIR.resolve()):
            return FileResponse(file)
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    return app


def _attachment(name: str) -> str:
    return f"attachment; filename*=UTF-8''{quote(name)}"
