"""採譜と生成を 1 本の列で順に流す (GPU を使う処理どうしが取り合わないように)。

ジョブはメモリにだけ置く。結果 (MIDI・テイクの状態) は各プロジェクトに書くので、サーバを止めても残る。
実行中の生成は、パッチを 1 つ作るたびに止める指示を見るので、途中で止められる。採譜はプロセスごと止める。
"""

from __future__ import annotations

import logging
import queue
import threading
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from .project import now

log = logging.getLogger(__name__)

MAX_LOG_LINES = 300
KEEP_FINISHED = 200


# 採譜の段階をログの行から見分ける (tsumugi の出力の書き出し)
_STAGES = (
    ("Separating stems", "separate"),
    ("Transcribing stem", "transcribe"),
    ("Refin", "refine"),
    ("elocity", "velocity"),
    ("beat, chord", "beat"),
)


class LiveTranscription:
    """採譜中に worker から届くノートと進み具合。events は届いた順の [id, stem, start, end, pitch, final] で、
    画面は since (前回までに受け取った数) から後ろだけを取りに来る。同じ id は伸びていく途中のノートの更新"""

    def __init__(self) -> None:
        self.events: list[list] = []
        self.stems: dict[str, dict] = {}  # ステム名 -> {"duration", "pos", "done"} (届いた順)
        self.stage: str | None = None
        # 今の段階の進み具合 {"done", "total"}。段階が変わったら None に戻す
        self.progress: dict | None = None
        self._lock = threading.Lock()

    def _stem(self, name: str) -> dict:
        return self.stems.setdefault(name, {"duration": 0.0, "pos": 0.0, "done": False})

    def on_live(self, kind: str, data: dict) -> None:
        with self._lock:
            stem = str(data.get("stem", ""))
            if kind == "stem":
                self.stems[stem] = {"duration": float(data.get("duration", 0.0)), "pos": 0.0, "done": False}
                self.stage = "transcribe"
                self.progress = None  # 分離は終わっている
            elif kind == "progress":
                self.stage = str(data.get("stage") or self.stage or "")
                self.progress = {"done": int(data.get("done", 0)), "total": int(data.get("total", 0))}
            elif kind == "notes":
                info = self._stem(stem)
                info["pos"] = max(info["pos"], float(data.get("pos", 0.0)))
                for note_id, start, end, pitch, final in data.get("notes", []):
                    self.events.append([note_id, stem, start, end, pitch, final])
            elif kind == "stem_done":
                info = self._stem(stem)
                info["done"], info["pos"] = True, info["duration"]

    def on_log(self, line: str) -> None:
        with self._lock:  # on_live / since と揃える (中途半端な組を返さない)
            for prefix, stage in _STAGES:
                if prefix in line:
                    if stage != self.stage:
                        self.progress = None  # 別の段階に移ったら前の段階の進み具合は捨てる
                    self.stage = stage
                    return

    def since(self, start: int) -> dict:
        with self._lock:
            start = max(0, min(start, len(self.events)))
            return {
                "seq": len(self.events),
                "events": self.events[start:],
                "stems": [{"stem": name, **info} for name, info in self.stems.items()],
                "stage": self.stage,
                "progress": self.progress,
            }


@dataclass
class Job:
    id: str
    kind: str  # transcribe | generate
    project: str
    takes: list[str] = field(default_factory=list)
    state: str = "queued"  # queued | running | done | error | cancelled
    progress: float | None = None  # 0..1 (分からなければ None)
    message: str = ""
    error: str | None = None
    log: list[str] = field(default_factory=list)
    created: str = field(default_factory=now)
    started: str | None = None
    finished: str | None = None
    _stop: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)
    _ended: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)
    # 採譜の途中経過 (LiveTranscription)。大きいので to_dict には入れず、別の API で差分だけ返す
    _live: LiveTranscription | None = field(default=None, repr=False, compare=False)

    @property
    def active(self) -> bool:
        return self.state in ("queued", "running")

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def add_log(self, line: str) -> None:
        self.log.append(line)
        del self.log[:-MAX_LOG_LINES]

    def wait(self, timeout: float | None = None) -> bool:
        return self._ended.wait(timeout)


class Runner:
    """handle(job) で中身を実行し、on_end(job) で後片付け (取り消し・失敗をプロジェクトに書く) をする"""

    def __init__(self, handle: Callable[[Job], None], on_end: Callable[[Job], None]) -> None:
        self._handle = handle
        self._on_end = on_end
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._work, name="cover-studio-runner", daemon=True)
        self._thread.start()

    def close(self) -> None:
        with self._lock:
            for job in self._jobs.values():
                if job.state == "queued":
                    self._end(job, "cancelled")
                elif job.state == "running":
                    job._stop.set()
        self._queue.put(None)
        self._thread.join(timeout=30)

    def submit(self, kind: str, project: str, takes: list[str] | None = None) -> Job:
        job = Job(uuid.uuid4().hex[:12], kind, project, list(takes or []))
        with self._lock:
            self._jobs[job.id] = job
            finished = [j.id for j in self._jobs.values() if not j.active]
            for jid in finished[: max(len(finished) - KEEP_FINISHED, 0)]:
                del self._jobs[jid]
        self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def jobs(self, project: str | None = None) -> list[Job]:
        """新しい順"""
        with self._lock:
            jobs = [j for j in self._jobs.values() if project is None or j.project == project]
        return jobs[::-1]

    def active_for(self, project: str) -> list[Job]:
        return [j for j in self.jobs(project) if j.active]

    def cancel(self, job_id: str) -> Job | None:
        """待っていれば取り消し、実行中なら止める指示を出す (止まるまで少しかかる)"""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or not job.active:
                return job
            if job.state == "queued":
                self._end(job, "cancelled")
                return job
            job._stop.set()
            job.message = "stopping"  # 画面が言語に合わせて訳す
            return job

    def forget(self, project: str) -> None:
        with self._lock:
            for jid in [j.id for j in self._jobs.values() if j.project == project and not j.active]:
                del self._jobs[jid]

    def _end(self, job: Job, state: str) -> None:
        job.state, job.finished = state, now()
        try:
            self._on_end(job)
        except Exception:
            log.exception("ジョブの後片付けに失敗しました")
        job._ended.set()

    def _work(self) -> None:
        from .engine import Cancelled

        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            job = self._jobs.get(job_id)
            if job is None or job.state != "queued":
                continue
            job.state, job.started = "running", now()
            try:
                self._handle(job)
            except Cancelled:
                self._end(job, "cancelled")
            except Exception as e:  # noqa: BLE001  失敗はジョブに書いて次へ進む
                job.add_log(traceback.format_exc().rstrip())
                log.error("%s", traceback.format_exc().rstrip())
                job.error = f"{type(e).__name__}: {e}"
                self._end(job, "error")
            else:
                job.progress = 1.0
                self._end(job, "done")
