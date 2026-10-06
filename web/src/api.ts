// API (cover_studio/app.py) の型と呼び出し

export type Params = {
  channel: number;
  temperature: number;
  top_p: number;
  source_cfg: number;
  channel_cfg: number;
  onset_bias: number;
  onset_bias_width: number;
  dynamics: number;
  density: number;
  fill: number;
  above: number;
  span: number;
  arrangement: boolean;
  seconds: number | null;
};

export type ModelEntry = {
  id: string;
  label: string;
  kind: "export" | "checkpoint" | "hub";
  spec: string;
  planner: string | null;
  num_channels: number | null;
};

// cover_studio/app.py の API_VERSION と同じにする (違えばサーバが古い)
export const API_VERSION = 2;

export type Config = {
  models: ModelEntry[];
  loaded: string | null;
  device: string;
  tsumugi: boolean;
  defaults: Params;
  busy: boolean;
  api_version?: number;
};

export type TakeState = "queued" | "running" | "done" | "error" | "cancelled";

export type ContinueFrom = { take: string; seconds: number };

export type Take = {
  id: string;
  number: number;
  name: string;
  created: string;
  state: TakeState;
  error: string | null;
  batch: number;
  batch_index: number;
  favorite: boolean;
  memo: string;
  duration: number | null;
  notes: number | null;
  params: Params;
  model: { id: string; label: string };
  seed: number;
  count: number;
  continue_from: ContinueFrom | null;
  job?: string;
  progress?: number | null;
  message?: string;
};

export type Transcribe = {
  state: "queued" | "running" | "done" | "error" | "cancelled";
  error: string | null;
  job?: string;
  message?: string;
  log?: string[];
};

export type ProjectDetail = {
  id: string;
  title: string;
  created: string;
  audio: string | null;
  source: { origin: string; created: string } | null;
  transcribe: Transcribe | null;
  takes: Take[];
  busy: boolean;
};

export type ProjectSummary = {
  id: string;
  title: string;
  created: string;
  audio: string | null;
  has_source: boolean;
  transcribe: Transcribe | null;
  takes: number;
  favorites: number;
  busy: boolean;
};

// [onset 秒, 長さ 秒, pitch, velocity, 種類 (groups の番号)]
export type SourceNote = [number, number, number, number, number];
export type SourceView = {
  duration: number;
  groups: string[];
  notes: SourceNote[];
  beats: number[];
  downbeats: number[];
  chords: [number, string][];
};

// [onset 秒, 長さ 秒, pitch, velocity, 鳴り終わり 秒 (ペダルで延びた所まで)]
export type CoverNote = [number, number, number, number, number];
export type CoverView = { duration: number; notes: CoverNote[]; pedals: [number, number][] };

export type LiveStem = { stem: string; duration: number; pos: number; done: boolean };
// 今の段階の進み具合 (ステム分離の "分離し終えた塊 / 全体の塊")
export type LiveProgress = { done: number; total: number };
export type LiveTranscriptionData = {
  seq: number;
  // [id, stem, start 秒, end 秒, pitch, final (0/1)]
  events: [string, string, number, number, number, number][];
  stems: LiveStem[];
  stage: string | null;
  progress: LiveProgress | null;
  active: boolean;
};

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") message = body.detail;
    } catch {
      // 本文が JSON でなければ状態の文字列のまま
    }
    throw new ApiError(res.status, message);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

const p = (pid: string) => `/api/projects/${encodeURIComponent(pid)}`;

export const api = {
  config: () => request<Config>("/api/config"),
  unload: () => request<Config>("/api/engine/unload", { method: "POST" }),
  projects: () => request<ProjectSummary[]>("/api/projects"),
  project: (pid: string) => request<ProjectDetail>(p(pid)),
  createProject: (form: FormData) => request<ProjectDetail>("/api/projects", { method: "POST", body: form }),
  renameProject: (pid: string, title: string) => request<ProjectDetail>(p(pid), json("PATCH", { title })),
  deleteProject: (pid: string) => request<void>(p(pid), { method: "DELETE" }),
  transcribe: (pid: string) => request<ProjectDetail>(`${p(pid)}/transcribe`, { method: "POST" }),
  transcribeLive: (pid: string, since: number) =>
    request<LiveTranscriptionData>(`${p(pid)}/transcribe/live?since=${since}`),
  uploadSource: (pid: string, file: File) => {
    const form = new FormData();
    form.append("midi", file);
    return request<ProjectDetail>(`${p(pid)}/source`, { method: "PUT", body: form });
  },
  sourceView: (pid: string) => request<SourceView>(`${p(pid)}/source/view`),
  generate: (
    pid: string,
    body: { params: Params; model: string | null; count: number; seed: number | null; continue_from: ContinueFrom | null },
  ) => request<ProjectDetail>(`${p(pid)}/takes`, json("POST", body)),
  updateTake: (pid: string, tid: string, body: Partial<Pick<Take, "name" | "favorite" | "memo">>) =>
    request<Take>(`${p(pid)}/takes/${tid}`, json("PATCH", body)),
  deleteTake: (pid: string, tid: string) => request<void>(`${p(pid)}/takes/${tid}`, { method: "DELETE" }),
  takeView: (pid: string, tid: string) => request<CoverView>(`${p(pid)}/takes/${tid}/view`),
  cancelJob: (jobId: string) => request<unknown>(`/api/jobs/${jobId}`, { method: "DELETE" }),
};

export const urls = {
  audio: (pid: string) => `${p(pid)}/audio`,
  sourceMidi: (pid: string) => `${p(pid)}/source.mid`,
  takeMidi: (pid: string, tid: string) => `${p(pid)}/takes/${tid}/cover.mid`,
};
