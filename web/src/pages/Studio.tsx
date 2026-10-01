// 1 曲の作業画面: 左で設定して生成し、真ん中でテイクを聴き比べ、右で選んだテイクを見る
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, urls, type Config, type ContinueFrom, type ProjectDetail, type Take } from "../api";
import { CreatePanel, takeLabel, type Draft } from "../components/CreatePanel";
import { Icon } from "../components/Icon";
import { LiveTranscription } from "../components/LiveTranscription";
import { PianoRoll } from "../components/PianoRoll";
import { PlayerBar, usePlayerState } from "../components/PlayerBar";
import { TakeCard, TakeDetail } from "../components/Takes";
import { progressText, useT } from "../i18n";
import { formatTime } from "../params";
import { downloadCoverWav, getPlayer, type Player } from "../player";

// 「作る」欄の設定は曲をまたいで同じものを使う (別の曲へ移っても、変えた設定のまま続けられる)
const DRAFT_KEY = "cover-studio:draft";

// 携帯では 3 つの欄をタブで切り替える (styles.css の .tabs / .studio[data-tab] と対)
type Tab = "create" | "play" | "detail";
const isPhone = () => window.matchMedia("(max-width: 820px)").matches;

function loadDraft(pid: string, config: Config): Draft {
  const fallback: Draft = { params: { ...config.defaults }, model: null, count: 2, seedLocked: false, seed: null };
  try {
    // 以前は曲ごとに覚えていたので、共通のものがまだなければその曲のものを使う
    const raw = localStorage.getItem(DRAFT_KEY) ?? localStorage.getItem(`cover-studio:draft:${pid}`);
    const saved = JSON.parse(raw ?? "null");
    if (saved) {
      const draft: Draft = { ...fallback, ...saved, params: { ...config.defaults, ...saved.params } };
      // 覚えていたモデルがもうなければ (消した・別の環境) 既定のモデルにする
      if (draft.model && !config.models.some((m) => m.id === draft.model)) draft.model = null;
      return draft;
    }
  } catch {
    // 保存できない環境 (プライベートモードなど) では毎回既定値から
  }
  return fallback;
}

function saveDraft(draft: Draft) {
  try {
    localStorage.setItem(DRAFT_KEY, JSON.stringify(draft));
  } catch {
    // 覚えておけなくても使える
  }
}

export function Studio({ pid, config, navigate }: { pid: string; config: Config; navigate: (to: string) => void }) {
  const qc = useQueryClient();
  const t = useT();
  const player = usePlayerState(getPlayer());
  const project = useQuery({
    queryKey: ["project", pid],
    queryFn: () => api.project(pid),
    refetchInterval: (q) => (q.state.data?.busy ? 1000 : false),
  });
  const data = project.data;
  const sourceView = useQuery({
    queryKey: ["source-view", pid, data?.source?.created],
    queryFn: () => api.sourceView(pid),
    enabled: !!data?.source,
    staleTime: Infinity,
  });

  const [selected, setSelected] = useState<string | null>(null);
  const [filter, setFilter] = useState<"all" | "favorite">("all");
  const [showSource, setShowSource] = useState(true);
  const [tab, setTab] = useState<Tab>("play");
  // 携帯で巻物をたたむかどうか (広い画面では styles.css がボタンごと隠す)
  const [rollOpen, setRollOpen] = useState(true);
  const [continueFrom, setContinueFrom] = useState<ContinueFrom | null>(null);
  const [draft, setDraftState] = useState<Draft>(() => loadDraft(pid, config));
  const setDraft = (d: Draft) => {
    setDraftState(d);
    saveDraft(d);
  };
  const [error, setError] = useState<string | null>(null);

  const takes = data?.takes ?? [];
  const doneTakes = takes.filter((t) => t.state === "done");
  const shown = filter === "favorite" ? takes.filter((t) => t.favorite) : takes;
  const selectedTake = takes.find((t) => t.id === selected) ?? null;

  // まだ何も選んでいなければ、いちばん新しい完成したテイク
  useEffect(() => {
    if ((!selected || !takes.some((t) => t.id === selected)) && doneTakes.length) setSelected(doneTakes[0].id);
  }, [selected, takes, doneTakes]);

  const takeView = useQuery({
    queryKey: ["take-view", pid, selected],
    queryFn: () => api.takeView(pid, selected!),
    enabled: selectedTake?.state === "done",
    staleTime: Infinity,
  });

  useEffect(() => {
    if (takeView.data) player.setNotes(takeView.data.notes);
    else if (selectedTake && selectedTake.state !== "done") player.setNotes([]);
  }, [takeView.data, selectedTake?.state, player]);

  useEffect(() => {
    void player.setOriginal(data?.audio ? urls.audio(pid) : null);
  }, [pid, data?.audio, player]);

  useEffect(() => {
    player.setExtraDuration(sourceView.data?.duration ?? 0);
  }, [sourceView.data, player]);

  // 曲を離れたら止めて片付ける
  useEffect(
    () => () => {
      player.pause();
      player.setNotes([]);
      player.setLoop(null);
      player.seek(0);
      void player.setOriginal(null);
    },
    [pid, player],
  );

  const refresh = (detail?: ProjectDetail) => {
    if (detail) qc.setQueryData(["project", pid], detail);
    else void qc.invalidateQueries({ queryKey: ["project", pid] });
  };
  const run = async (fn: () => Promise<unknown>) => {
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const generate = useMutation({
    mutationFn: () =>
      api.generate(pid, {
        params: draft.params,
        model: draft.model,
        count: draft.count,
        seed: draft.seedLocked ? draft.seed : null,
        continue_from: continueFrom,
      }),
    onSuccess: (detail) => {
      refresh(detail);
      setContinueFrom(null);
      setError(null);
      // 携帯では、作り始めたら演奏の面へ移して進み具合を見せる
      if (isPhone()) setTab("play");
      void qc.invalidateQueries({ queryKey: ["config"] });
    },
    onError: (e) => setError(e instanceof Error ? e.message : String(e)),
  });

  const updateTake = (take: Take, body: Partial<Pick<Take, "name" | "favorite" | "memo">>) =>
    run(async () => {
      await api.updateTake(pid, take.id, body);
      refresh();
    });
  const deleteTake = (take: Take) =>
    run(async () => {
      if (!confirm(t.confirmDeleteTake(takeLabel(take, t)))) return;
      await api.deleteTake(pid, take.id);
      if (selected === take.id) setSelected(null);
      refresh();
    });
  const reuse = (take: Take) => {
    setDraft({ ...draft, params: { ...config.defaults, ...take.params }, model: take.model.id, seed: take.seed });
  };
  const playTake = (take: Take) => {
    if (selected === take.id) player.toggle();
    else {
      setSelected(take.id);
      if (!player.playing) player.play();
    }
  };

  // キー操作: Space 再生、← → 5 秒、↑ ↓ テイクを切り替え (位置はそのまま)、L ループ、F お気に入り
  const keys = useRef({ doneTakes, selectedTake });
  keys.current = { doneTakes, selectedTake };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) && (target as HTMLInputElement).type !== "range")
        return;
      const { doneTakes, selectedTake } = keys.current;
      if (e.code === "Space") {
        e.preventDefault();
        player.toggle();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        player.seek(player.position() - 5);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        player.seek(player.position() + 5);
      } else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        e.preventDefault();
        const i = doneTakes.findIndex((t) => t.id === selectedTake?.id);
        const next = doneTakes[Math.max(0, Math.min(doneTakes.length - 1, i + (e.key === "ArrowUp" ? -1 : 1)))];
        if (next) setSelected(next.id);
      } else if (e.key === "l" || e.key === "L") {
        player.toggleLoop();
      } else if ((e.key === "f" || e.key === "F") && selectedTake) {
        void updateTake(selectedTake, { favorite: !selectedTake.favorite });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [player, pid]);

  const continueTake = continueFrom ? (takes.find((t) => t.id === continueFrom.take) ?? null) : null;
  const disabledReason = !data?.source
    ? t.waitForSource
    : !config.models.length
      ? t.noModels
      : null;
  const takeById = useMemo(() => new Map(takes.map((t) => [t.id, t])), [takes]);

  if (project.isError) {
    return (
      <div className="empty-state">
        <p>{(project.error as Error).message}</p>
        <button className="btn" onClick={() => navigate("/")}>
          {t.songs}
        </button>
      </div>
    );
  }
  if (!data) return <div className="empty-state">{t.loading}</div>;

  return (
    <>
      {/* 携帯だけに出る切り替え (広い画面では styles.css が .tabs を隠す) */}
      <nav className="tabs" role="tablist" aria-label={t.viewSwitch}>
        {(
          [
            ["create", t.tabCreate],
            ["play", t.tabPlay],
            ["detail", t.tabDetail],
          ] as [Tab, string][]
        ).map(([id, label]) => (
          <button
            key={id}
            role="tab"
            aria-selected={tab === id}
            className={tab === id ? "on" : ""}
            onClick={() => setTab(id)}
          >
            {label}
            {id === "play" && takes.length > 0 && <span className="count">{takes.length}</span>}
          </button>
        ))}
      </nav>
      <div className="studio" data-tab={tab}>
        <CreatePanel
          config={config}
          draft={draft}
          onChange={setDraft}
          continueFrom={continueFrom}
          continueTake={continueTake}
          onClearContinue={() => setContinueFrom(null)}
          onGenerate={() => generate.mutate()}
          generating={generate.isPending}
          disabled={disabledReason}
        />

        <section className="center">
          <div className="song-head">
            <div className="song-title-block">
              <h1>{data.title}</h1>
              <div className="muted small">
                {data.audio ?? t.noAudio} · {t.sourceMidi}:{" "}
                {data.source ? (data.source.origin === "tsumugi" ? t.sourceByTsumugi : t.sourceUploaded) : t.sourceNotYet}
              </div>
            </div>
            <div className="song-actions">
              {data.source && (
                <a className="btn small" href={urls.sourceMidi(pid)} download>
                  <Icon name="download" size={15} /> {t.sourceMidi}
                </a>
              )}
              <SourceUpload pid={pid} onDone={refresh} onError={setError} />
            </div>
          </div>

          {error && (
            <div className="alert" onClick={() => setError(null)}>
              {error}
            </div>
          )}

          {!data.source ? (
            <TranscribePanel project={data} pid={pid} player={player} onChange={refresh} onError={setError} />
          ) : (
            <div className={`roll-card ${rollOpen ? "" : "collapsed"}`}>
              <div className="roll-head">
                <span className="roll-title">
                  <span className="roll-title-text">
                    {selectedTake ? takeLabel(selectedTake, t) : t.sourceMidiTitle}
                  </span>
                  <span className="spinner small" style={{ visibility: takeView.isFetching ? "visible" : "hidden" }} />
                </span>
                {/* 携帯だけに出る巻物の開閉 (広い画面では styles.css が隠す) */}
                <button
                  className={`roll-toggle ${rollOpen ? "open" : ""}`}
                  onClick={() => setRollOpen(!rollOpen)}
                  aria-expanded={rollOpen}
                  title={rollOpen ? t.rollCollapse : t.rollExpand}
                >
                  <Icon name="chevron" size={14} />
                  {rollOpen ? t.rollCollapse : t.rollExpand}
                </button>
                <div className="legend">
                  <label className="check">
                    <input type="checkbox" checked={showSource} onChange={(e) => setShowSource(e.target.checked)} />
                    {t.overlaySource}
                  </label>
                  {showSource && (
                    <span className="legend-items">
                      {["--src-melody", "--src-bass", "--src-keys", "--src-guitar", "--src-other"].map((c, i) => (
                        <span key={c}>
                          <i style={{ background: `var(${c})` }} />
                          {t.legend[i]}
                        </span>
                      ))}
                    </span>
                  )}
                </div>
              </div>
              <PianoRoll
                player={player}
                cover={selectedTake?.state === "done" ? (takeView.data ?? null) : null}
                source={sourceView.data ?? null}
                showSource={showSource}
                continueAt={continueFrom?.seconds ?? null}
              />
              <div className="roll-hint muted small">
                <span className="roll-hint-text">{t.rollHint}</span>
                {player.loop && (
                  <>
                    {" · "}
                    <button className="link" onClick={() => player.setLoop(null)}>
                      {t.clearLoop(formatTime(player.loop[0]), formatTime(player.loop[1]))}
                    </button>
                  </>
                )}
              </div>
            </div>
          )}

          {/* 巻物 (ピアノロール) は固定したまま、この下のテイク一覧だけを独立スクロールさせる */}
          <div className="takes-scroll">
            <div className="takes-head">
              <h2>{t.takes}</h2>
              <div className="segmented small">
                <button className={filter === "all" ? "on" : ""} onClick={() => setFilter("all")}>
                  {t.all} {takes.length}
                </button>
                <button className={filter === "favorite" ? "on" : ""} onClick={() => setFilter("favorite")}>
                  ★ {takes.filter((x) => x.favorite).length}
                </button>
              </div>
            </div>
            <div className="takes">
              {shown.length === 0 && (
                <div className="empty-takes muted">
                  {data.source
                    ? filter === "favorite"
                      ? t.noFavorites
                      : t.emptyTakes
                    : t.waitTranscribe}
                </div>
              )}
              {shown.map((take) => {
                const cf = take.continue_from;
                const base = cf ? takeById.get(cf.take) : undefined;
                return (
                  <TakeCard
                    key={take.id}
                    take={take}
                    defaults={config.defaults}
                    selected={take.id === selected}
                    playing={take.id === selected && player.playing}
                    continueLabel={cf ? t.continueChip(base ? takeLabel(base, t) : cf.take, formatTime(cf.seconds)) : null}
                    onSelect={() => {
                      setSelected(take.id);
                      // 携帯では選んだら詳細へ (広い画面ではタブ自体が出ないので何も変わらない)
                      if (isPhone()) setTab("detail");
                    }}
                    onPlay={() => playTake(take)}
                    onFavorite={() => updateTake(take, { favorite: !take.favorite })}
                    onReuse={() => reuse(take)}
                    onCancel={() => take.job && run(async () => (await api.cancelJob(take.job!), refresh()))}
                    onDelete={() => deleteTake(take)}
                  />
                );
              })}
            </div>
          </div>
        </section>

        <aside className="right">
          {selectedTake ? (
            <TakeDetail
              take={selectedTake}
              config={config}
              position={() => (player.loopOn && player.loop ? player.loop[0] : player.position())}
              midiUrl={urls.takeMidi(pid, selectedTake.id)}
              onRename={(name) => updateTake(selectedTake, { name })}
              onMemo={(memo) => updateTake(selectedTake, { memo })}
              onFavorite={() => updateTake(selectedTake, { favorite: !selectedTake.favorite })}
              onReuse={() => reuse(selectedTake)}
              onContinue={(seconds) => {
                setContinueFrom({ take: selectedTake.id, seconds: Math.round(seconds * 100) / 100 });
                reuse(selectedTake);
              }}
              onWav={() =>
                run(async () => {
                  const view = takeView.data ?? (await api.takeView(pid, selectedTake.id));
                  await downloadCoverWav(view.notes, `${data.title}_${takeLabel(selectedTake, t)}.wav`);
                })
              }
              onDelete={() => deleteTake(selectedTake)}
            />
          ) : (
            <div className="detail empty muted">{t.selectTakeHint}</div>
          )}
        </aside>
      </div>
      <PlayerBar
        player={player}
        title={selectedTake ? `${takeLabel(selectedTake, t)} — ${data.title}` : data.title}
        subtitle={
          player.originalLoading
            ? t.loadingAudio
            : selectedTake
              ? `seed ${selectedTake.seed}`
              : data.audio
                ? t.original
                : t.chooseTake
        }
      />
    </>
  );
}

function SourceUpload({
  pid,
  onDone,
  onError,
}: {
  pid: string;
  onDone: (d: ProjectDetail) => void;
  onError: (m: string) => void;
}) {
  const t = useT();
  const input = useRef<HTMLInputElement>(null);
  return (
    <>
      <button className="btn small" onClick={() => input.current?.click()} title={t.replaceMidiHelp}>
        <Icon name="layers" size={15} /> {t.replaceMidi}
      </button>
      <input
        ref={input}
        type="file"
        accept=".mid,.midi"
        hidden
        onChange={async (e) => {
          const file = e.target.files?.[0];
          e.target.value = "";
          if (!file) return;
          try {
            onDone(await api.uploadSource(pid, file));
          } catch (err) {
            onError(err instanceof Error ? err.message : String(err));
          }
        }}
      />
    </>
  );
}

function TranscribePanel({
  project,
  pid,
  player,
  onChange,
  onError,
}: {
  project: ProjectDetail;
  pid: string;
  player: Player;
  onChange: (d?: ProjectDetail) => void;
  onError: (m: string) => void;
}) {
  const t = useT();
  const tr = project.transcribe;
  const active = tr?.state === "queued" || tr?.state === "running";
  const logRef = useRef<HTMLPreElement>(null);
  useEffect(() => {
    logRef.current?.scrollTo(0, logRef.current.scrollHeight);
  }, [tr?.log?.length]);
  return (
    <div className="transcribe">
      <div className="transcribe-head">
        {active ? <span className="spinner" /> : <Icon name="music" size={22} />}
        <div>
          <b>
            {active
              ? tr?.state === "queued"
                ? t.transcribeQueued
                : t.transcribeRunning
              : tr?.state === "error"
                ? t.transcribeFailed
                : tr?.state === "cancelled"
                  ? t.transcribeCancelled
                  : t.transcribeNone}
          </b>
          <div className="muted small">
            {active
              ? t.transcribeSteps
              : (tr?.error ?? t.transcribeHint)}
          </div>
        </div>
        <div className="grow" />
        {active && tr?.job ? (
          <button className="btn" onClick={async () => (await api.cancelJob(tr.job!), onChange())}>
            <Icon name="stop" size={14} /> {t.stop}
          </button>
        ) : (
          project.audio && (
            <button
              className="btn primary"
              onClick={async () => {
                try {
                  onChange(await api.transcribe(pid));
                } catch (e) {
                  onError(e instanceof Error ? e.message : String(e));
                }
              }}
            >
              {t.transcribe}
            </button>
          )
        )}
      </div>
      {tr?.message && active && !tr.message.startsWith("@@") && <div className="transcribe-msg mono small">{progressText(t, tr.message)}</div>}
      {tr?.job && <LiveTranscription pid={pid} player={player} active={active} />}
      {tr?.log && tr.log.length > 0 && (
        <details className="log-box" open={tr.state === "error"}>
          <summary className="muted small">{t.showLog}</summary>
          <pre className="log" ref={logRef}>
            {tr.log.filter((line) => !line.startsWith("@@")).join("\n")}
          </pre>
        </details>
      )}
    </div>
  );
}
