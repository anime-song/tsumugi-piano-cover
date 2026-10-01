// 画面の文言 (日本語 / 英語)。選んだ言語はブラウザに覚えておく。
// サーバが返すエラーの文や tsumugi のログは訳さない (そのまま出す)
import { useSyncExternalStore } from "react";
import type { NumberKey } from "./params";

export type Lang = "ja" | "en";

type ParamText = { label: string; short: string; help: string };

const ja = {
  lang: "ja-JP",
  // 共通
  loading: "読み込み中…",
  sec: "秒",
  delete: "削除",
  remove: "外す",
  cancel: "やめる",
  stop: "止める",
  close: "閉じる",
  switchSong: "曲を切り替える",
  searchSongs: "曲を探す",
  allSongs: "すべての曲を一覧で見る",
  noMatch: "見つかりません",
  noServer: "サーバにつながりません。python -m cover_studio を起動してください",
  // 上の帯
  songs: "曲の一覧",
  noModelLoaded: "モデル未読み込み",
  freeVram: "VRAM を空ける",
  freeVramHelp: "カバーモデルを手放して VRAM を空ける (次の生成で読み直す)",
  device: "デバイス",
  language: "English",
  languageHelp: "Switch to English",
  // 曲の一覧
  newSong: "新しい曲",
  dropHelp: "音源 (mp3 / wav / flac …) をドロップ。採譜済みの MIDI も一緒に入れると採譜を飛ばします",
  noTsumugi: " (tsumugi がないので MIDI が必要です)",
  unsupportedFile: (name: string) => `対応していないファイルです: ${name}`,
  songTitle: "曲名",
  uploading: "アップロード中…",
  create: "作る",
  createAndTranscribe: "作って採譜する",
  confirmDeleteSong: (title: string, takes: number) => `「${title}」とそのテイク ${takes} 本を削除しますか？`,
  noSongs: "まだ曲がありません",
  statusQueued: "採譜の順番待ち",
  statusRunning: "採譜中…",
  statusFailed: "採譜に失敗",
  statusNoSource: "原曲の MIDI なし",
  takesCount: (n: number) => `テイク ${n}`,
  generatingNow: "生成中…",
  // 作る欄
  createTitle: "作る",
  resetAll: "すべて既定に戻す",
  resetTo: (v: string) => `既定値 (${v}) に戻す`,
  continueUntil: (take: string, time: string) => `${take} の ${time} まで`,
  continueRest: "をそのまま使い、続きだけを作り直します",
  model: "モデル",
  modelKinds: { export: "書き出した重み", checkpoint: "学習のチェックポイント", hub: "Hugging Face" },
  loaded: " ● 読み込み済み",
  length: "長さ",
  lengthHelp: "短くすると早く聴けます。良い設定が見つかったら全曲で",
  fullSong: "全曲",
  arrangementSwitch: "オフにすると編曲の性質を指定しない",
  performer: "演奏者",
  performerNumber: "番号",
  performerHelp: (max?: number) => `0 で指定なし${max ? ` (1〜${max})` : ""}`,
  countHelp: "同じ設定で一度に作る本数",
  seedHelp: "固定すると同じ設定で同じ結果になります",
  random: "ランダム",
  generate: "生成",
  regenerateRest: "続きを作り直す",
  waitForSource: "原曲の MIDI ができるまで待ってください",
  noModels: "使えるモデルがありません",
  groups: {
    source: { title: "原曲", note: "" },
    expression: { title: "表現", note: "Planner が原曲から予測した曲線を何倍にするか (0 で指定なし)" },
    arrangement: { title: "編曲", note: "Planner の予測を全カバーでの標準偏差の単位でずらす (0 で予測のまま)" },
    sampling: { title: "サンプリング", note: "" },
  },
  params: {
    source_cfg: {
      label: "原曲への忠実さ",
      short: "忠実さ",
      help: "1 より大きいほど原曲 (メロディ・コード) に沿わせる。上げすぎると硬くなる",
    },
    onset_bias: {
      label: "原曲のリズムに寄せる",
      short: "リズム寄せ",
      help: "出力の発音を原曲の発音 (全楽器の音と拍) に寄せる強さ。0 で使わない、4 前後がよい",
    },
    dynamics: {
      label: "強弱のメリハリ",
      short: "強弱",
      help: "強弱の曲線の山と谷を何倍にするか。1 で学習データと同じくらい、大きいほどサビが盛り上がる",
    },
    density: {
      label: "音数の起伏",
      short: "音数",
      help: "音の多さの曲線の倍率。大きいほど静かな所と賑やかな所の差が付く",
    },
    fill: {
      label: "合いの手・オブリ",
      short: "合いの手",
      help: "メロディの隙間を埋める音の量。+0.5 前後で合いの手の多い演奏者くらい",
    },
    above: {
      label: "メロディの上に重ねる",
      short: "上に重ねる",
      help: "メロディより高い音を重ねる割合 (メロディが常に最高音ではない弾き方)",
    },
    span: { label: "音域の広さ", short: "音域", help: "低音から高音までの広さ" },
    temperature: { label: "temperature", short: "temp", help: "大きいほど大胆、小さいほど無難になる" },
    top_p: { label: "top-p", short: "top-p", help: "確率の高い候補から合計 p までだけを使う。小さいほど無難" },
    channel_cfg: {
      label: "演奏者らしさ",
      short: "演奏者らしさ",
      help: "1 より大きいほど選んだ演奏者の弾き方を強める (演奏者を選んだときだけ効く)",
    },
    onset_bias_width: { label: "寄せる範囲", short: "寄せる範囲", help: "原曲の発音に寄せる範囲 (秒)" },
  } satisfies Record<NumberKey, ParamText>,
  chipPerformer: (n: number) => `演奏者 #${n}`,
  chipNoArrangement: "編曲の指定なし",
  chipSeconds: (n: number) => `${n} 秒`,
  // 作業画面
  noAudio: "音源なし",
  sourceMidi: "原曲 MIDI",
  sourceByTsumugi: "tsumugi で採譜",
  sourceUploaded: "アップロード",
  sourceNotYet: "まだ",
  replaceMidi: "MIDI を差し替え",
  replaceMidiHelp: "手元の MIDI を原曲として使う",
  sourceMidiTitle: "原曲の MIDI",
  overlaySource: "原曲の MIDI を重ねる",
  legend: ["メロディ", "ベース", "鍵盤", "ギター", "その他"],
  rollHint: "クリックで移動 · ドラッグでループ区間 · ホイールで横移動 (Ctrl で拡大) · ↑↓ でテイクを切り替えても位置はそのまま",
  clearLoop: (a: string, b: string) => `ループ ${a}–${b} を消す`,
  view: "表示",
  viewSeconds: "表示する秒数",
  takes: "テイク",
  all: "すべて",
  noFavorites: "お気に入りはまだありません",
  emptyTakes: "左の「生成」でテイクを作ります。設定を変えながら何度でも作り直せます",
  waitTranscribe: "採譜が終わるとテイクを作れます",
  confirmDeleteTake: (name: string) => `${name} を削除しますか？`,
  selectTakeHint: "テイクを選ぶと、設定とメモがここに出ます",
  loadingAudio: "音源を読み込み中…",
  original: "原曲",
  chooseTake: "テイクを選んでください",
  // 採譜
  transcribeQueued: "採譜の順番待ち",
  transcribeRunning: "tsumugi で採譜しています",
  transcribeFailed: "採譜に失敗しました",
  transcribeCancelled: "採譜を止めました",
  transcribeNone: "原曲の MIDI がありません",
  transcribeSteps: "ステム分離 → 採譜 → 楽器の判定 → ベロシティ → ビート・コード。1 曲数分かかります (一度だけ)",
  transcribeHint: "音源から採譜するか、採譜済みの MIDI を入れてください",
  transcribe: "採譜する",
  // テイク
  take: (n: number) => `Take ${n}`,
  batchHelp: "同じ設定で一度に作った組",
  play: "再生",
  defaultSettings: "既定の設定",
  queued: "待ち",
  generating: "生成中",
  stopping: "止めています",
  preparingModel: "モデルを準備しています",
  startingTsumugi: "tsumugi を起動しています",
  stopped: "止めました",
  notes: (n: number | null) => `${n} 音`,
  favorite: "お気に入り (F)",
  reuseHelp: "この設定を「作る」に戻す",
  continueChip: (take: string, time: string) => `${take} の ${time} から`,
  reuse: "設定を使う",
  continueHere: "ここから作り直す",
  continueHereHelp: "再生位置 (ループ中はループの始め) より前をそのまま使い、続きを作り直す",
  wav: "WAV",
  wavRendering: "書き出し中…",
  wavHelp: "ピアノの音源で WAV に書き出す",
  seed: "seed",
  fullLength: "全曲",
  notSpecified: "指定なし",
  defaultValue: (v: string) => `(既定 ${v})`,
  memo: "メモ",
  memoPlaceholder: "気づいたこと (サビの合いの手が良い、など)",
  // 再生バー
  pianoLoading: (p: number) => `ピアノ音源を読み込み中 ${p}%`,
  pianoFallback: "ピアノ音源を読めないので簡易音で再生",
  back5: "5 秒戻る (←)",
  forward5: "5 秒進む (→)",
  playPause: "再生 / 停止 (Space)",
  loopHelp: "ループ (L)。ピアノロールをドラッグして区間を選ぶ",
  cover: "カバー",
  mixHelp: "原曲とカバーの配分 (ダブルクリックで半々)",
  mixLabel: "原曲とカバーの音量の配分",
};

type Dict = typeof ja;

const en: Dict = {
  lang: "en-US",
  loading: "Loading…",
  sec: "s",
  delete: "Delete",
  remove: "Remove",
  cancel: "Cancel",
  stop: "Stop",
  close: "Close",
  switchSong: "Switch song",
  searchSongs: "Search songs",
  allSongs: "View all songs",
  noMatch: "No matches",
  noServer: "Cannot reach the server. Start it with python -m cover_studio",
  songs: "Songs",
  noModelLoaded: "No model loaded",
  freeVram: "Free VRAM",
  freeVramHelp: "Unload the cover model to free VRAM (it is reloaded on the next generation)",
  device: "Device",
  language: "日本語",
  languageHelp: "日本語に切り替える",
  newSong: "New song",
  dropHelp: "Drop an audio file (mp3 / wav / flac …). Add a transcribed MIDI too to skip transcription",
  noTsumugi: " (tsumugi is not available, so a MIDI file is required)",
  unsupportedFile: (name) => `Unsupported file: ${name}`,
  songTitle: "Title",
  uploading: "Uploading…",
  create: "Create",
  createAndTranscribe: "Create and transcribe",
  confirmDeleteSong: (title, takes) => `Delete "${title}" and its ${takes} take(s)?`,
  noSongs: "No songs yet",
  statusQueued: "Waiting to transcribe",
  statusRunning: "Transcribing…",
  statusFailed: "Transcription failed",
  statusNoSource: "No source MIDI",
  takesCount: (n) => `${n} take${n === 1 ? "" : "s"}`,
  generatingNow: "Generating…",
  createTitle: "Create",
  resetAll: "Reset all",
  resetTo: (v) => `Reset to default (${v})`,
  continueUntil: (take, time) => `Keep ${take} up to ${time}`,
  continueRest: " and regenerate only the rest",
  model: "Model",
  modelKinds: { export: "Exported weights", checkpoint: "Training checkpoints", hub: "Hugging Face" },
  loaded: " ● loaded",
  length: "Length",
  lengthHelp: "Short previews are faster. Render the full song once you like the settings",
  fullSong: "Full",
  arrangementSwitch: "Turn off to leave the arrangement unspecified",
  performer: "Performer",
  performerNumber: "Number",
  performerHelp: (max) => `0 = unspecified${max ? ` (1–${max})` : ""}`,
  countHelp: "How many takes to make with the same settings",
  seedHelp: "A fixed seed gives the same result for the same settings",
  random: "random",
  generate: "Generate",
  regenerateRest: "Regenerate the rest",
  waitForSource: "Wait until the source MIDI is ready",
  noModels: "No models available",
  groups: {
    source: { title: "Source", note: "" },
    expression: { title: "Expression", note: "Scale the curves the Planner predicts from the source (0 = unspecified)" },
    arrangement: {
      title: "Arrangement",
      note: "Shift the Planner's prediction, in standard deviations across all covers (0 = as predicted)",
    },
    sampling: { title: "Sampling", note: "" },
  },
  params: {
    source_cfg: {
      label: "Faithfulness to source",
      short: "faithful",
      help: "Above 1 follows the source (melody, chords) more closely. Too high sounds stiff",
    },
    onset_bias: {
      label: "Snap to source rhythm",
      short: "rhythm snap",
      help: "Pull note onsets toward the source onsets (all instruments and beats). 0 = off, around 4 works well",
    },
    dynamics: {
      label: "Dynamics contrast",
      short: "dynamics",
      help: "Scale the peaks and valleys of the loudness curve. 1 ≈ training data; higher makes choruses build more",
    },
    density: {
      label: "Density contrast",
      short: "density",
      help: "Scale the note density curve. Higher widens the gap between quiet and busy parts",
    },
    fill: {
      label: "Fills & obbligato",
      short: "fills",
      help: "How much to fill the gaps in the melody. About +0.5 matches performers who play many fills",
    },
    above: {
      label: "Notes above melody",
      short: "above",
      help: "How often notes are stacked above the melody (the melody is not always the top note)",
    },
    span: { label: "Range", short: "range", help: "How wide the range from low to high notes is" },
    temperature: { label: "temperature", short: "temp", help: "Higher is bolder, lower is safer" },
    top_p: {
      label: "top-p",
      short: "top-p",
      help: "Sample only from the most likely candidates up to total probability p. Lower is safer",
    },
    channel_cfg: {
      label: "Performer strength",
      short: "performer",
      help: "Above 1 emphasizes the selected performer's style (only when a performer is selected)",
    },
    onset_bias_width: { label: "Snap width", short: "snap width", help: "Window for snapping to source onsets (s)" },
  },
  chipPerformer: (n) => `performer #${n}`,
  chipNoArrangement: "no arrangement",
  chipSeconds: (n) => `${n}s`,
  noAudio: "no audio",
  sourceMidi: "Source MIDI",
  sourceByTsumugi: "transcribed with tsumugi",
  sourceUploaded: "uploaded",
  sourceNotYet: "not yet",
  replaceMidi: "Replace MIDI",
  replaceMidiHelp: "Use your own MIDI file as the source",
  sourceMidiTitle: "Source MIDI",
  overlaySource: "Overlay source MIDI",
  legend: ["Melody", "Bass", "Keys", "Guitar", "Other"],
  rollHint: "Click to seek · drag to set a loop · scroll to pan (Ctrl to zoom) · ↑↓ switches takes and keeps the position",
  clearLoop: (a, b) => `Clear loop ${a}–${b}`,
  view: "View",
  viewSeconds: "Seconds shown",
  takes: "Takes",
  all: "All",
  noFavorites: "No favorites yet",
  emptyTakes: "Press Generate on the left to make a take. Change the settings and regenerate as often as you like",
  waitTranscribe: "You can make takes once transcription is done",
  confirmDeleteTake: (name) => `Delete ${name}?`,
  selectTakeHint: "Select a take to see its settings and notes here",
  loadingAudio: "Loading audio…",
  original: "Original",
  chooseTake: "Select a take",
  transcribeQueued: "Waiting to transcribe",
  transcribeRunning: "Transcribing with tsumugi",
  transcribeFailed: "Transcription failed",
  transcribeCancelled: "Transcription stopped",
  transcribeNone: "No source MIDI",
  transcribeSteps:
    "Stem separation → transcription → instrument labels → velocity → beats & chords. Takes a few minutes per song (only once)",
  transcribeHint: "Transcribe the audio, or add a MIDI file you already have",
  transcribe: "Transcribe",
  take: (n) => `Take ${n}`,
  batchHelp: "Made together with the same settings",
  play: "Play",
  defaultSettings: "default settings",
  queued: "Queued",
  generating: "Generating",
  stopping: "Stopping",
  preparingModel: "Preparing the model",
  startingTsumugi: "Starting tsumugi",
  stopped: "Stopped",
  notes: (n) => `${n} notes`,
  favorite: "Favorite (F)",
  reuseHelp: "Load these settings into Create",
  continueChip: (take, time) => `from ${take} at ${time}`,
  reuse: "Reuse settings",
  continueHere: "Regenerate from here",
  continueHereHelp: "Keep everything before the playhead (or the loop start) and regenerate the rest",
  wav: "WAV",
  wavRendering: "Rendering…",
  wavHelp: "Render to WAV with the piano samples",
  seed: "seed",
  fullLength: "Full",
  notSpecified: "unspecified",
  defaultValue: (v) => `(default ${v})`,
  memo: "Notes",
  memoPlaceholder: "What you noticed (e.g. nice fills in the chorus)",
  pianoLoading: (p) => `Loading piano samples ${p}%`,
  pianoFallback: "Piano samples unavailable; using a simple synth",
  back5: "Back 5 s (←)",
  forward5: "Forward 5 s (→)",
  playPause: "Play / pause (Space)",
  loopHelp: "Loop (L). Drag on the piano roll to select a range",
  cover: "Cover",
  mixHelp: "Balance between original and cover (double-click to center)",
  mixLabel: "Volume balance between the original and the cover",
};

const DICTS: Record<Lang, Dict> = { ja, en };

function initialLang(): Lang {
  try {
    const saved = localStorage.getItem("cover-studio:lang");
    if (saved === "ja" || saved === "en") return saved;
  } catch {
    // 覚えておけない環境ではブラウザの言語から
  }
  return navigator.language.toLowerCase().startsWith("ja") ? "ja" : "en";
}

let current: Lang = initialLang();
const listeners = new Set<() => void>();
document.documentElement.lang = current;

export function setLang(lang: Lang) {
  current = lang;
  document.documentElement.lang = lang;
  try {
    localStorage.setItem("cover-studio:lang", lang);
  } catch {
    // 覚えておけなくても切り替えはできる
  }
  for (const l of listeners) l();
}

export function getLang(): Lang {
  return current;
}

const subscribe = (l: () => void) => {
  listeners.add(l);
  return () => listeners.delete(l);
};

/** 今の言語の文言 (言語を切り替えると描き直す) */
export function useT(): Dict {
  const lang = useSyncExternalStore(subscribe, getLang);
  return DICTS[lang];
}

export function useLang(): [Lang, (lang: Lang) => void] {
  return [useSyncExternalStore(subscribe, getLang), setLang];
}

/** サーバが返す進み具合の文 ("loading"・"starting"・"stopping"・"12/60") を今の言語にする。それ以外 (tsumugi のログなど) はそのまま */
export function progressText(t: Dict, message: string | undefined): string | null {
  if (!message) return null;
  if (message === "loading") return t.preparingModel;
  if (message === "stopping") return t.stopping;
  if (message === "starting") return t.startingTsumugi;
  const m = message.match(/^(\d+)\/(\d+)$/);
  if (m) return `${m[1]} / ${m[2]} ${t.sec}`;
  return message;
}
