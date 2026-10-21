// テイクの楽譜 (MusicXML) を画面いっぱいに出す。描くのはブラウザ内の Verovio (WebAssembly) なので、
// MuseScore などの楽譜ソフトは要らない。印刷は A4 で描き直して別の窓からブラウザの印刷 (PDF にもできる) を使う
import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { api, urls } from "../api";
import { useT } from "../i18n";
import { Icon } from "./Icon";

type Toolkit = import("verovio/esm").VerovioToolkit;

let toolkitPromise: Promise<Toolkit> | null = null;

// 7 MB ほどあるので、楽譜を初めて開いたときにだけ読む
function loadToolkit(): Promise<Toolkit> {
  if (!toolkitPromise) {
    toolkitPromise = (async () => {
      const [{ default: createModule }, { VerovioToolkit }] = await Promise.all([
        import("verovio/wasm"),
        import("verovio/esm"),
      ]);
      return new VerovioToolkit(await createModule());
    })();
    toolkitPromise.catch(() => {
      toolkitPromise = null; // 読めなかったら次に開いたときに読み直す
    });
  }
  return toolkitPromise;
}

const COMMON = { svgViewBox: true, footer: "none", breaks: "auto", font: "Leipzig" };
// 画面: 幅に合わせて段を折り返し、ページの高さは中身に合わせる。scale は Verovio の拡大率 (%)
const ZOOMS = [25, 30, 35, 40, 45, 50, 60];
// 印刷: A4 (Verovio の単位は 0.1 mm)
const PRINT = { pageWidth: 2100, pageHeight: 2970, scale: 40, adjustPageHeight: false };

// Verovio はペダルの線 (line="yes") の始まりと終わりを音符に結び付けるので、同じ音を伸ばしている間に踏み替えると
// 線ごと捨てる。画面と印刷では Ped. と * の記号で描く (ダウンロードする MusicXML は線のまま)
const pedalMarks = (xml: string) => xml.replace(/(<pedal type="[a-z]+") line="yes"/g, '$1 line="no"');

function render(tk: Toolkit, xml: string, options: Record<string, unknown>): string[] {
  tk.setOptions({ ...COMMON, ...options });
  if (!tk.loadData(pedalMarks(xml))) throw new Error(tk.getLog() || "loadData");
  return Array.from({ length: tk.getPageCount() }, (_, i) => tk.renderToSVG(i + 1));
}

export function ScoreViewer({
  pid,
  tid,
  version,
  title,
  onClose,
}: {
  pid: string;
  tid: string;
  version: string | null | undefined;
  title: string;
  onClose: () => void;
}) {
  const t = useT();
  const xml = useQuery({
    queryKey: ["score-xml", pid, tid, version],
    queryFn: () => api.scoreXml(pid, tid),
    staleTime: Infinity,
  });
  const box = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(3);
  const [width, setWidth] = useState(0);
  const [pages, setPages] = useState<string[] | null>(null);
  const [redraw, setRedraw] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // 幅が変わったら段の折り返しを描き直す
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const observer = new ResizeObserver(() => setWidth(Math.round(el.clientWidth / 40) * 40));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!xml.data || !width) return;
    let cancelled = false;
    setPages(null);
    setError(null);
    loadToolkit()
      .then((tk) => {
        if (cancelled) return;
        const scale = ZOOMS[zoom];
        // 画面の幅 (px) がそのまま 1 段の幅になるように、ページの幅を Verovio の単位に直す
        const pageWidth = Math.max(1000, Math.round(((width - 32) * 100) / scale));
        setPages(render(tk, xml.data, { scale, pageWidth, pageHeight: 60000, adjustPageHeight: true }));
      })
      .catch((e) => !cancelled && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
    };
  }, [xml.data, width, zoom, redraw]);

  const print = async () => {
    if (!xml.data) return;
    // 窓は先に開く (await の後だとポップアップとして止められる)
    const w = window.open("", "_blank");
    if (!w) return;
    try {
      const tk = await loadToolkit();
      const printed = render(tk, xml.data, PRINT);
      const doc = w.document;
      doc.title = title;
      const style = doc.createElement("style");
      style.textContent =
        "@page{size:A4;margin:0}body{margin:0;background:#fff}" +
        ".page{width:210mm;height:297mm;break-after:page;overflow:hidden}.page svg{width:100%;height:100%}";
      doc.head.appendChild(style);
      for (const svg of printed) {
        const page = doc.createElement("div");
        page.className = "page";
        page.innerHTML = svg;
        doc.body.appendChild(page);
      }
      w.setTimeout(() => w.print(), 300);
    } catch (e) {
      w.close();
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRedraw((n) => n + 1); // 印刷用の設定に変えたので、画面用の設定で描き直す
    }
  };

  return (
    <div className="score-overlay" role="dialog" aria-label={t.score}>
      <div className="score-bar">
        <Icon name="score" size={18} />
        <span className="score-title">{title}</span>
        <div className="grow" />
        <button className="icon-btn" onClick={() => setZoom(Math.max(0, zoom - 1))} disabled={zoom === 0} title={t.zoomOut}>
          <Icon name="zoomOut" size={17} />
        </button>
        <button
          className="icon-btn"
          onClick={() => setZoom(Math.min(ZOOMS.length - 1, zoom + 1))}
          disabled={zoom === ZOOMS.length - 1}
          title={t.zoomIn}
        >
          <Icon name="zoomIn" size={17} />
        </button>
        <button className="btn small" onClick={print} disabled={!pages} title={t.printHelp}>
          <Icon name="print" size={15} /> {t.print}
        </button>
        <a className="btn small" href={urls.takeScore(pid, tid)} download>
          <Icon name="download" size={15} /> MusicXML
        </a>
        <button className="icon-btn" onClick={onClose} title={t.close}>
          <Icon name="close" size={18} />
        </button>
      </div>
      <div className="score-pages" ref={box}>
        {error || xml.isError ? (
          <div className="score-status error">
            {t.scoreRenderFailed}: {error ?? (xml.error as Error).message}
          </div>
        ) : !pages ? (
          <div className="score-status muted">
            <span className="spinner" /> {t.scoreLoading}
          </div>
        ) : (
          pages.map((svg, i) => <div className="score-page" key={i} dangerouslySetInnerHTML={{ __html: svg }} />)
        )}
      </div>
    </div>
  );
}
