import { QueryClient, QueryClientProvider, useQuery, useQueryClient } from "@tanstack/react-query";
import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { API_VERSION, api } from "./api";
import { useLang, useT } from "./i18n";
import { Icon } from "./components/Icon";
import { NewSongDialog } from "./components/NewSong";
import { SongSwitcher } from "./components/SongSwitcher";
import { shortModel } from "./components/Takes";
import { Library } from "./pages/Library";
import { Studio } from "./pages/Studio";
import "./styles.css";

function usePath(): [string, (to: string) => void] {
  const [path, setPath] = useState(window.location.pathname);
  useEffect(() => {
    const onPop = () => setPath(window.location.pathname);
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);
  const navigate = (to: string) => {
    window.history.pushState(null, "", to);
    setPath(to);
  };
  return [path, navigate];
}

function App() {
  const [path, navigate] = usePath();
  const qc = useQueryClient();
  const t = useT();
  const [lang, setLang] = useLang();
  const [newSong, setNewSong] = useState(false);
  const config = useQuery({ queryKey: ["config"], queryFn: api.config, refetchInterval: 5000 });
  const match = path.match(/^\/p\/([^/]+)/);
  const pid = match ? decodeURIComponent(match[1]) : null;
  const loaded = config.data?.models.find((m) => m.id === config.data?.loaded);

  return (
    <div className="app">
      <header className="topbar">
        <a
          className="brand"
          href="/"
          onClick={(e) => {
            e.preventDefault();
            navigate("/");
          }}
        >
          <span className="logo">
            <Icon name="music" size={16} />
          </span>
          Cover Studio
        </a>
        {pid && (
          <>
            <span className="crumb-sep">/</span>
            <SongSwitcher
              pid={pid}
              onSelect={(id) => navigate(`/p/${encodeURIComponent(id)}`)}
              onNew={() => setNewSong(true)}
              onAll={() => navigate("/")}
            />
            <button className="btn small new-song-btn" onClick={() => setNewSong(true)}>
              <Icon name="plus" size={14} /> {t.newSong}
            </button>
          </>
        )}
        <div className="grow" />
        {config.data && (
          <div className="engine" title={`${t.device}: ${config.data.device}`}>
            <Icon name="chip" size={15} />
            <span>{config.data.device}</span>
            <span className="muted">·</span>
            <span className={loaded ? "" : "muted"}>{loaded ? shortModel(loaded.label) : t.noModelLoaded}</span>
            {loaded && (
              <button
                className="link"
                title={t.freeVramHelp}
                onClick={async () => {
                  try {
                    await api.unload();
                  } finally {
                    void qc.invalidateQueries({ queryKey: ["config"] });
                  }
                }}
              >
                {t.freeVram}
              </button>
            )}
          </div>
        )}
        <button
          className="lang-btn"
          onClick={() => setLang(lang === "ja" ? "en" : "ja")}
          title={t.languageHelp}
          aria-label={t.languageHelp}
        >
          {t.language}
        </button>
      </header>
      {config.data && config.data.api_version !== API_VERSION && <div className="stale-banner">{t.staleServer}</div>}
      <main className="content">
        {config.isError ? (
          <div className="empty-state">
            {t.noServer}
          </div>
        ) : !config.data ? (
          <div className="empty-state">{t.loading}</div>
        ) : pid ? (
          <Studio key={pid} pid={pid} config={config.data} navigate={navigate} />
        ) : (
          <Library config={config.data} navigate={navigate} />
        )}
      </main>
      {newSong && config.data && (
        <NewSongDialog
          config={config.data}
          onClose={() => setNewSong(false)}
          onCreated={(id) => {
            setNewSong(false);
            navigate(`/p/${encodeURIComponent(id)}`);
          }}
        />
      )}
    </div>
  );
}

const client = new QueryClient({ defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } } });

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
);
