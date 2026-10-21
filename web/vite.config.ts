import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// npm run dev: 画面は http://localhost:5173、/api は python -m cover_studio (:8100) へ中継する
// npm run build: cover_studio/static に書き出す (python -m cover_studio が / で配る)
export default defineConfig({
  plugins: [react()],
  // 楽譜を描く Verovio (WebAssembly を含む 8 MB ほど) は楽譜を開いたときだけ読む別の塊なので、大きさの警告は出さない
  build: { outDir: "../cover_studio/static", emptyOutDir: true, chunkSizeWarningLimit: 9000 },
  server: { proxy: { "/api": "http://127.0.0.1:8100" } },
});
