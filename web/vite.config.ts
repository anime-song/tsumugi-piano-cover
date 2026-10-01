import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// npm run dev: 画面は http://localhost:5173、/api は python -m cover_studio (:8100) へ中継する
// npm run build: cover_studio/static に書き出す (python -m cover_studio が / で配る)
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../cover_studio/static", emptyOutDir: true },
  server: { proxy: { "/api": "http://127.0.0.1:8100" } },
});
