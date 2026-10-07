import { defineConfig } from "vite";

// `npm run build` writes the bundle into the Python package, where
// `vizopt.server` serves it. `npm run dev` proxies the WebSocket to a
// running `vizopt.server.serve(...)` on its default port.
export default defineConfig({
  build: {
    outDir: "../src/vizopt/server/static",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/ws": { target: "ws://127.0.0.1:8765", ws: true },
    },
  },
});
