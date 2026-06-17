import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/dashboard-data.json": {
        target: "http://127.0.0.1:8001",
        changeOrigin: true,
      },
      "/api": {
        target: "http://127.0.0.1:8001",
        changeOrigin: true,
      },
      "/health": {
        target: "http://127.0.0.1:8001",
        changeOrigin: true,
      },
      "/models": {
        target: "http://127.0.0.1:8001",
        changeOrigin: true,
      },
      "/feedback": {
        target: "http://127.0.0.1:8001",
        changeOrigin: true,
      },
      "/predict": {
        target: "http://127.0.0.1:8001",
        changeOrigin: true,
      },
      "/explain": {
        target: "http://127.0.0.1:8001",
        changeOrigin: true,
      },
    },
  },
});
