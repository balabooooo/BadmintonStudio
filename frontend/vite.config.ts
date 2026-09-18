import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// 开发时 Vite 跑在 5273，API 与 WebSocket 代理到本地 FastAPI（默认 8000）。
const API = process.env.BMS_API || 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5273,
    strictPort: false,
    proxy: {
      '/api': { target: API, changeOrigin: true },
      '/ws': { target: API.replace(/^http/, 'ws'), ws: true },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    chunkSizeWarningLimit: 1600,
  },
})
