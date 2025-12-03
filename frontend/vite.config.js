import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

const proxyHost = process.env.VITE_PROXY_HOST || 'http://localhost:8000';
const apiBase = process.env.VITE_API_BASE || '/api';

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      [apiBase]: {
        target: proxyHost,
        changeOrigin: true,
        rewrite: (path) => path.replace(apiBase, ''),
        secure: false,
      },
    },
  },
});