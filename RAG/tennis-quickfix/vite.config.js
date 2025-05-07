import path from "path"; // Import path module
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
// https://vite.dev/config/
export default defineConfig({
    plugins: [react()],
    resolve: {
        alias: {
            "@": path.resolve(__dirname, "./src"), // Add alias resolution
        },
    },
    server: {
        port: 4000,
    },
});
