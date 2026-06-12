import { defineConfig } from "vite";

export default defineConfig(({ mode }) => ({
  base: mode === "production" && process.env.GITHUB_PAGES === "true"
    ? "/badminton-shuttle-detector/"
    : "/",
  server: {
    host: "0.0.0.0"
  }
}));