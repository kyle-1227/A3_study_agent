import { fileURLToPath } from "node:url"

import { defineConfig } from "vitest/config"

const frontendRoot = fileURLToPath(new URL(".", import.meta.url))

export default defineConfig({
  resolve: {
    alias: {
      "@": frontendRoot,
    },
  },
  test: {
    environment: "node",
    setupFiles: ["./test/setup.ts"],
    clearMocks: true,
    restoreMocks: true,
  },
})
