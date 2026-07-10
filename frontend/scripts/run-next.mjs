import { spawn } from "node:child_process"
import { existsSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const COMMANDS = new Set(["dev", "build", "start"])
const command = process.argv[2]
if (!command || !COMMANDS.has(command)) {
  throw new Error("run-next requires one of: dev, build, start")
}

const frontendRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const repositoryEnv = resolve(frontendRoot, "..", ".env")
if (existsSync(repositoryEnv)) {
  if (typeof process.loadEnvFile !== "function") {
    throw new Error("Node.js 20.12 or newer is required to load the repository .env")
  }
  process.loadEnvFile(repositoryEnv)
}

if (!process.env.NEXT_PUBLIC_API_URL?.trim()) {
  throw new Error(
    "NEXT_PUBLIC_API_URL is required in the process environment or repository root .env",
  )
}

const nextBin = resolve(frontendRoot, "node_modules", "next", "dist", "bin", "next")
const child = spawn(process.execPath, [nextBin, command, ...process.argv.slice(3)], {
  cwd: frontendRoot,
  env: process.env,
  stdio: "inherit",
})

child.once("error", (error) => {
  throw error
})
child.once("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal)
    return
  }
  process.exit(code ?? 1)
})
