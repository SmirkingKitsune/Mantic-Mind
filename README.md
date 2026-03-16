# mantic-mind

Distributed LLM inference cluster — two executables that turn any collection of machines into a coordinated AI backend.

- **mantic-mind** — cluster node; spawns and manages a `llama-server` subprocess; FTXUI status TUI
- **mantic-mind-control** — cluster head; manages nodes, agents, conversations, memories; FTXUI management TUI

## Prerequisites

| Tool | Minimum version |
|---|---|
| CMake | 3.24 |
| vcpkg | latest |
| MSVC (Windows) | VS 2022 |
| GCC / Clang (Linux) | GCC 12 / Clang 15 |
| llama-server | any recent llama.cpp build |

Set the `VCPKG_ROOT` environment variable to your vcpkg installation directory.

## Build

```sh
# Configure (vcpkg installs all dependencies automatically)
cmake -B build -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"

# Build both executables
cmake --build build

# Binaries land at:
#   build/src/node/Debug/mantic-mind[.exe]
#   build/src/control/Debug/mantic-mind-control[.exe]
```

For a Release build add `-DCMAKE_BUILD_TYPE=Release`.

## Sync llama.cpp

Use the helper in `tools/` to keep `llama.cpp` current and (optionally) rebuild `llama-server`.

Linux / WSL:
```sh
./tools/update-llama-cpp.sh --build
```

Windows PowerShell:
```powershell
.\tools\update-llama-cpp.ps1 -Build
```

Both scripts print the resolved `llama-server` path. Point node config to it:

```toml
llama_server_path = "C:\\path\\to\\llama-server.exe"  # Windows
# or
llama_server_path = "/path/to/llama-server"           # Linux
```

You can also use env vars instead:
```sh
MM_LLAMA_PATH=/path/to/llama-server
```

## Quick Start

Local `mantic-mind.toml` and `mantic-mind-control.toml` copies are intentionally gitignored. Keep committed defaults in `tools/`, then copy them into the working directory for local runs.

### 1. Run the control server

```sh
cp tools/mantic-mind-control.toml .   # edit locally as needed
./mantic-mind-control
```

`mantic-mind-control` enforces a single running instance per `(data_dir, listen_port)` pair. Starting a duplicate exits with an error.

### 2. Run a node

```sh
cp tools/mantic-mind.toml .            # set control_url and llama_server_path locally
./mantic-mind
# or entirely via env vars:
MM_CONTROL_URL=http://192.168.1.100:9090 \
MM_SELF_URL=http://192.168.1.5:7070      \
MM_LLAMA_PATH=/path/to/llama-server      \
./mantic-mind
```

The node TUI shows the generated API key. In the control TUI, press **`[+] Add Node`**, paste the URL and key — the node joins the cluster.

### 3. Create an agent and chat

```sh
# Create
curl -X POST http://localhost:9090/v1/agents \
  -H "Content-Type: application/json" \
  -d '{"name":"assistant","model_path":"/models/llama3.gguf"}' | jq

# Chat (streaming SSE)
curl -N -X POST http://localhost:9090/v1/agents/<id>/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello!"}'
```

## Configuration

Config loading order:

1. `MM_NODE_CONFIG_FILE` / `MM_CONTROL_CONFIG_FILE` (binary-specific override)
2. `MM_CONFIG_FILE` (shared override)
3. Search for `mantic-mind.toml` / `mantic-mind-control.toml` from the current directory upward (up to 10 parent directories)

After file loading, matching environment variables override config values.

### `mantic-mind.toml` — node config

| Key | Env var | Default | Description |
|---|---|---|---|
| `listen_port` | `MM_LISTEN_PORT` | `7070` | Node API port |
| `control_url` | `MM_CONTROL_URL` | *(empty)* | Control base URL |
| `control_api_key` | `MM_CONTROL_API_KEY` | *(empty)* | Bearer token for control |
| `llama_server_path` | `MM_LLAMA_PATH` | `llama-server` | llama-server binary |
| `llama_port` | `MM_LLAMA_PORT` | `8080` | Internal llama-server port |
| `llama_port_range_start` | `MM_LLAMA_PORT_RANGE_START` | `8080` | First port in the per-slot llama-server range |
| `llama_port_range_end` | `MM_LLAMA_PORT_RANGE_END` | `8090` | Last port in the per-slot llama-server range |
| `max_slots` | `MM_MAX_SLOTS` | `4` | Maximum concurrent model slots/processes |
| `models_dir` | `MM_MODELS_DIR` | `models` | Scanned for `.gguf` files |
| `data_dir` | `MM_DATA_DIR` | `data` | Node runtime data root |
| `kv_cache_dir` | `MM_KV_CACHE_DIR` | `data/kv_cache` | KV cache checkpoint directory |
| `pairing_key` | `MM_PAIRING_KEY` | *(empty)* | PSK for automatic node/control pairing |
| `discovery_port` | `MM_DISCOVERY_PORT` | `7072` | UDP discovery port |
| `log_file` | `MM_LOG_FILE` | `logs/mantic-mind.log` | Log file |

### `mantic-mind-control.toml` — control config

| Key | Env var | Default | Description |
|---|---|---|---|
| `listen_port` | `MM_CONTROL_PORT` | `9090` | API server port |
| `data_dir` | `MM_DATA_DIR` | `data` | Agent database root |
| `models_dir` | `MM_MODELS_DIR` | `models` | Model distribution root |
| `node_health_poll_interval_s` | `MM_POLL_INTERVAL_S` | `30` | Health poll interval |
| `pairing_key` | `MM_PAIRING_KEY` | *(empty)* | PSK for automatic node/control pairing |
| `discovery_port` | `MM_DISCOVERY_PORT` | `7072` | UDP discovery port |
| `log_file` | `MM_LOG_FILE` | `logs/mantic-mind-control.log` | Log file |

Additional env-only settings:

| Env var | Scope | Description |
|---|---|---|
| `MM_SELF_URL` | node | Public URL advertised to control (defaults to `http://127.0.0.1:<listen_port>`) |
| `MM_API_KEY` | node | Initial node API key (generated if omitted) |
| `MM_LLAMA_PATH_CACHE_FILE` | node | Path to persisted resolved `llama-server` binary path |
| `MM_LLAMA_REPO_URL` | node | Optional llama.cpp git remote for updater jobs |
| `MM_LLAMA_INSTALL_ROOT` | node | Optional install root for updater jobs |
| `MM_LLAMA_UPDATER_LOG_DIR` | node | Optional log directory for updater jobs |

## REST API

### Node API (`Authorization: Bearer <node-api-key>`)

```
POST   /api/node/load-model         { model_path, settings }
POST   /api/node/unload-model
POST   /api/node/infer              { InferenceRequest }     → SSE
POST   /api/node/llama/update       { build?: bool, force?: bool } → starts node-side updater
POST   /api/node/llama/check-update {}                        → refreshes version check
GET    /api/node/llama/status       → runtime/update/version status
GET    /api/node/llama/jobs/{id}/log?offset=&limit=          → updater log slices
GET    /api/node/health             → NodeHealthMetrics
GET    /api/node/status             → { node_id, loaded_model, ... }
GET    /api/node/models             → { models: [{path, name, size_mb}] }
POST   /api/node/api-keys           { key }
DELETE /api/node/api-keys/{key}
```

### Node → Control handshake

```
POST   /api/control/register-node  { node_url, api_key, platform }
                                   → { node_id, accepted }
```

### External Client API

```
GET/POST       /v1/agents
GET/PUT/DELETE /v1/agents/{id}
POST           /v1/agents/{id}/chat               { message, conversation_id? } → SSE
GET/POST       /v1/agents/{id}/conversations
GET/DELETE     /v1/agents/{id}/conversations/{cid}
POST           /v1/agents/{id}/conversations/{cid}/activate
POST           /v1/agents/{id}/conversations/{cid}/compact
GET            /v1/agents/{id}/memories
PUT/DELETE     /v1/agents/{id}/memories/{mid}
POST           /v1/agents/{id}/memories/extract   { conversation_id, start_index, end_index, context_before? }
GET            /v1/nodes
```

### SSE Chat Events

```
data: {"type":"thinking","content":"..."}
data: {"type":"delta",   "content":"..."}
data: {"type":"tool_call","name":"...","arguments":"..."}
data: {"type":"done",    "conv_id":"...","success":true}
data: [DONE]
```

## Architecture

```
External Client
      │  REST + SSE
      ▼
mantic-mind-control  (:9090)
  ├─ AgentManager      — per-agent SQLite  (data/agents/{uuid}/agent.db)
  ├─ NodeRegistry      — node list + health polling
  ├─ ModelRouter       — preferred → any → load → evict+load
  ├─ AgentQueue        — per-agent FIFO worker threads
  └─ ControlApiServer  — REST endpoints + SSE chat proxy
         │  REST + SSE
         ▼
mantic-mind  (:7070)
  ├─ NodeState          — API keys, metrics, loaded model
  ├─ LlamaServerProcess — subprocess (Windows CreateProcess / Linux fork+exec)
  ├─ LlamaCppClient     — OpenAI-compat HTTP → llama-server
  └─ NodeApiServer      — node endpoints + SSE infer proxy
         │  HTTP (OpenAI-compat)
         ▼
   llama-server  (:8080)
```

## TUI Keyboard Shortcuts

| Key | Action |
|---|---|
| `1` / `2` / `3` / `4` / `5` | Switch tabs (Nodes / Agents / Activity / Chat / Curation) |
| `u` (Nodes tab) | Trigger llama.cpp update on selected node |
| `q` | Quit |
| `Esc` | Close modal / editor / quit |

Node TUI extras:

- `j` / `k` or Arrow Down / Arrow Up: scroll `llama-server` output
- `PgUp` / `PgDn`: faster log scrolling
- `End`: jump back to live tail
- `u`: run node-side llama.cpp updater (status includes persistent updater log path)

## Implementation Status

All seven phases complete.

| Phase | Description |
|---|---|
| 1 | Scaffolding — CMake, vcpkg, stubs, models.hpp |
| 2 | Database layer (AgentDB, AgentManager) |
| 3 | LLM client + conversation / memory logic |
| 4 | Worker node (subprocess, API, FTXUI) |
| 5 | Control core (routing, queue, REST API) |
| 6 | Control UI (FTXUI five-tab TUI, including Chat and Curation) |
| 7 | Hardening (config files, reconnection, graceful shutdown) |
