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
  -d '{"name":"assistant","model_path":"llama3.gguf"}' | jq

# Chat (streaming SSE)
curl -N -X POST http://localhost:9090/v1/agents/<id>/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello!"}'
```

## CLI Mode (Terminal Assistant)

Both binaries now support explicit mode selection:

```sh
./mantic-mind --mode tui   # default
./mantic-mind --mode cli   # interactive REPL
./mantic-mind --mode cli --output json

./mantic-mind-control --mode tui   # default
./mantic-mind-control --mode cli   # interactive REPL
./mantic-mind-control --mode cli --output json
```

Use `--help` on either binary for full command help:

```sh
./mantic-mind --help
./mantic-mind-control --help
```

### Assistant-oriented CLI flow

```sh
# Terminal 1: control CLI
./mantic-mind-control --mode cli --output text

# Terminal 2: node CLI
./mantic-mind --mode cli --output text
```

Example control CLI session:

```text
mm-control> nodes discovered
mm-control> nodes pair psk http://127.0.0.1:7070 my-shared-key
mm-control> nodes llama check <node_id>
mm-control> models list
mm-control> agents create {"name":"assistant","model_path":"llama3.gguf"}
mm-control> agents list
mm-control> chat send <agent_id> "Hello from CLI mode"
mm-control> activity tail 50
```

Example node CLI session:

```text
mm-node> status
mm-node> metrics
mm-node> slots
mm-node> models list
mm-node> llama check-update
mm-node> logs tail 100
```

For automation-friendly output:

```sh
./mantic-mind-control --mode cli --output json
./mantic-mind --mode cli --output json
```

`--output json` emits one JSON object per command result. Streaming `chat send` emits JSON event lines (`delta`, `thinking`, `tool_call`, `done`) followed by a deterministic completion object.

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

Model reference contract:

- Prefer catalog filenames for `agent.model_path` (example: `llama3.gguf`).
- Control catalog includes GGUF files discovered in `models_dir` plus resolvable GGUF files referenced by agent `model_path`.
- Legacy absolute paths are still accepted, but control normalizes to `basename(model_path)` for placement and distribution decisions.
- Node pull/delete APIs reject path traversal and non-filename model references.

## REST API

### Node API (`Authorization: Bearer <node-api-key>`)

```
POST   /api/node/load-model         { model_path, settings }
POST   /api/node/unload-model
POST   /api/node/infer              { InferenceRequest }     -> SSE
POST   /api/node/models/pull        { model_filename, force? }
DELETE /api/node/models/{filename}  (node-local delete only)
POST   /api/node/llama/update       { build?: bool, force?: bool } -> starts node-side updater
POST   /api/node/llama/check-update {}                        -> refreshes version check
GET    /api/node/llama/status       -> runtime/update/version status
GET    /api/node/llama/jobs/{id}/log?offset=&limit=          -> updater log slices
GET    /api/node/health             -> NodeHealthMetrics
GET    /api/node/status             -> { node_id, loaded_model, ... }
GET    /api/node/models             -> { models: [{path, name, size_mb}] }
GET    /api/node/storage            -> { disk_free_mb, stored_models }
GET    /api/node/logs?tail=n        -> { lines: [...] }
GET    /api/node/api-keys           -> { keys: [...] }
POST   /api/node/api-keys           { key }
DELETE /api/node/api-keys/{key}
GET    /api/node/pair-status        -> { pending, mode?, expires_ms?, challenge?, pin? }
```

### Node -> Control handshake

```
POST   /api/control/register-node  { node_url, api_key, platform }
                                   -> { node_id, accepted }
```

### Control Internal API (node-authenticated bearer token)

```
GET    /api/control/models
GET    /api/control/models/{filename}/content   (streamed GGUF bytes with hash/size headers)
```

Bearer auth rule: token must match a registered node `api_key` in `NodeRegistry`.

### External Client API

```
GET/POST       /v1/agents
GET/PUT/DELETE /v1/agents/{id}
POST           /v1/agents/{id}/chat               { message, conversation_id? } -> SSE
GET/POST       /v1/agents/{id}/conversations
GET/DELETE     /v1/agents/{id}/conversations/{cid}
POST           /v1/agents/{id}/conversations/{cid}/activate
POST           /v1/agents/{id}/conversations/{cid}/compact
GET            /v1/agents/{id}/memories
PUT/DELETE     /v1/agents/{id}/memories/{mid}
POST           /v1/agents/{id}/memories/extract   { conversation_id, start_index, end_index, context_before? }
GET            /v1/nodes
POST           /v1/nodes                            { url, api_key, platform? }
DELETE         /v1/nodes/{id}
GET            /v1/nodes/discovered
POST           /v1/nodes/pair/start                 { url }
POST           /v1/nodes/pair/complete              { url, nonce, pin_or_psk }
POST           /v1/nodes/pair/psk                   { url, psk? }   (falls back to MM_PAIRING_KEY)
POST           /v1/nodes/{id}/llama/check-update
POST           /v1/nodes/{id}/llama/update          { build?, force? }
GET            /v1/models
GET            /v1/nodes/{id}/models
POST           /v1/nodes/{id}/models/pull         { model_filename, force? }
DELETE         /v1/nodes/{id}/models/{filename}
GET            /v1/activity?tail=n&level=info|warn|error|0|1|2
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
| `p` (Nodes tab) | Pull selected control catalog model to selected node |
| `d` (Nodes tab) | Delete selected model from selected node |
| `r` (Nodes tab) | Refresh model catalog and selected node inventory |
| `q` | Quit |
| `Esc` | Close modal / editor / quit |

Node TUI extras:

- `j` / `k` or Arrow Down / Arrow Up: scroll `llama-server` output
- `PgUp` / `PgDn`: faster log scrolling
- `End`: jump back to live tail
- `u`: run node-side llama.cpp updater (status includes persistent updater log path)
- `p`: prompt for model filename and pull from control
- `d`: prompt for model filename and delete local copy
