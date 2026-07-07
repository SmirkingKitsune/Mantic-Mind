# mantic-mind

Distributed LLM inference cluster — two executables that turn any collection of machines into a coordinated AI backend.

- **mantic-mind** — cluster node; spawns and manages `vllm serve` engine subprocesses; FTXUI status TUI
- **mantic-mind-control** — cluster head; manages nodes, agents, conversations, memories; FTXUI management TUI

> **Branch note:** this is the `vllm-runtime` line — vLLM is the inference backend.
> The earlier llama.cpp-based runtime is preserved on the `llama-cpp-runtime` branch.

## Prerequisites

| Tool | Minimum version |
|---|---|
| CMake | 3.25 |
| vcpkg | latest (bootstrapped) |
| MSVC (Windows) | VS 2022 |
| GCC / Clang (Linux) | GCC 12 / Clang 15 |
| Apple Clang (macOS) | Xcode 15 command-line tools |
| Python + vLLM | on each node — the node launches `vllm serve`; it can use an existing `vllm` executable or auto-provision a managed runtime |

Runtime extras (node side, optional):

- **Ray** (Linux) — required only for multi-node engine groups (one large model spanning nodes).
- **`hf` CLI** (Hugging Face) — used by the optional out-of-band model pre-fetch endpoint.

Set the `VCPKG_ROOT` environment variable to your bootstrapped vcpkg directory, or put `vcpkg` on `PATH`. If vcpkg is not available, CMake must find all dependencies through the system package manager or another toolchain.

On Ubuntu/Debian, the non-vcpkg build expects these development packages:

```sh
sudo apt-get install cmake g++ libftxui-dev libsqlitecpp-dev libcpp-httplib-dev nlohmann-json3-dev libspdlog-dev libssl-dev
```

For Linux x64-to-AArch64 cross builds, also install the GNU AArch64 toolchain:

```sh
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

## Build

Use the portable wrapper for the host shell:

Linux / macOS / WSL / Git Bash:

```sh
./scripts/build.sh
```

Windows PowerShell:

```powershell
.\scripts\build.ps1
```

By default the wrapper builds Release into `build/<os>-<arch>-release`, auto-detects vcpkg when possible, and uses all available CPU cores. Common options:

```sh
./scripts/build.sh --debug
./scripts/build.sh --arch aarch64
./scripts/build.sh --config RelWithDebInfo
./scripts/build.sh --generator Ninja
./scripts/build.sh --triplet x64-windows
./scripts/build.sh --install-prefix dist
./scripts/build.sh -- -DBUILD_TESTING=OFF
```

The same options are available in PowerShell:

```powershell
.\scripts\build.ps1 -DebugBuild
.\scripts\build.ps1 -Arch aarch64
.\scripts\build.ps1 -Config RelWithDebInfo
.\scripts\build.ps1 -Generator Ninja
.\scripts\build.ps1 -Triplet x64-windows
.\scripts\build.ps1 -InstallPrefix dist
.\scripts\build.ps1 -- -DBUILD_TESTING=OFF
```

You can also use CMake directly:

```sh
# Configure (vcpkg installs all dependencies automatically)
cmake -B build -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"

# Build both executables
cmake --build build
```

For a Release build with single-config generators, add `-DCMAKE_BUILD_TYPE=Release`. For multi-config generators such as Visual Studio or Xcode, build with `--config Release`.

Binaries are produced at:

- `build/<...>/src/node/<config>/mantic-mind[.exe]`
- `build/<...>/src/control/<config>/mantic-mind-control[.exe]`

### AArch64 Builds

Native AArch64 hosts (e.g. an Apple Silicon Mac or an ARM64 Linux box) build through the default wrapper path:

```sh
./scripts/build.sh
```

To request an AArch64 target explicitly:

```sh
./scripts/build.sh --arch aarch64
.\scripts\build.ps1 -Arch aarch64
```

The wrapper maps `--arch aarch64` / `-Arch aarch64` to the appropriate vcpkg triplet when vcpkg is enabled:

| Host target | vcpkg triplet | Extra behavior |
|---|---|---|
| Linux AArch64 | `arm64-linux` | Uses `aarch64-linux-gnu-gcc/g++` automatically when cross-building from non-AArch64 Linux |
| macOS Apple Silicon | `arm64-osx` | Sets `CMAKE_OSX_ARCHITECTURES=arm64` |
| Windows ARM64 | `arm64-windows` | Adds `-A ARM64` for Visual Studio generators (including the default when `-G` is omitted) |

If your cross compiler is not named `aarch64-linux-gnu-g++`, pass a toolchain or compiler settings after `--`:

```sh
./scripts/build.sh --arch aarch64 --triplet arm64-linux -- \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=/path/to/aarch64-gcc \
  -DCMAKE_CXX_COMPILER=/path/to/aarch64-g++
```

CMake presets are available for IDEs and CLI workflows (the most robust path for cross/ARM targets, since they pin arch + triplet explicitly):

```sh
cmake --preset vcpkg-release
cmake --build --preset vcpkg-release
ctest --preset vcpkg-release

cmake --preset vcpkg-aarch64-linux-release
cmake --build --preset vcpkg-aarch64-linux-release
```

Presets also exist for `vcpkg-aarch64-macos-release` and `vcpkg-aarch64-windows-release`.

Install both binaries into a predictable layout with:

```sh
cmake --install build --config Release --prefix dist
# dist/bin/mantic-mind[.exe]
# dist/bin/mantic-mind-control[.exe]
```

### GitHub Release Packages

Create GitHub-ready ZIP assets from a Release build with:

```powershell
.\scripts\package-release.ps1
```

The packager writes `dist/mantic-mind-<version>-<platform>-<arch>.zip`, a
matching `-symbols.zip` when PDBs are available, and `dist/checksums.txt`.
The runtime archive includes `bin/`, default root config files, `tools/`,
`README.md`, `LICENSE`, and a release manifest. It does not bundle vLLM,
Python environments, model weights, runtime data, logs, or local agent
databases.

## vLLM Runtime

Each node manages one or more `vllm serve` engine subprocesses. Agent configs
carry an `inference_backend` of `vllm` plus a `vllm_settings` block that drives
the engine launch (`--max-model-len`, `--max-num-seqs`, `--gpu-memory-utilization`,
`--tensor-parallel-size`, `--pipeline-parallel-size`, tool-parser flags, sleep
mode, etc.).

Engine source policy (for nodes that build vLLM from source):

- Linux nodes target the upstream vLLM repository: `https://github.com/vllm-project/vllm`
- Windows nodes target the Windows fork: `https://github.com/SystemPanic/vllm-windows`, branch `vllm-for-windows`
- Apple Silicon macOS nodes target the Metal plugin: `https://github.com/vllm-project/vllm-metal`

When `vllm_server_path` / `MM_VLLM_PATH` cannot be resolved, nodes can
auto-provision a managed runtime under `vllm_provision_dir` (default
`data/runtimes/vllm`). Windows prefers release wheels from
`SystemPanic/vllm-windows`, Linux uses a managed venv for official vLLM, and
Apple Silicon macOS uses `vllm-metal` with native arm64 Python 3.12 plus Xcode
command line tools. Release packages still do not bundle vLLM, Python
environments, or model weights.

The runtime is designed to use vLLM to its full capability:

- **Per-node GPU budget.** A node hands each engine an explicit
  `--gpu-memory-utilization` slice out of a configurable `vllm_gpu_budget`, so
  multiple small models can co-reside on one GPU. Loads are clamped to the
  remaining budget and rejected when it is exhausted.
- **Shared engines.** Multiple agents that request the same model with
  compatible launch settings attach to one running engine (continuous batching)
  instead of spawning a second process. `max_num_seqs` defaults to 16.
- **Sleep / wake suspension.** Idle engines are suspended via vLLM sleep mode
  (`/sleep` offloads weights to host RAM, freeing the GPU) and woken in seconds,
  falling back to a full stop when sleep is unavailable.
- **Load-aware routing.** Control scrapes each engine's Prometheus `/metrics`
  (running/waiting requests, KV-cache usage) and routes new agents to the
  least-loaded compatible engine; busy engines are protected from eviction.
- **Multi-node engine groups.** Control plans Ray-backed groups that span nodes
  for models too large for one machine (tensor parallel within a node, pipeline
  parallel across nodes), gated on advertised node capabilities (arch, vLLM
  build fingerprint, comm backend, Ray support). The live multi-node launch is
  Linux-only and requires a matching-architecture node set.
- **Hugging Face model cache.** A model reference is resolved by vLLM as an HF
  repo id (auto-downloaded to the HF cache), a cache hit, or a local directory.
  Nodes report which repos they have cached; control prefers a node that already
  holds the model. Point every node's `hf_cache_dir` at one shared location to
  download a model once for the whole cluster.

Qwen3-TTS can be routed directly to a vLLM-compatible speech endpoint by setting
`tts_enabled = true` and `tts_backend = "vllm"` in `mantic-mind-control.toml`
(or `MM_TTS_BACKEND=vllm`). The route, base URL, and served model are
configurable through `tts_vllm_base_url`, `tts_vllm_speech_path`, and
`tts_vllm_model_id`; the older Python sidecar remains available as
`tts_backend = "sidecar"`.

## Agent API Mode

Agents can also run against an OpenAI-compatible chat completions API instead of
a local node engine. Set `inference_backend` to `api`, keep `model_path` as the
remote model id, and configure `api_settings` with the provider base URL and
chat route:

```sh
curl -X POST http://localhost:9090/v1/agents \
  -H "Content-Type: application/json" \
  -d '{
    "name": "frontier-assistant",
    "inference_backend": "api",
    "model_path": "gpt-4.1",
    "api_settings": {
      "base_url": "https://api.openai.com",
      "chat_completions_path": "/v1/chat/completions",
      "api_key_env": "OPENAI_API_KEY"
    }
  }' | jq
```

`api_settings.api_key` may be supplied in a create/update request for the
current control process, but it is not serialized in API responses and is not
persisted to SQLite. Prefer `api_key_env` for restart-safe deployments. API
agents bypass node placement, but they still use the normal Mantic-Mind agent
harness: conversations, streaming chat, memory extraction, compaction, and tool
rounds.

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
cp tools/mantic-mind.toml .            # set control_url (and vllm_server_path if needed) locally
./mantic-mind
# or entirely via env vars:
MM_CONTROL_URL=http://192.168.1.100:9090 \
MM_SELF_URL=http://192.168.1.5:7070      \
MM_VLLM_PATH=vllm                        \
./mantic-mind
```

The node TUI shows the generated API key. In the control TUI, press **`[+] Add Node`**, paste the URL and key — the node joins the cluster.

### 3. Create an agent and chat

```sh
# Create (model_path is an HF repo id, a cached model, or a local model dir)
curl -X POST http://localhost:9090/v1/agents \
  -H "Content-Type: application/json" \
  -d '{"name":"assistant","inference_backend":"vllm","model_path":"Qwen/Qwen2.5-0.5B-Instruct"}' | jq

# Chat (streaming SSE)
curl -N -X POST http://localhost:9090/v1/agents/<id>/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello!"}'
```

## CLI Mode (Terminal Assistant)

Both binaries support explicit mode selection:

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
mm-control> models list
mm-control> agents create {"name":"assistant","inference_backend":"vllm","model_path":"Qwen/Qwen2.5-0.5B-Instruct"}
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
| `control_url` | `MM_CONTROL_URL` | *(empty)* | Control base URL (empty = standalone) |
| `control_api_key` | `MM_CONTROL_API_KEY` | *(empty)* | Bearer token for control |
| `vllm_server_path` | `MM_VLLM_PATH` | `vllm` | `vllm` CLI / wrapper used to launch engines |
| `vllm_auto_provision` | `MM_VLLM_AUTO_PROVISION` | `true` | Auto-create a managed vLLM runtime when `vllm_server_path` is unresolved |
| `vllm_provision_dir` | `MM_VLLM_PROVISION_DIR` | `data/runtimes/vllm` | Managed vLLM runtime root |
| `vllm_install_method` | `MM_VLLM_INSTALL_METHOD` | `auto` | Provisioning method: `auto`, `wheel`, or `source` |
| `vllm_version` | `MM_VLLM_VERSION` | `latest` | vLLM release tag/version/commit used by provisioning |
| `vllm_python_path` | `MM_VLLM_PYTHON_PATH` | *(auto)* | Python executable used for managed venv creation |
| `vllm_gpu_budget` | `MM_VLLM_GPU_BUDGET` | `0.90` | Total GPU fraction all vLLM engines on this node may claim |
| `max_slots` | `MM_MAX_SLOTS` | `4` | Maximum concurrent engine slots |
| `runtime_port_range_start` | `MM_RUNTIME_PORT_RANGE_START` | `8080` | First port in the per-engine port range |
| `runtime_port_range_end` | `MM_RUNTIME_PORT_RANGE_END` | `8090` | Last port in the per-engine port range |
| `models_dir` | `MM_MODELS_DIR` | `models` | Optional local model directory root |
| `data_dir` | `MM_DATA_DIR` | `data` | Node runtime data root; remembered pairing keys live in `api_keys.json` |
| `kv_cache_dir` | `MM_KV_CACHE_DIR` | `data/kv_cache` | KV-cache checkpoint directory |
| `comm_backends` | `MM_COMM_BACKENDS` | *(auto)* | CSV of collective backends advertised (`nccl,gloo`); auto-detected when empty |
| `supports_ray` | `MM_SUPPORTS_RAY` | *(auto)* | Whether this node can join multi-node Ray engine groups (auto = true on Linux) |
| `node_gpu_count` | `MM_NODE_GPU_COUNT` | `0` (auto) | GPUs advertised; `0` auto-detects via `nvidia-smi` |
| `interconnect_gbps` | `MM_INTERCONNECT_GBPS` | `0` | Node-to-node link hint for group scoring |
| `ray_path` | `MM_RAY_PATH` | `ray` | Ray CLI for multi-node engine groups (Linux) |
| `ray_port` | `MM_RAY_PORT` | `6379` | Ray head GCS port |
| `hf_cache_dir` | `MM_HF_CACHE_DIR` | *(auto)* | HF hub cache dir override; pins `HF_HUB_CACHE` for the engine (set the same shared path on every node to download once cluster-wide) |
| `hf_cli_path` | `MM_HF_CLI_PATH` | `hf` | Hugging Face CLI used for out-of-band model pre-fetch (Linux) |
| `pairing_key` | `MM_PAIRING_KEY` | *(empty)* | PSK for automatic node/control pairing |
| `discovery_port` | `MM_DISCOVERY_PORT` | `7072` | UDP discovery port |
| `log_file` | `MM_LOG_FILE` | `logs/mantic-mind.log` | Log file |

### `mantic-mind-control.toml` — control config

| Key | Env var | Default | Description |
|---|---|---|---|
| `listen_port` | `MM_CONTROL_PORT` | `9090` | API server port |
| `openai_compat_port` | `MM_OPENAI_COMPAT_PORT` | `9091` | OpenAI-compatible text API port; set `0` to disable |
| `data_dir` | `MM_DATA_DIR` | `data` | Agent database root; remembered nodes are stored in `nodes.json` |
| `models_dir` | `MM_MODELS_DIR` | `models` | Optional local model directory root |
| `node_health_poll_interval_s` | `MM_POLL_INTERVAL_S` | `30` | Health/metrics poll interval |
| `external_api_token` | `MM_CONTROL_EXTERNAL_API_TOKEN` | *(empty)* | When set, required as `Authorization: Bearer` on all external `/v1/*` routes |
| `pairing_key` | `MM_PAIRING_KEY` | *(empty)* | PSK for automatic node/control pairing |
| `discovery_port` | `MM_DISCOVERY_PORT` | `7072` | UDP discovery port |
| `log_file` | `MM_LOG_FILE` | `logs/mantic-mind-control.log` | Log file |

Text-to-speech is configured through the `tts_*` keys (`tts_enabled`,
`tts_backend`, `tts_vllm_base_url`, `tts_vllm_speech_path`, `tts_vllm_model_id`,
`tts_vllm_api_key` / `tts_vllm_api_key_env`, `tts_service_url`,
`tts_service_command`, `tts_timeout_s`, `tts_cache_dir`, and the voice/clone
model ids), each overridable with the matching `MM_TTS_*` environment variable.

Additional env-only settings:

| Env var | Scope | Description |
|---|---|---|
| `MM_SELF_URL` | node | Public URL advertised to control (defaults to `http://127.0.0.1:<listen_port>`) |
| `MM_API_KEY` | node | Initial node API key (generated if omitted) |

### Model reference contract

`agent.model_path` is passed to vLLM and resolved as one of:

- a **Hugging Face repo id** (`org/name`, e.g. `Qwen/Qwen2.5-0.5B-Instruct`) — downloaded to the node's HF cache on first use;
- a **cached model** already present in the HF cache; or
- a **local model directory** on the node.

Nodes report their cached HF repo ids to control, which prefers a node that
already holds the model when placing an agent. Models can be pre-fetched
out-of-band (see `POST /api/node/models/pull`) so a large download does not
happen inside the engine's load timeout window.

## REST API

### Node API (`Authorization: Bearer <node-api-key>`)

```
POST   /api/node/load-model     { model_path, vllm_settings?, agent_id? }
POST   /api/node/unload-model   { slot_id? }            (omit slot_id to unload all)
POST   /api/node/detach-agent   { slot_id, agent_id }   (unloads engine when its last agent leaves)
POST   /api/node/suspend-slot   { slot_id }             (vLLM sleep, or stop)
POST   /api/node/restore-slot   { model_path, vllm_settings?, agent_id? }
POST   /api/node/infer          { InferenceRequest, slot_id? }  -> SSE
POST   /api/node/models/pull    { model_ref }           (HF pre-fetch; Linux only, 501 elsewhere)
GET    /api/node/runtime/vllm   -> { vllm_runtime }
POST   /api/node/runtime/vllm/provision -> { vllm_runtime }
POST   /api/node/ray/start      { role: "head"|"worker", head_address? }  (Linux only)
POST   /api/node/ray/stop
GET    /api/node/health         -> NodeHealthMetrics
GET    /api/node/status         -> { node_id, slots, cached_models, capabilities, vllm_runtime, vllm_gpu_budget, ... }
GET    /api/node/logs?tail=n    -> { lines: [...] }
GET    /api/node/api-keys       -> { keys: [...] }
POST   /api/node/api-keys       { key }
DELETE /api/node/api-keys/{key}
GET    /api/node/pair-status    -> { pending, mode?, expires_ms?, challenge?, pin? }
POST   /api/node/pair-request
POST   /api/node/pair-complete
```

### Node → Control handshake

```
POST   /api/control/register-node  { node_url, api_key, platform }
                                   -> { node_id, accepted }
```

### External Client API

If `external_api_token` is configured on control, every external `/v1` and
`/v1/*` route requires `Authorization: Bearer <external_api_token>`, including
SSE chat and mutating routes. Registered node API keys authenticate only the
internal `/api/control/*` routes; they are not accepted for external `/v1/*`
clients.

```
GET/POST       /v1/agents
GET/PUT/DELETE /v1/agents/{id}
POST           /v1/agents/{id}/chat                       { message, conversation_id? } -> SSE
GET/POST       /v1/agents/{id}/conversations
GET/DELETE     /v1/agents/{id}/conversations/{cid}
PUT            /v1/agents/{id}/conversations/{cid}
POST           /v1/agents/{id}/conversations/{cid}/activate
POST           /v1/agents/{id}/conversations/{cid}/compact
GET/POST       /v1/agents/{id}/conversations/{cid}/local-memories
PUT/DELETE     /v1/agents/{id}/conversations/{cid}/local-memories/{mid}
GET            /v1/agents/{id}/memories
PUT/DELETE     /v1/agents/{id}/memories/{mid}
POST           /v1/agents/{id}/memories                   (create)
POST           /v1/agents/{id}/memories/extract           { conversation_id, start_index, end_index, context_before? }
POST           /v1/agents/{id}/curation/proposals
POST           /v1/agents/{id}/curation/proposals/apply
POST           /v1/agents/{id}/curation/apply
GET            /v1/agents/{id}/voice
GET/POST       /v1/agents/{id}/voice/proposals
POST           /v1/agents/{id}/voice/proposals/{pid}/approve | reject | sample
GET            /v1/agents/{id}/voice/proposals/{pid}/sample
POST           /v1/agents/{id}/speech                     (synthesize)
GET            /v1/agents/{id}/speech/cache/{cache_id}
POST           /v1/audio/speech                           (OpenAI-compatible speech)
GET            /v1/nodes
POST           /v1/nodes                                  { url, api_key, platform?, remember? }
DELETE         /v1/nodes/{id}
POST           /v1/nodes/{id}/forget
GET            /v1/nodes/discovered
POST           /v1/nodes/pair/start                       { url }
POST           /v1/nodes/pair/complete                    { url, nonce, pin_or_psk, remember? }
POST           /v1/nodes/pair/psk                         { url, psk?, remember? }   (falls back to MM_PAIRING_KEY)
GET            /v1/placements                             -> current agent → node/slot placements
GET            /v1/activity?tail=n&level=info|warn|error|0|1|2
GET            /v1/models                                 -> agent model catalog for compatibility clients
```

### OpenAI-Compatible API (`openai_compat_port`, default `:9091`)

This listener is intentionally separate from the Mantic API port. It is for
clients that support an OpenAI-compatible custom base URL, while richer app
integrations should keep using the Mantic `/v1/agents`, `/v1/nodes`, and
conversation/memory routes on `listen_port`.

```
GET  /v1/models
GET  /v1/models/{model}
POST /v1/chat/completions
```

`/v1/models` exposes agents as model IDs in the form `agent:{agent_id}`. The
chat-completions route also accepts a bare agent ID, unique agent name, unique
`model_path`, or unique `served_model_name`, but `agent:{agent_id}` is the
stable form to configure in external clients. When `external_api_token` is set,
the compatibility port uses the same `Authorization: Bearer <token>` gate.

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
  ├─ NodeRegistry      — node list + health/metrics polling, capabilities
  ├─ AgentScheduler    — placement: existing/suspended → shared engine →
  │                       cached-model node → VRAM → sleep/evict + retry;
  │                       multi-node engine-group planning
  ├─ AgentQueue        — per-agent FIFO worker threads
  └─ ControlApiServer  — REST endpoints + SSE chat proxy
         │  REST + SSE
         ▼
mantic-mind  (:7070)
  ├─ NodeState          — API keys, metrics, capabilities
  ├─ SlotManager        — per-engine slots: GPU budget, shared engines, sleep/wake
  ├─ RuntimeProcess     — engine subprocess supervisor (Windows CreateProcess / POSIX fork+exec)
  ├─ RuntimeClient      — OpenAI-compatible HTTP client → the engine
  └─ NodeApiServer      — node endpoints + SSE infer proxy
         │  HTTP (OpenAI-compatible)
         ▼
   vllm serve  (engine, :8080+)
```

(`RuntimeProcess` supervises and `RuntimeClient` talks to `vllm serve`.)

## TUI Keyboard Shortcuts

| Key | Action |
|---|---|
| `1` / `2` / `3` / `4` / `5` | Switch tabs (Nodes / Agents / Activity / Chat / Curation) |
| `q` | Quit |
| `Esc` | Close modal / editor, or quit |

Node TUI extras:

- `j` / `k` or Arrow Down / Arrow Up: scroll engine output
- `PgUp` / `PgDn`: faster log scrolling
- `End`: jump back to live tail

                                                                 ░             ░█                   
                                                           ░█░██  ░ ██       █░█                    
                                          ░░░▒████████▒░  ░████▒  ███░     ▒█░█░                    
                                    ░██████████████████████████ ░  ░█░   ░██░ ░ ░███░  ░██░  ░ ░    
                                ██████████████▒░░  ░░▓████████░░███▒ ███░   ░█ ████░░░███░     ░    
                            ▒████████▓ ░░██████████████████░░░░████████░░░ ░█░ ░░   ░░░██    ░      
                       ░░░███████░░▓████           ░  ░    ██████░░███ ░██░    ░░   ░████░  █░  ░   
                       ███████ ░███░                       █▓ ░█████░░███░██░░░█░ █████████▓▒       
                     ██████░░██░                          ██  ██ ██░███░░█████░   ░░▓███████░      ░
                  ░██████ ███          ░ ░░ ░░░      ░░▓░ ░    ░██ ░█░░██░░█░░▓░████████▒▓      ░░█░
                 ░█████░██░          ▒██████░ ░██░ ▓█▓░ ░▒██████░  ░  ░░░█░░ █████████▓█░░░     ░  ░
                █████░██░        ▒████   ▓█▒░░  ██░█░░░░░▒█    ░█░░  ███░░██░████████ ▓██░         ░
              ░████░░██   ░   ███░░░░ ███░ ░██░░▓█░█ ████░░░ ░█░░░████░ █░░████████░▓░░ ░ ░         
             ░████░██░      ░██ █▓  █▒  ░██░    ▓█░█ ░░░░░▓█░ ░░██▓░███░████████   ░░░░█▓ ░░        
            ░████░██       ░██▓█░ █░   ░█░░███░░██░█░█░░▒▓░ █░  ░▒██████████████░░▓░█████▒░░   ░░   
           ░████░██     ░░███   ░█░ ░███████░  █░█░░█       ░░████████████░░ ░ ░ ░░░  ░░░░███       
          ░████░██     ░██░ ░█░░░█░░░  ░        █░░█    █░░███████████░░░███░░████░░  █░ ░░██▓░    ░
         ░████░██      ░█░░░█  ░▓░ ░  ░░        ░█░█     ░░░██████░▓░░░ ░██░ █░░ ░ ░░██░░ ░███████ ░
         ▓███░██      ░██ ░█  ░░████▓ ░░█   ███░██░█ ███▒░███░▒▓██  ░█░░░░█░   ░░░ █░      ▓░      ░
         ████░█░     ░█░  █   █░  ▓░           ░█░░██ ███░░░█▓      ░ ██░░░░█  ████████████░       ░
        ▒███░██      ██   █░░░░   ░██           ░█░██░████░       ▓██ ░██  ░██ ░ ███░███░    ███    
        ████░█░      ██    ░░░ ░█░  ░█   ░░░░  ██░░ ░░  ▓█░░█░███    ████░▓███  ░░░███████░         
        ████░█░     ░█░█░█░     ░▓  ░   ░░░░█░░░    ░██            ░█ ░░░█░▒█░░▓█░    █▓            
       ░███░██      ██  ░█   ░  ░█░     ░░      █░ █░              █░    ░█▓█░▒███  ░░░  ░          
       ░███░██      ░█░  ░  ░█    ░█            ░█░█              █▒   ▓░░█░░▒█░▒█ ░██ ██████       
       ▒███░██       ░█   ░█░▓    ░███░░░        █░█       ░░███░░     █░░█ ░░██ ░░▓░░██░████     ░░
       ░███░██        ██  ▓░ █░  ░      ░█▒   ░ █▓░█ ░   ░█░    ░   █ ░█░░░  ░█░░░    ██░███████████
       ░███░██         █░█░█   ▓░         ░   ░▓█▓░██              ░▓ ░▒ ░█  █░ ░█ ░  ██░███████████
        ████░█       ░███░▓░   ░░████            █░█          ▒████░ ░█░░  █ █░      ░█▒▓███░ ░░░  ░
        ████░█░        █░█░░▒░       ░█  █       █░█        ░█           ░█▒██       ░█░████        
        ░███░██        ▓█░▒▒          ░██      ░█░░█░      █░░          ▒░███        ██░███▓       ░
         ████░█░         ███      ░     ░█   ░█░░█░█░███░ ░     ░     ░▒░░█▒        ░█░████         
         ░███░██           ░██████░      ░░░░░░ ░█▒█             ░████░ ▓██         ██░████         
          ████ ██             ░  ░█░            █░░█▒           ░█ ░  ▓█           ██░████░         
          ░████░██                    ██░░░░███░ ██░░██▓░░░░██░      ░█░          ▒█░████░          
           ░████░██                               ██       ░        ██░          ▒█░████░           
             ████░██                          ░    ██░    ▒   ░  ███░           ██░████░            
             ░████░██░                              ░██░  ▓█▓██▓░             ░██░████░             
               ████░░█▒                               ▓█   █                 ░█░░████░              
                █████░██░                              ██  ██               ██░█████                
                 ░█████░██░                            ░████░            ░██░█████░                 
                   ██████░██░                                           ██ ▓█████                   
                    ░██████░▒██                                     ░██▓░██████░                    
                       ██████▓░███▒                              ▒███ ▒██████                       
                         ░███████░░████░                   ░░████░░███████░                         
                           ░▒████████░░░████████████████████░░░████████▒░                           
                                ████████████▒░░░  ░░░░░▓████████████░                               
                                    ▒██████████████████████████▒░                                   
                                         ░░ ▒██████████▒  ░                                         
                                                   ░ 