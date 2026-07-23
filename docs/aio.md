# Mantic-Mind AIO

`mantic-mind-aio` runs the Mantic-Mind control plane and one inference node in
the same process. It is intended for a single workstation, or for users who
want a local node without control-to-node HTTP traffic.

The standalone `mantic-mind-control` and `mantic-mind` programs remain
available. AIO is an additional deployment mode, not a replacement for either
one.

## How it works

- The embedded node is registered with control as the local node during
  startup. Control invokes it through an in-process transport rather than the
  node REST API.
- The control TUI is the single interactive interface. The local node appears
  in its Nodes view alongside any configured remote nodes.
- Agent placement still uses the normal Mantic-Mind best-fit scheduler. The
  local node is available immediately; when clustering is enabled, remote nodes
  can also win placement when they are the better fit.
- Local model paths are passed directly to the embedded node. AIO does not
  upload a model back to the same machine, and node cache cleanup does not own
  or remove control's canonical model files.
- The embedded node still launches `llama-server` subprocesses on its configured
  runtime port range. "In process" refers to the control-to-node boundary; it
  does not embed llama.cpp itself in the AIO executable.

## Default network behavior

Clustering is disabled by default. In that mode:

- control-facing HTTP listeners bind to loopback only;
- no node API listener is created for the embedded node;
- UDP discovery listening and broadcasting are disabled;
- node registration and reconnect workers are not started; and
- remembered remote nodes are not contacted.

This removes control-to-node network chatter while preserving the loopback
control API used by local clients and integrations. The local `llama-server`
subprocess connections remain loopback traffic.

Clustering is a startup setting. Enabling or disabling it requires editing the
AIO configuration and restarting the process. When enabled, AIO can discover,
pair with, poll, and schedule work on remote nodes. Its embedded node remains a
private in-process node; AIO does not expose or advertise that node's REST API.

Binding `control.bind_host` beyond loopback requires both
`control.external_api_token` and `cluster.pairing_key`. AIO refuses an unsafe
LAN bind instead of starting an unauthenticated service. These credentials
provide authorization over HTTP; they do not add TLS encryption.

## Configuration

AIO uses `mantic-mind-aio.toml`. Its settings are namespaced into `[shared]`,
`[control]`, `[node]`, and `[cluster]` sections. Environment overrides use the
same namespace in the form `MM_AIO_<SECTION>_<KEY>`.

Configuration-file precedence is `--config <path>`, then
`MM_AIO_CONFIG_FILE`, then an upward search for `mantic-mind-aio.toml` from the
current directory. Namespaced environment values override the loaded file.

The shipped template is `tools/mantic-mind-aio.toml`. Pass it explicitly when
running from a source checkout, then copy it only when you want a local
customization:

```text
./mantic-mind-aio --config ./tools/mantic-mind-aio.toml
```

The sections have distinct responsibilities:

| Section | Purpose |
| --- | --- |
| `[shared]` | `data_dir`, `models_dir`, and `log_file` used by the composed process |
| `[control]` | The shared control/OpenAI-compatible `bind_host`, their listener ports, and `external_api_token` |
| `[node]` | llama.cpp provisioning and `runtime_network_policy`, slot and runtime-port limits, hardware overrides, KV cache, and model-cache policy |
| `[cluster]` | The restart-gated `enabled` switch, `discovery_enabled`, `discovery_port`, and `pairing_key` |

The one control bind host applies to both control-facing listeners. The shared
model directory is the canonical local model catalog; embedded-node cache
cleanup never removes those canonical files. Runtime ports must not overlap the
control or OpenAI-compatible listener ports. AIO has no setting that opens an
embedded node listener.

The existing `mantic-mind-control.toml` and `mantic-mind.toml` files continue to
configure only their standalone executables. They are not implicitly merged by
AIO.

## Control API additions

`GET /v1/nodes` remains backward compatible and adds `kind`, whose value is
`"embedded"` for `local` and `"remote"` for network nodes. The embedded entry
has an empty `url`, never includes an API key, and cannot be removed, forgotten,
or paired.

Authenticated control clients can manage either transport through these
control-side routes:

- `GET /v1/nodes/{id}/status` and `/logs?tail=N`
- `POST /v1/nodes/{id}/actions/cancel`
- `GET /v1/nodes/{id}/runtime/llama`
- `POST /v1/nodes/{id}/runtime/llama/{provision,update,check-update,switch,diagnose,recover}`

The payloads mirror the existing node runtime actions. A prompt-policy action
that can use the network must include `"allow_network": true`; offline policy
rejects it even when that field is present. Local remove/forget/pair attempts
return HTTP 409. Remote-node mutations return HTTP 403 while clustering is
disabled.

## Lifecycle and compatibility

AIO composes the same `ControlHost` and `NodeHost` service graphs used by the
standalone executables. The standalone shells add their own TUI/CLI and, for a
remote node, its HTTP API, discovery, and registration workers; AIO injects the
host-owned `NodeService` directly into local control operations and starts none
of those node networking components.

AIO owns both the control and node singleton locks. It will not start alongside
a conflicting standalone instance using the same resources. Startup brings the
embedded node online before accepting control work; a managed-runtime download
or build can continue in the background and is reported through the control UI.

On exit, AIO stops accepting new work, cancels or drains in-flight control work,
then stops node background workers and unloads inference slots. This ordering
prevents an in-flight request from outliving the embedded node.

With the default `runtime_network_policy = "prompt"`, startup only resolves an
already installed runtime. Provisioning, switching builds, recovery downloads,
and online update checks require explicit TUI confirmation, CLI
`--allow-network`, or an API payload containing `"allow_network": true`.
`auto` permits configured provisioning and periodic checks; `offline` rejects
all network-dependent runtime actions.

The remote node REST protocol, public control APIs, standalone configuration
files, and standalone executable names are unchanged. Release archives include
all three programs and all three default configuration templates.
