#!/bin/bash
# Stage run directories + configs for control and node.
set -e
SRC=/mnt/z/AI/Ubiquitious-Memories/Mantic-Mind
BUILD="$HOME/mm-build"
RUN="$HOME/mm-run"

rm -rf "$RUN"
mkdir -p "$RUN/control/logs" "$RUN/control/data" "$RUN/control/models"
mkdir -p "$RUN/node/logs" "$RUN/node/data" "$RUN/node/models"

# Configs (strip any CR).
sed 's/\r$//' "$SRC/wsl_run/mantic-mind-control.toml" > "$RUN/control/mantic-mind-control.toml"
sed 's/\r$//' "$SRC/wsl_run/mantic-mind.toml"        > "$RUN/node/mantic-mind.toml"

# Symlink binaries in place.
ln -sf "$BUILD/src/control/mantic-mind-control" "$RUN/control/mantic-mind-control"
ln -sf "$BUILD/src/node/mantic-mind"            "$RUN/node/mantic-mind"

echo "=== run tree ==="
ls -l "$RUN/control" "$RUN/node"
echo "=== PREP DONE ==="
