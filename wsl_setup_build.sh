#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"$script_dir/wsl_setup.sh"

export VCPKG_ROOT="${VCPKG_ROOT:-$HOME/vcpkg}"

# Manifest mode needs access to the configured builtin baseline.
if [ -f "$VCPKG_ROOT/.git/shallow" ]; then
  git -C "$VCPKG_ROOT" fetch --unshallow --quiet || git -C "$VCPKG_ROOT" fetch --quiet
fi

exec "$script_dir/scripts/build.sh" --config Release --generator Ninja "$@"
