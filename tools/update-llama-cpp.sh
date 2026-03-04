#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

REPO_URL="${LLAMA_CPP_REPO_URL:-https://github.com/ggml-org/llama.cpp.git}"
REPO_DIR="${LLAMA_CPP_DIR:-$PROJECT_ROOT/external/llama.cpp}"
DO_BUILD=0
CUDA="${LLAMA_CPP_CUDA:-auto}"
CONFIG="${LLAMA_CPP_BUILD_CONFIG:-Release}"
GENERATOR="${LLAMA_CPP_GENERATOR:-}"
JOBS="${LLAMA_CPP_JOBS:-}"

usage() {
  cat <<'EOF'
Usage: tools/update-llama-cpp.sh [options]

Clone or fast-forward pull llama.cpp, and optionally build llama-server.

Options:
  --repo-url <url>        Git URL (default: github.com/ggml-org/llama.cpp.git)
  --repo-dir <path>       Repo checkout path (default: external/llama.cpp)
  --build                 Build llama-server after updating
  --cuda                  Force-enable CUDA GPU support (-DGGML_CUDA=ON)
  --no-cuda               Force-disable CUDA (skip auto-detection)
  --config <name>         Build config (default: Release)
  --generator <name>      CMake generator (optional)
  --jobs <n>              Parallel build jobs (optional)
  --help                  Show this help

Environment overrides:
  LLAMA_CPP_REPO_URL, LLAMA_CPP_DIR, LLAMA_CPP_CUDA,
  LLAMA_CPP_BUILD_CONFIG, LLAMA_CPP_GENERATOR, LLAMA_CPP_JOBS
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-url) REPO_URL="${2:?missing value}"; shift 2 ;;
    --repo-dir) REPO_DIR="${2:?missing value}"; shift 2 ;;
    --build) DO_BUILD=1; shift ;;
    --cuda) CUDA=1; shift ;;
    --no-cuda) CUDA=0; shift ;;
    --config) CONFIG="${2:?missing value}"; shift 2 ;;
    --generator) GENERATOR="${2:?missing value}"; shift 2 ;;
    --jobs) JOBS="${2:?missing value}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

echo "[llama.cpp] repo url : $REPO_URL"
echo "[llama.cpp] repo dir : $REPO_DIR"

if [[ -d "$REPO_DIR/.git" ]]; then
  echo "[llama.cpp] existing checkout found; pulling latest..."
  git -C "$REPO_DIR" fetch --tags origin
  git -C "$REPO_DIR" pull --ff-only
else
  echo "[llama.cpp] cloning fresh checkout..."
  mkdir -p "$(dirname "$REPO_DIR")"
  git clone "$REPO_URL" "$REPO_DIR"
fi

COMMIT="$(git -C "$REPO_DIR" rev-parse --short HEAD)"
echo "[llama.cpp] at commit $COMMIT"

if [[ "$DO_BUILD" -eq 0 ]]; then
  echo "[llama.cpp] update complete (build skipped)"
  exit 0
fi

BUILD_DIR="$REPO_DIR/build"
echo "[llama.cpp] configuring CMake in $BUILD_DIR..."

# Resolve CUDA: explicit flag > auto-detect via nvcc
if [[ "$CUDA" == "auto" ]]; then
  if command -v nvcc &>/dev/null; then
    CUDA=1
    echo "[llama.cpp] CUDA toolkit detected ($(command -v nvcc)) — enabling CUDA automatically"
  else
    CUDA=0
  fi
fi

CMAKE_ARGS=(-S "$REPO_DIR" -B "$BUILD_DIR" -DLLAMA_BUILD_SERVER=ON)
if [[ "$CUDA" -eq 1 ]]; then
  CMAKE_ARGS+=(-DGGML_CUDA=ON)
  echo "[llama.cpp] CUDA enabled"
else
  echo "[llama.cpp] CUDA not enabled (no CUDA toolkit found; pass --cuda or set LLAMA_CPP_CUDA=1 to force)"
fi
if [[ -n "$GENERATOR" ]]; then
  CMAKE_ARGS+=(-G "$GENERATOR")
fi
cmake "${CMAKE_ARGS[@]}"

BUILD_ARGS=(--build "$BUILD_DIR" --config "$CONFIG" --target llama-server)
if [[ -n "$JOBS" ]]; then
  BUILD_ARGS+=(--parallel "$JOBS")
fi

echo "[llama.cpp] building llama-server..."
cmake "${BUILD_ARGS[@]}"

if [[ -f "$BUILD_DIR/bin/llama-server" ]]; then
  OUT_PATH="$BUILD_DIR/bin/llama-server"
elif [[ -f "$BUILD_DIR/bin/$CONFIG/llama-server" ]]; then
  OUT_PATH="$BUILD_DIR/bin/$CONFIG/llama-server"
elif [[ -f "$BUILD_DIR/bin/llama-server.exe" ]]; then
  OUT_PATH="$BUILD_DIR/bin/llama-server.exe"
elif [[ -f "$BUILD_DIR/bin/$CONFIG/llama-server.exe" ]]; then
  OUT_PATH="$BUILD_DIR/bin/$CONFIG/llama-server.exe"
else
  OUT_PATH="(not found in expected build/bin paths)"
fi

echo "[llama.cpp] build complete"
echo "[llama.cpp] llama-server path: $OUT_PATH"
echo "[llama.cpp] set MM_LLAMA_PATH to this path for mantic-mind node startup"

if [[ "$OUT_PATH" == "(not found in expected build/bin paths)" ]]; then
  echo "[llama.cpp] error: build succeeded but llama-server binary was not found" >&2
  exit 3
fi
