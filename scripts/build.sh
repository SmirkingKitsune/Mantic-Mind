#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(CDPATH= cd -- "$script_dir/.." && pwd)"

config="${MM_BUILD_CONFIG:-Release}"
build_dir=""
generator="${CMAKE_GENERATOR:-}"
install_prefix=""
use_vcpkg="auto"
target_arch="${MM_TARGET_ARCH:-}"
triplet="${VCPKG_DEFAULT_TRIPLET:-}"
cmake_extra_args=()

usage() {
  cat <<'EOF'
Usage: scripts/build.sh [options] [-- <extra cmake configure args>]

Options:
  --config <name>          Build configuration: Release, Debug, RelWithDebInfo.
  --debug                  Shortcut for --config Debug.
  --build-dir <path>       Build directory. Defaults to build/<os>-<arch>-<config>.
  --arch <name>            Target architecture: native, x64, x86_64, arm64, aarch64.
  -G, --generator <name>   CMake generator to use.
  --vcpkg-root <path>      vcpkg root. Defaults to VCPKG_ROOT or vcpkg on PATH.
  --triplet <name>         vcpkg target triplet.
  --no-vcpkg               Do not pass a vcpkg toolchain file.
  --install-prefix <path>  Run cmake --install after building.
  -h, --help               Show this help.

Examples:
  scripts/build.sh
  scripts/build.sh --debug
  scripts/build.sh --arch aarch64
  scripts/build.sh --config Release --install-prefix dist
  scripts/build.sh --generator Ninja -- -DBUILD_TESTING=OFF
EOF
}

lower() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

detect_os() {
  case "$(uname -s)" in
    Linux*) printf 'linux\n' ;;
    Darwin*) printf 'darwin\n' ;;
    CYGWIN*|MINGW*|MSYS*) printf 'windows\n' ;;
    *) lower "$(uname -s)" ;;
  esac
}

normalize_arch() {
  case "$(lower "$1")" in
    ""|native) lower "$(uname -m)" ;;
    x86_64|amd64|x64) printf 'x64\n' ;;
    aarch64|arm64|arm64-v8a) printf 'aarch64\n' ;;
    armv7|armv7l) printf 'armv7\n' ;;
    *) lower "$1" ;;
  esac
}

vcpkg_arch_name() {
  case "$1" in
    aarch64) printf 'arm64\n' ;;
    x86_64|x64) printf 'x64\n' ;;
    armv7|armv7l) printf 'arm\n' ;;
    *) printf '%s\n' "$1" ;;
  esac
}

vcpkg_os_name() {
  case "$1" in
    darwin) printf 'osx\n' ;;
    *) printf '%s\n' "$1" ;;
  esac
}

has_cmake_arg() {
  local key="$1"
  local arg
  for arg in "${cmake_extra_args[@]}"; do
    case "$arg" in
      -D"$key"=*|-D"$key":*) return 0 ;;
    esac
  done
  return 1
}

detect_jobs() {
  if [ -n "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]; then
    printf '%s\n' "$CMAKE_BUILD_PARALLEL_LEVEL"
  elif command -v getconf >/dev/null 2>&1 && getconf _NPROCESSORS_ONLN >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN
  elif command -v sysctl >/dev/null 2>&1 && sysctl -n hw.ncpu >/dev/null 2>&1; then
    sysctl -n hw.ncpu
  elif command -v nproc >/dev/null 2>&1; then
    nproc
  else
    printf '2\n'
  fi
}

detect_vcpkg_root() {
  if [ -n "${VCPKG_ROOT:-}" ] && [ -f "$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" ]; then
    printf '%s\n' "$VCPKG_ROOT"
    return 0
  fi

  if command -v vcpkg >/dev/null 2>&1; then
    local exe dir parent
    exe="$(command -v vcpkg)"
    dir="$(CDPATH= cd -- "$(dirname -- "$exe")" && pwd)"
    parent="$(CDPATH= cd -- "$dir/.." && pwd)"
    if [ -f "$dir/scripts/buildsystems/vcpkg.cmake" ]; then
      printf '%s\n' "$dir"
      return 0
    fi
    if [ -f "$parent/scripts/buildsystems/vcpkg.cmake" ]; then
      printf '%s\n' "$parent"
      return 0
    fi
  fi

  local candidate
  for candidate in /c/vcpkg /c/src/vcpkg /c/tools/vcpkg /mnt/c/vcpkg /mnt/c/src/vcpkg /mnt/c/tools/vcpkg; do
    if [ -f "$candidate/scripts/buildsystems/vcpkg.cmake" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

normalize_path_text() {
  if [ -z "$1" ]; then
    return 0
  fi

  if command -v cygpath >/dev/null 2>&1; then
    cygpath -m "$1" | sed 's:/*$::'
  else
    printf '%s\n' "$1" | sed 's:\\:/:g; s:/*$::'
  fi
}

cmake_cache_value() {
  local cache_path="$1"
  local key="$2"

  if [ ! -f "$cache_path" ]; then
    return 0
  fi

  sed -n "s|^$key:[^=]*=||p" "$cache_path" | head -n 1
}

cmake_cache_type() {
  local cache_path="$1"
  local key="$2"

  if [ ! -f "$cache_path" ]; then
    return 0
  fi

  sed -n "s|^$key:\\([^=]*\\)=.*|\\1|p" "$cache_path" | head -n 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)
      shift
      config="${1:?missing value for --config}"
      ;;
    --debug)
      config="Debug"
      ;;
    --build-dir)
      shift
      build_dir="${1:?missing value for --build-dir}"
      ;;
    --arch)
      shift
      target_arch="${1:?missing value for --arch}"
      ;;
    -G|--generator)
      shift
      generator="${1:?missing value for --generator}"
      ;;
    --vcpkg-root)
      shift
      export VCPKG_ROOT="${1:?missing value for --vcpkg-root}"
      ;;
    --triplet)
      shift
      triplet="${1:?missing value for --triplet}"
      ;;
    --no-vcpkg)
      use_vcpkg="no"
      ;;
    --install-prefix)
      shift
      install_prefix="${1:?missing value for --install-prefix}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      cmake_extra_args+=("$@")
      break
      ;;
    -D*)
      cmake_extra_args+=("$1")
      ;;
    *)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [ -z "$build_dir" ]; then
  os="$(detect_os)"
  arch="$(normalize_arch "$target_arch")"
  config_slug="$(lower "$config")"
  build_dir="$source_dir/build/$os-$arch-$config_slug"
fi

os="${os:-$(detect_os)}"
arch="${arch:-$(normalize_arch "$target_arch")}"

cmake_configure=(cmake -S "$source_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE="$config")

if [ -n "$generator" ]; then
  cmake_configure+=(-G "$generator")
fi

host_arch="$(normalize_arch native)"

if [ "$os" = "darwin" ] && [ "$arch" = "aarch64" ]; then
  cmake_configure+=(-DCMAKE_OSX_ARCHITECTURES=arm64)
fi

if [ "$os" = "linux" ] && [ "$arch" = "aarch64" ] && [ "$host_arch" != "aarch64" ]; then
  if command -v aarch64-linux-gnu-gcc >/dev/null 2>&1 && command -v aarch64-linux-gnu-g++ >/dev/null 2>&1; then
    has_cmake_arg CMAKE_SYSTEM_NAME || cmake_configure+=(-DCMAKE_SYSTEM_NAME=Linux)
    has_cmake_arg CMAKE_SYSTEM_PROCESSOR || cmake_configure+=(-DCMAKE_SYSTEM_PROCESSOR=aarch64)
    has_cmake_arg CMAKE_C_COMPILER || cmake_configure+=(-DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc)
    has_cmake_arg CMAKE_CXX_COMPILER || cmake_configure+=(-DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++)
  elif has_cmake_arg CMAKE_CXX_COMPILER || has_cmake_arg CMAKE_TOOLCHAIN_FILE; then
    has_cmake_arg CMAKE_SYSTEM_NAME || cmake_configure+=(-DCMAKE_SYSTEM_NAME=Linux)
    has_cmake_arg CMAKE_SYSTEM_PROCESSOR || cmake_configure+=(-DCMAKE_SYSTEM_PROCESSOR=aarch64)
  else
    printf 'AArch64 Linux cross compiler not detected. Install g++-aarch64-linux-gnu or pass a cross toolchain/compiler after --.\n' >&2
    exit 2
  fi
fi

if [ "$os" = "windows" ] && [ "$arch" = "aarch64" ] && [ -n "$generator" ]; then
  case "$generator" in
    *Visual\ Studio*) cmake_configure+=(-A ARM64) ;;
  esac
fi

expected_toolchain_file=""
if [ "$use_vcpkg" != "no" ]; then
  if vcpkg_root="$(detect_vcpkg_root)"; then
    expected_toolchain_file="$vcpkg_root/scripts/buildsystems/vcpkg.cmake"
    cmake_configure+=(-DCMAKE_TOOLCHAIN_FILE="$expected_toolchain_file")
    if [ -z "$triplet" ] && [ -n "$target_arch" ] && [ "$(lower "$target_arch")" != "native" ]; then
      triplet="$(vcpkg_arch_name "$arch")-$(vcpkg_os_name "$os")"
    fi
    if [ -n "$triplet" ]; then
      cmake_configure+=(-DVCPKG_TARGET_TRIPLET="$triplet")
    fi
  else
    printf 'vcpkg not detected; configuring without a toolchain. Dependencies must be discoverable by CMake.\n' >&2
  fi
fi

cmake_configure+=("${cmake_extra_args[@]}")

jobs="$(detect_jobs)"

cache_toolchain_file="$(cmake_cache_value "$build_dir/CMakeCache.txt" CMAKE_TOOLCHAIN_FILE)"
cache_toolchain_type="$(cmake_cache_type "$build_dir/CMakeCache.txt" CMAKE_TOOLCHAIN_FILE)"
if [ "$(normalize_path_text "$cache_toolchain_file")" != "$(normalize_path_text "$expected_toolchain_file")" ] ||
   { [ -n "$expected_toolchain_file" ] && [ "$cache_toolchain_type" = "UNINITIALIZED" ]; }; then
  printf '==> Refreshing CMake cache: toolchain changed\n'
  cmake_configure=(cmake --fresh "${cmake_configure[@]:1}")
fi

printf '==> Configure: %s\n' "$build_dir"
"${cmake_configure[@]}"

printf '==> Build: %s (%s, %s jobs)\n' "$build_dir" "$config" "$jobs"
cmake --build "$build_dir" --config "$config" --parallel "$jobs"

if [ -n "$install_prefix" ]; then
  printf '==> Install: %s\n' "$install_prefix"
  cmake --install "$build_dir" --config "$config" --prefix "$install_prefix"
fi
