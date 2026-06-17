#!/bin/bash
# Bootstrap build prerequisites + vcpkg inside WSL Ubuntu.
set -e
echo "=== prereq check ==="
need=()
for t in git ninja-build pkg-config zip unzip curl tar make cmake g++; do
  bin=$t
  [ "$t" = "ninja-build" ] && bin=ninja
  if ! command -v "$bin" >/dev/null 2>&1; then need+=("$t"); fi
done
# perl + autoconf/automake/libtool needed by some vcpkg ports (openssl, etc.)
for t in perl autoconf automake libtool; do
  command -v "$t" >/dev/null 2>&1 || need+=("$t")
done
if [ ${#need[@]} -gt 0 ]; then
  echo "installing: ${need[*]}"
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${need[@]}" linux-libc-dev >/dev/null
else
  echo "all prereqs present"
fi

echo "=== vcpkg ==="
if [ ! -x "$HOME/vcpkg/vcpkg" ]; then
  rm -rf "$HOME/vcpkg"
  git clone --depth 1 https://github.com/microsoft/vcpkg "$HOME/vcpkg"
  "$HOME/vcpkg/bootstrap-vcpkg.sh" -disableMetrics
else
  echo "vcpkg already bootstrapped"
fi
echo "VCPKG OK: $($HOME/vcpkg/vcpkg version | head -1)"
echo "=== SETUP DONE ==="
