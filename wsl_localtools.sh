#!/bin/bash
# Fetch zip + unzip into ~/local without sudo (needed by vcpkg bootstrap).
set -e
mkdir -p "$HOME/local" "$HOME/debs"
cd "$HOME/debs"

echo "=== apt-get download zip unzip (no sudo) ==="
if apt-get download zip unzip 2>/dev/null; then
  echo "downloaded via apt-get download"
else
  echo "apt-get download failed; trying direct archive URLs"
  python3 - <<'PYEOF'
import urllib.request, re, sys
base="http://archive.ubuntu.com/ubuntu/pool/main/"
pkgs={"zip":"z/zip/","unzip":"u/unzip/"}
for name,sub in pkgs.items():
    idx=urllib.request.urlopen(base+sub, timeout=30).read().decode("utf-8","ignore")
    cands=re.findall(r'href="('+re.escape(name)+r'_[^"]*_amd64\.deb)"', idx)
    if not cands:
        print("NO DEB for", name); sys.exit(1)
    deb=sorted(cands)[-1]
    urllib.request.urlretrieve(base+sub+deb, deb)
    print("fetched", deb)
PYEOF
fi

echo "=== extract debs into ~/local (no sudo) ==="
for d in *.deb; do dpkg-deb -x "$d" "$HOME/local"; echo "extracted $d"; done

echo "=== verify ==="
export PATH="$HOME/local/usr/bin:$PATH"
command -v zip && zip --version | head -2
command -v unzip && unzip -v | head -1
echo "=== LOCALTOOLS DONE ==="
