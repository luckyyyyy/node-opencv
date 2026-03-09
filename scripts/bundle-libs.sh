#!/usr/bin/env bash
# bundle-libs.sh — copy OpenCV shared libraries into lib/ and patch RPATH
# Usage: bash scripts/bundle-libs.sh <path-to-.node-file> [<opencv-lib-dir>]
#
# On Linux:  copies libopencv_*.so.NNN → lib/, patches RUNPATH to $ORIGIN/lib
# On macOS:  copies libopencv_*.dylib  → lib/, patches install names

set -euo pipefail

NODE_FILE="${1:?Usage: $0 <path-to-.node> [<opencv-lib-dir>]}"
LIB_DIR="$(dirname "$NODE_FILE")/lib"
mkdir -p "$LIB_DIR"

# ── Gather OpenCV library paths ───────────────────────────────────────────────
if [[ "$(uname -s)" == "Darwin" ]]; then
  # On macOS, find dylibs linked into the .node
  OPENCV_LIBS=$(otool -L "$NODE_FILE" \
    | awk '/libopencv/{print $1}')
else
  # On Linux, find .so paths via ldd
  OPENCV_LIBS=$(ldd "$NODE_FILE" \
    | awk '/libopencv/{print $3}')
fi

if [[ -z "$OPENCV_LIBS" ]]; then
  echo "ERROR: no libopencv_* found in $NODE_FILE"
  exit 1
fi

echo "Found OpenCV libraries:"
echo "$OPENCV_LIBS"

# ── Copy libs ─────────────────────────────────────────────────────────────────
for lib_path in $OPENCV_LIBS; do
  lib_name=$(basename "$lib_path")
  dest="$LIB_DIR/$lib_name"
  echo "Copying $lib_path → $dest"
  cp -L "$lib_path" "$dest"
  chmod 755 "$dest"
done

# ── Linux: patchelf RUNPATH ───────────────────────────────────────────────────
if [[ "$(uname -s)" == "Linux" ]]; then
  if ! command -v patchelf &>/dev/null; then
    echo "Installing patchelf..."
    if command -v apt-get &>/dev/null; then
      sudo apt-get install -y -q patchelf
    elif command -v conda &>/dev/null; then
      conda install -y patchelf
    else
      echo "ERROR: patchelf not found. Install it manually."
      exit 1
    fi
  fi

  echo "Patching RUNPATH of $NODE_FILE → \$ORIGIN/lib"
  patchelf --set-rpath '$ORIGIN/lib' "$NODE_FILE"
  echo "RUNPATH after patch:"
  readelf -d "$NODE_FILE" | grep -E "RPATH|RUNPATH"
fi

# ── macOS: fix install names ──────────────────────────────────────────────────
if [[ "$(uname -s)" == "Darwin" ]]; then
  # For each bundled library, rewrite the .node's reference to use @loader_path/lib/
  for lib_path in $OPENCV_LIBS; do
    lib_name=$(basename "$lib_path")
    echo "Rewriting macOS install name: $lib_path → @loader_path/lib/$lib_name"
    install_name_tool -change "$lib_path" "@loader_path/lib/$lib_name" "$NODE_FILE"
  done

  # Also fix each bundled dylib's own install name and its references to sibling dylibs
  for lib_path in $OPENCV_LIBS; do
    lib_name=$(basename "$lib_path")
    dest="$LIB_DIR/$lib_name"
    # Fix id
    install_name_tool -id "@loader_path/$lib_name" "$dest"
    # Fix cross-references between bundled opencv dylibs
    for dep_path in $OPENCV_LIBS; do
      dep_name=$(basename "$dep_path")
      if otool -L "$dest" | grep -q "$dep_path"; then
        install_name_tool -change "$dep_path" "@loader_path/$dep_name" "$dest"
      fi
    done
    # codesign on macOS Sonoma+
    if command -v codesign &>/dev/null; then
      codesign --force --sign - "$dest" 2>/dev/null || true
    fi
  done

  if command -v codesign &>/dev/null; then
    codesign --force --sign - "$NODE_FILE" 2>/dev/null || true
  fi
fi

echo "Done. Contents of $LIB_DIR:"
ls -lh "$LIB_DIR"
