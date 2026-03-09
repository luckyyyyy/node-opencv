#!/usr/bin/env bash
# bundle-libs.sh — copy OpenCV shared libraries into lib/ and patch load paths.
# Usage: bash scripts/bundle-libs.sh <path-to-.node-file>
#
# Linux:  copies libopencv_*.so.NNN into lib/, sets RUNPATH to $ORIGIN/lib
# macOS:  resolves @rpath/libopencv_*.dylib, copies to lib/,
#         adds @loader_path/lib to .node RPATH and @loader_path to each dylib

set -euo pipefail

NODE_FILE="${1:?Usage: $0 <path-to-.node>}"
LIB_DIR="$(dirname "$(realpath "$NODE_FILE")")/lib"
mkdir -p "$LIB_DIR"

# ── macOS ─────────────────────────────────────────────────────────────────────
if [[ "$(uname -s)" == "Darwin" ]]; then

  # 1. Read all RPATH entries embedded in the .node binary
  RPATHS=$(otool -l "$NODE_FILE" \
    | awk '/LC_RPATH/{f=1} f && /^ *path /{print $2; f=0}')
  echo "RPATHs: $RPATHS"

  # 2. For each @rpath/libopencv_*.dylib, resolve to an absolute path
  #    by searching the RPATH list.
  declare -a ORIG_REFS=()   # the original @rpath/... string
  declare -a ABS_PATHS=()   # the resolved absolute path

  while IFS= read -r ref; do
    abs="$ref"
    if [[ "$ref" == @rpath/* ]]; then
      lib_name="${ref#@rpath/}"
      for rpath in $RPATHS; do
        if [[ -f "$rpath/$lib_name" ]]; then
          abs="$rpath/$lib_name"
          break
        fi
      done
    fi
    if [[ "$abs" == @rpath/* ]]; then
      echo "WARNING: Cannot resolve $ref — skipping"
      continue
    fi
    ORIG_REFS+=("$ref")
    ABS_PATHS+=("$abs")
  done < <(otool -L "$NODE_FILE" | tail -n +2 | awk '{print $1}' | grep 'libopencv')

  echo "Bundling ${#ABS_PATHS[@]} OpenCV dylibs..."

  # 3. Copy each dylib into lib/
  for i in "${!ABS_PATHS[@]}"; do
    src="${ABS_PATHS[$i]}"
    dest="$LIB_DIR/$(basename "$src")"
    echo "  $src → $dest"
    cp -fL "$src" "$dest"
    chmod 755 "$dest"
  done

  # 4. Add @loader_path/lib to the .node's own RPATH so the existing
  #    @rpath/libopencv_*.dylib entries resolve against our bundled lib/.
  install_name_tool -add_rpath "@loader_path/lib" "$NODE_FILE" 2>/dev/null || \
    echo "(add_rpath skipped — already present)"

  # 5. For each bundled dylib, add @loader_path to its RPATH so that when
  #    it pulls in sibling opencv dylibs (also @rpath/…) they are found in lib/.
  for i in "${!ABS_PATHS[@]}"; do
    dest="$LIB_DIR/$(basename "${ABS_PATHS[$i]}")"
    install_name_tool -add_rpath "@loader_path" "$dest" 2>/dev/null || true
    codesign --force --sign - "$dest" 2>/dev/null || true
  done

  # 6. Re-sign the .node itself after modifying its load commands
  codesign --force --sign - "$NODE_FILE" 2>/dev/null || true

fi

# ── Linux ─────────────────────────────────────────────────────────────────────
if [[ "$(uname -s)" == "Linux" ]]; then

  OPENCV_LIBS=$(ldd "$NODE_FILE" | awk '/libopencv/{print $3}')
  if [[ -z "$OPENCV_LIBS" ]]; then
    echo "ERROR: no libopencv_* found in $NODE_FILE"
    exit 1
  fi

  echo "Bundling OpenCV libs..."
  for lib_path in $OPENCV_LIBS; do
    dest="$LIB_DIR/$(basename "$lib_path")"
    echo "  $lib_path → $dest"
    cp -fL "$lib_path" "$dest"
    chmod 755 "$dest"
  done

  if ! command -v patchelf &>/dev/null; then
    echo "ERROR: patchelf not found"
    exit 1
  fi

  echo "Setting RUNPATH → \$ORIGIN/lib"
  patchelf --set-rpath '$ORIGIN/lib' "$NODE_FILE"
  readelf -d "$NODE_FILE" | grep -E "RPATH|RUNPATH"

fi

echo "Done. lib/ contents:"
ls -lh "$LIB_DIR"
