#!/usr/bin/env sh
# Build Ramulator 2 at a pinned commit (MIT license) for the C4 cross-check.
# Output: .cache/ramulator2 with the Python binding built for $RAMULATOR_PYTHON (default python3.14).
set -eu
COMMIT=72427a1bba3771564c4fb0e494ba02242fd1eaa7
HERE=$(CDPATH= cd -P "$(dirname "$0")" && pwd)
ROOT=$(CDPATH= cd -P "$HERE/../.." && pwd)
DEST=${RAMULATOR_DIR:-"$ROOT/.cache/ramulator2"}
PYTHON=${RAMULATOR_PYTHON:-python3.14}
command -v cmake >/dev/null || { echo "BLOCKED: cmake not found" >&2; exit 2; }
command -v "$PYTHON" >/dev/null || { echo "BLOCKED: $PYTHON not found" >&2; exit 2; }
if [ ! -d "$DEST/.git" ]; then
  git clone --quiet https://github.com/CMU-SAFARI/ramulator2.git "$DEST"
fi
cd "$DEST"
git checkout --quiet "$COMMIT"
# Apple clang requires the `template` keyword for a dependent member template call.
if git apply --check "$HERE/param_template.patch" 2>/dev/null; then
  git apply "$HERE/param_template.patch"
fi
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$(command -v "$PYTHON")" >/dev/null
cmake --build build --parallel "${JOBS:-4}" >/dev/null
echo "built Ramulator 2 $COMMIT in $DEST"
