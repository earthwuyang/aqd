#!/usr/bin/env bash
# ==============================================================
#  build.sh – one-shot wrapper for the “router” project
#  * Defaults to C++11  (override with CXX_STD=14/17/20 if you like)
#  * Knows where nlohmann/json.hpp lives
#  * Adds first-class support for LIGHTGBM_ROOT / FANN_ROOT
# ==============================================================

set -euo pipefail

# ---------- user-overridable variables ------------------------
CMAKE_BIN="${CMAKE_BIN:-cmake}"
BUILD_DIR="${BUILD_DIR:-build}"
CXX_STD="${CXX_STD:-11}"                 # ← keep C++11 as the default

# nlohmann/json single-header directory
JSON_ROOT_DEFAULT="/home/wuy/software/json-develop/single_include"
JSON_ROOT="${JSON_ROOT:-${JSON_ROOT_DEFAULT}}"

# ---------- third-party default prefixes (you can export instead) ----
LIGHTGBM_ROOT_DEFAULT="/home/wuy/software/LightGBM"
FANN_ROOT_DEFAULT="/home/wuy/software/fann"

LIGHTGBM_ROOT="${LIGHTGBM_ROOT:-${LIGHTGBM_ROOT_DEFAULT}}"
FANN_ROOT="${FANN_ROOT:-${FANN_ROOT_DEFAULT}}"

# --------------------------------------------------------------
mkdir -p "${BUILD_DIR}"
cd       "${BUILD_DIR}"

declare -a CFG
CFG+=("-DCMAKE_BUILD_TYPE=Release")
CFG+=("-DCMAKE_CXX_STANDARD=${CXX_STD}")
CFG+=("-DTHREADS_PREFER_PTHREAD_FLAG=ON")

# optional components
CFG+=("-DWITH_LIGHTGBM=${WITH_LIGHTGBM:-ON}")
CFG+=("-DWITH_FANN=${WITH_FANN:-ON}")

# header locations
CFG+=("-DJSON_ROOT=${JSON_ROOT}")

# honour explicit or default prefixes
[[ -n "${LIGHTGBM_ROOT}" ]] && CFG+=("-DLIGHTGBM_ROOT=${LIGHTGBM_ROOT}")
[[ -n "${FANN_ROOT}"      ]] && CFG+=("-DFANN_ROOT=${FANN_ROOT}")

# forward additional -D flags passed to build.sh
CFG+=("$@")

echo "==> configuring (${BUILD_DIR}) …"
"${CMAKE_BIN}" .. "${CFG[@]}"

echo "==> building …"
if "${CMAKE_BIN}" --build . --parallel "$(nproc)" 2>/dev/null; then
    :
else
    "${CMAKE_BIN}" --build . -- -j"$(nproc)"
fi

echo "==> done – binary is at $(pwd)/router"

# --------- (optional) linker-path hint ------------------------
for d in "${LIGHTGBM_ROOT}" "${FANN_ROOT}"; do
    if [[ -d "$d" && ! "$(ldconfig -v 2>/dev/null | grep -F "$d")" ]]; then
        echo "ℹ️  If runtime can’t find lib_lightgbm.so or libfann.so:"
        echo "   export LD_LIBRARY_PATH=\"$d:\$LD_LIBRARY_PATH\""
        break
    fi
done
