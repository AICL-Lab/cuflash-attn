#!/usr/bin/env bash
# Run clang-tidy static analysis on the host-compilable C++ sources.
#
# CUDA .cu translation units are intentionally excluded: clang-tidy parses
# them as plain host C++ and chokes on device-side builtins (__global__,
# <<<...>>>, shared-memory intrinsics), producing noise rather than signal.
# To analyze device code, use a CUDA-aware clang locally.
#
# Usage: ./scripts/run_clang_tidy.sh [build-dir]
#   build-dir defaults to "build". A compile database (compile_commands.json)
#   is required; the script configures one if missing (needs the CUDA toolkit).

set -euo pipefail

BUILD_DIR="${1:-build}"
CLANG_TIDY="${CLANG_TIDY:-clang-tidy}"

if ! command -v "$CLANG_TIDY" >/dev/null 2>&1; then
    echo "error: $CLANG_TIDY not found on PATH" >&2
    exit 1
fi

if [ ! -f "$BUILD_DIR/compile_commands.json" ]; then
    echo "compile_commands.json not found; configuring into $BUILD_DIR ..."
    cmake -B "$BUILD_DIR" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_TESTS=ON >/dev/null
fi

# Host-only sources that are part of the main build's compile database.
HOST_SOURCES=(
    tests/test_main.cpp
    tests/integration/test_api_smoke.cpp
)

status=0
for src in "${HOST_SOURCES[@]}"; do
    echo "==> clang-tidy $src"
    if ! "$CLANG_TIDY" "$src" -p "$BUILD_DIR" --quiet; then
        status=1
    fi
done

if [ "$status" -eq 0 ]; then
    echo "clang-tidy: no findings"
fi
exit "$status"
