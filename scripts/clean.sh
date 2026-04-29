#!/bin/bash
# Clean build artifacts and caches for CuFlash-Attn
# Usage: ./scripts/clean.sh

set -e

echo "🧹 Cleaning build artifacts and caches..."

# Remove build directory
if [ -d "build" ]; then
    rm -rf build/
    echo "  ✓ Removed build/"
fi

# Remove Python caches
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
echo "  ✓ Removed Python caches"

# Remove CMake cache files
find . -name "CMakeCache.txt" -delete 2>/dev/null || true
find . -name "CMakeFiles" -type d -exec rm -rf {} + 2>/dev/null || true
echo "  ✓ Removed CMake caches"

# Remove compile_commands.json (will be regenerated)
rm -f compile_commands.json 2>/dev/null || true
echo "  ✓ Removed compile_commands.json"

echo ""
echo "✅ Clean completed!"
echo ""
echo "To rebuild, run:"
echo "  cmake --preset release && cmake --build --preset release"
