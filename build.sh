#!/usr/bin/env bash
set -e

echo "==> [1/5] Installing Python dependencies via Poetry..."
poetry install --no-interaction

echo "==> [2/5] Fetching Prisma query engine binaries..."
poetry run python -m prisma py fetch

echo "==> [3/5] Locating downloaded Prisma query engine..."
CACHE_DIR="/opt/render/.cache/prisma-python/binaries"

# Debug: show exactly what was downloaded
echo "    Contents of cache directory:"
find "$CACHE_DIR" -type f 2>/dev/null || echo "    (cache dir not found)"

# Find the query engine binary (could be named query-engine-* or prisma-query-engine-*)
ENGINE_FILE=$(find "$CACHE_DIR" -type f -name "*query-engine*" | head -1)

if [ -z "$ENGINE_FILE" ]; then
    echo "ERROR: Could not find query engine binary in $CACHE_DIR"
    exit 1
fi

echo "    Found engine: $ENGINE_FILE"

echo "==> [4/5] Copying query engine to project root..."
# Copy with the exact name Prisma expects
cp "$ENGINE_FILE" ./prisma-query-engine-debian-openssl-3.0.x
chmod +x ./prisma-query-engine-debian-openssl-3.0.x
echo "    Copied to ./prisma-query-engine-debian-openssl-3.0.x"

echo "==> [5/5] Generating Prisma client..."
poetry run python -m prisma generate

echo "==> Build complete!"
