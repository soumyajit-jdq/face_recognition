#!/usr/bin/env bash
set -e

# Define a persistent cache directory inside the project root
export PRISMA_BINARY_CACHE_DIR="/opt/render/project/src/.prisma-binaries"

echo "==> [1/4] Installing Python dependencies via Poetry..."
poetry install --no-interaction

echo "==> [2/4] Fetching Prisma query engine binaries..."
poetry run python -m prisma py fetch

echo "==> [3/4] Ensuring binaries are executable..."
chmod -R +x "$PRISMA_BINARY_CACHE_DIR"

echo "==> [4/4] Generating Prisma client..."
poetry run python -m prisma generate

echo "==> Build complete!"
