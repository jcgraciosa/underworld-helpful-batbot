#!/bin/bash
# Fly.io entrypoint — clone UW3 repo on first deploy, then start the app.

set -e

REPO_DIR="/app/content_cache/underworld3"
REPO_URL="https://github.com/underworldcode/underworld3.git"

if [ ! -d "$REPO_DIR/.git" ]; then
    echo "Cloning UW3 repository (first deploy)..."
    git clone --depth 1 --single-branch --branch main "$REPO_URL" "$REPO_DIR"
    echo "Clone complete."
else
    echo "UW3 repository already present, pulling latest..."
    git -C "$REPO_DIR" pull --ff-only origin main || echo "Pull failed (non-critical), using existing copy."
fi

export BOT_REPO_PATH="$REPO_DIR"

echo "Starting HelpfulBatBot..."
exec python HelpfulBat_app.py
