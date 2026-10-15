#!/usr/bin/env bash
# Copy the locally built "library" (pre-generated concept maps and timelines) to the server.
#
#   1. Build locally, pinned to the SAME model as the server so the cache keys match:
#        VENICE_MODEL=z-ai-glm-5-3-flash python scripts/prewarm_cache.py
#   2. deploy/sync-library.sh
#
# Why not build on the server? The app container is memory-capped (so it can never take your site down);
# a second Python process with PyTorch inside it gets killed. Building on your Mac avoids that, and cached
# objects are served with no AI call. Content only: no keys or secrets are in these files.
set -euo pipefail
HOST="${DEPLOY_HOST:-ladanio}"; KEY="${DEPLOY_KEY:-$HOME/.ssh/id_ed25519}"
SSH="ssh -o BatchMode=yes -o IdentitiesOnly=yes -i $KEY"
cd "$(dirname "$0")/.."
# (macOS rsync has no --chown: fix ownership afterwards; uid 1000 = the app user inside the container)
rsync -az -e "$SSH" --include='*.json' --exclude='*' artifacts/generated/ "$HOST":/var/lib/discover-study/generated/
$SSH "$HOST" 'chown -R 1000:1000 /var/lib/discover-study/generated; echo "library on server: $(ls /var/lib/discover-study/generated | wc -l) objects"'
