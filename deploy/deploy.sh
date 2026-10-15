#!/usr/bin/env bash
# Deploy (or update) Discover Canada Study on the droplet, straight from git.
#
#   deploy/deploy.sh --init     first time: folders, Caddy path block, then deploy
#   deploy/deploy.sh            update: pull origin/main, rebuild the image, swap the container
#
# Run it from your laptop. It never reads or copies your local .env. The Venice key lives only in
# /etc/discover-study.env ON THE SERVER, which you create yourself (see deploy/README.md).
#
#   DEPLOY_HOST (default: ladanio)   DEPLOY_KEY (default: ~/.ssh/id_ed25519)   DEPLOY_BRANCH (default: main)
#   FORCE=1  skip the low-memory safety check (don't, unless you know why)
set -euo pipefail

HOST="${DEPLOY_HOST:-ladanio}"
KEY="${DEPLOY_KEY:-$HOME/.ssh/id_ed25519}"
BRANCH="${DEPLOY_BRANCH:-main}"
INIT=0; [[ "${1:-}" == "--init" ]] && INIT=1
SNIPPET="$(cd "$(dirname "$0")" && pwd)/caddy-discover-study.snippet"

ssh_run() { ssh -o BatchMode=yes -o IdentitiesOnly=yes -i "$KEY" "$HOST" "$@"; }

echo "==> deploying branch '$BRANCH' to $HOST (init=$INIT)"
[[ $INIT -eq 1 ]] && SNIPPET_B64="$(base64 < "$SNIPPET" | tr -d '\n')" || SNIPPET_B64=""

ssh_run "INIT=$INIT BRANCH=$BRANCH FORCE=${FORCE:-0} SNIPPET_B64=$SNIPPET_B64 bash -s" <<'REMOTE'
set -euo pipefail
APP=/opt/discover-canada-semantic-tutor
REPO=https://github.com/ladangol/discover-canada-semantic-tutor.git
ENVFILE=/etc/discover-study.env
DATA=/var/lib/discover-study
NAME=discover-study
PORT=8502

# --- 0. safety: never put this on a box that is too small (the OOM killer could take your site down) -------
total_mb=$(awk '/MemTotal/ {printf "%d", $2/1024}' /proc/meminfo)
if [[ "$total_mb" -lt 1800 && "${FORCE:-0}" != "1" ]]; then
  echo "!! This server has only ${total_mb} MB RAM. The app needs ~700 MB on top of your site; refusing to deploy."
  echo "   Resize the droplet to 2 GB or more (see deploy/README.md), then run this again."
  exit 2
fi

# --- 1. folders + code -------------------------------------------------------------------------------------
mkdir -p "$DATA/generated" && chown -R 1000:1000 "$DATA"        # uid 1000 = the 'app' user inside the container
if [[ ! -d "$APP/.git" ]]; then git clone "$REPO" "$APP"; fi
cd "$APP"
git fetch --quiet origin "$BRANCH"
git reset --hard "origin/$BRANCH"                                 # server copy is disposable; git is the source of truth
echo "==> code at $(git rev-parse --short HEAD): $(git log -1 --pretty=%s)"

# --- 2. secrets must already exist (created by you); never print them ---------------------------------------
if [[ ! -f "$ENVFILE" ]]; then
  echo "!! $ENVFILE is missing. Create it from deploy/discover-study.env.example (see deploy/README.md step 2)."; exit 3
fi
if ! grep -Eq '^VENICE_API_KEY=.+' "$ENVFILE" || grep -q 'your_key_here' "$ENVFILE"; then
  echo "!! VENICE_API_KEY is not set in $ENVFILE. Edit it on the server, then re-run."; exit 3
fi

# --- 3. Caddy path block (first time only; backs up, validates, rolls back on any problem) -------------------
if [[ "${INIT:-0}" == "1" ]]; then
  CF=/etc/caddy/Caddyfile
  if grep -q '# discover-study (added by' "$CF"; then
    echo "==> Caddy block already present"
  else
    cp "$CF" "$CF.bak-$(date +%Y%m%d-%H%M%S)"
    echo "$SNIPPET_B64" | base64 -d > /tmp/discover-study.snippet
    # insert just before the site's `root * /var/www` line, next to the existing /thriftstore/ block
    awk 'BEGIN{done=0} !done && /^[ \t]*root \* \/var\/www/ {while((getline line < "/tmp/discover-study.snippet")>0) print line; print ""; done=1} {print}' "$CF" > /tmp/Caddyfile.new
    if ! grep -q '# discover-study (added by' /tmp/Caddyfile.new; then echo "!! could not find the insertion point in $CF; not touching it"; exit 4; fi
    if caddy validate --config /tmp/Caddyfile.new --adapter caddyfile >/dev/null 2>&1; then
      cp /tmp/Caddyfile.new "$CF" && systemctl reload caddy && echo "==> Caddy reloaded with the /discover_can_embedding_demo/ path"
    else
      echo "!! new Caddyfile failed validation; original left untouched"; caddy validate --config /tmp/Caddyfile.new --adapter caddyfile 2>&1 | tail -5; exit 4
    fi
  fi
fi

# --- 4. build (slow the first time: PyTorch + the model), keeping the previous image for rollback -------------
docker image inspect $NAME:latest >/dev/null 2>&1 && docker tag $NAME:latest $NAME:prev || true
docker build -t $NAME:latest .

# --- 5. swap the container, with a memory cap so it can never take the site down with it ----------------------
run_container() {
  docker rm -f $NAME >/dev/null 2>&1 || true
  docker run -d --name $NAME --restart unless-stopped \
    --memory 1100m --memory-swap 1100m --cpus 1.0 \
    -p 127.0.0.1:$PORT:8501 \
    --env-file "$ENVFILE" -e STUDY_DATA_DIR=/data \
    -v "$DATA":/data -v "$DATA/generated":/app/artifacts/generated \
    "$1" >/dev/null
}
wait_healthy() {
  for i in $(seq 1 45); do
    curl -fsS "http://127.0.0.1:$PORT/discover_can_embedding_demo/_stcore/health" >/dev/null 2>&1 && return 0
    sleep 4
  done
  return 1
}
run_container $NAME:latest
if wait_healthy; then
  echo "==> healthy: https://ladan.io/discover_can_embedding_demo/"
  docker image prune -f >/dev/null 2>&1 || true
else
  echo "!! new version did not become healthy. Last logs:"; docker logs --tail 30 $NAME 2>&1 | sed -E 's/(sk|key)[-_A-Za-z0-9]{12,}/<redacted>/g'
  if docker image inspect $NAME:prev >/dev/null 2>&1; then
    echo "==> rolling back to the previous image"; run_container $NAME:prev && wait_healthy && echo "==> rolled back OK"
  fi
  exit 1
fi
REMOTE
echo "==> done"
