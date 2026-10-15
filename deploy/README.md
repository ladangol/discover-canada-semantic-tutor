# Deploying Discover Canada Study at https://ladan.io/discover_can_embedding_demo

What I found on the droplet (read-only inspection) and what that means:

| Finding | Consequence |
|---|---|
| Caddy serves ladan.io (static files from `/var/www`) and already proxies `/thriftstore/` to a Docker container on `127.0.0.1:5050` | The new app gets its own path the same way: `/discover_can_embedding_demo/` → `127.0.0.1:8502`. The existing blocks are not edited. |
| Docker 20.10 is installed | The app runs in its own container, so the host's Ubuntu 18.04 / Python 3.6 / glibc 2.27 don't matter. |
| **985 MB RAM, ~460 MB free, no swap, 1 vCPU** | **Too small.** The app needs ~620 MB steady (~780 MB while loading) on top of what the site already uses (~360 MB). Memory exhaustion could make Linux kill the thriftstore container or Caddy. `deploy.sh` refuses to run below 1.8 GB. |
| Ubuntu 18.04, up 4+ years | Out of standard support (no security updates), unrelated to this app but worth planning an OS upgrade or a fresh droplet. |

```mermaid
flowchart LR
    V[Visitor] -->|HTTPS| C[Caddy on the droplet]
    C -->|/thriftstore/*| T[thriftstore container :5050 (unchanged)]
    C -->|/discover_can_embedding_demo/*| D[discover-study container :8502<br/>memory-capped]
    C -->|everything else| W[/var/www static site (unchanged)/]
    D --> L[(/var/lib/discover-study<br/>usage ledger + cached maps)]
    D -->|only if the day's allowance remains| P[Venice API]
```

## 0. One-time: give the droplet enough memory

DigitalOcean → Droplets → *your droplet* → **Resize** → choose **CPU and RAM only** (reversible) with **2 GB RAM**
or more. Resizing powers the droplet off briefly, so ladan.io is down for a few minutes: do it at a quiet time and
take a snapshot first. (Alternative: a second small 2 GB droplet just for this demo, if you'd rather not touch the
site's machine at all. The same scripts work; change `DEPLOY_HOST`.)

## 1. Spending control: three layers (the "$1 a day" question)

1. **A dedicated Venice key with its own USD limit.** Venice returns HTTP 402 `API_KEY_USD_SPEND_LIMIT_EXCEEDED`
   when a key hits its limit, so keys support USD limits. I couldn't confirm whether that limit resets daily, so
   treat it as the hard stop, not the daily control. Keep the account balance modest.
2. **The app's own daily budget** (`src/budget.py`): one ledger shared by all visitors; the app stops calling Venice
   once today's allowance is gone and resumes at local midnight (`STUDY_TZ`).
   - Dollars are enforced only if you set `VENICE_PRICE_IN_PER_M` and `VENICE_PRICE_OUT_PER_M` (your model's
     price per 1M tokens: copy from Venice's pricing page; I didn't guess) plus `STUDY_DAILY_USD_BUDGET=1.00`.
   - Otherwise only the token cap applies (`STUDY_DAILY_TOKEN_BUDGET`, default 500,000/day).
   - Read it: `ssh ladanio 'sudo sqlite3 /var/lib/discover-study/usage.sqlite3 "select * from usage order by day desc limit 10;"'`
     (install `sqlite3` on the host if missing).
3. **Abuse limits:** per browser session (`STUDY_MAX_ACTIONS`, 40), per visitor per day
   (`STUDY_MAX_ACTIONS_PER_VISITOR_DAY`, 60; a salted hash of the IP, never stored raw), and
   `STUDY_LIVE_TIMELINES=0` so whole-chapter timelines (~8 AI calls) are only served pre-built.
   Caddy has no built-in rate limiter, so these app-level caps are the front line.

## 2. Create the secret file on the server (you do this; I never see your key)

```bash
ssh ladanio
sudo cp /opt/discover-canada-semantic-tutor/deploy/discover-study.env.example /etc/discover-study.env   # after the first deploy.sh --init clone
# or create it by hand before the first deploy:
sudo nano /etc/discover-study.env          # paste your key + prices (template: deploy/discover-study.env.example)
sudo chmod 600 /etc/discover-study.env
```
`deploy.sh` checks that the key is set without ever printing it, and passes it to the container as an environment
variable only. It is never in git, the image, the page, logs, or error messages.

## 3. Deploy and update, straight from git

```bash
git push origin main                 # your changes must be on GitHub; the server pulls from there
deploy/deploy.sh --init              # first time. The 1st run clones the repo, then stops at the secret check
                                     # (exit 3) if /etc/discover-study.env doesn't exist yet: create it (step 2), run it again.
deploy/deploy.sh                     # every later update: pull, rebuild, swap, health-check
```
What it does on the server: clones/updates `/opt/discover-canada-semantic-tutor` (`git reset --hard origin/main`),
checks memory and the secret file, builds the image (first build is slow: PyTorch + the MiniLM model), starts a new
container capped at 1.1 GB with `--restart unless-stopped`, waits for the health check, and **rolls back to the
previous image automatically** if the new one doesn't become healthy. The Caddy edit is validated before it's
applied and the original is saved as `/etc/caddy/Caddyfile.bak-<timestamp>`.

Defaults: host `ladanio`, key `~/.ssh/id_ed25519`, branch `main` (override with `DEPLOY_HOST`, `DEPLOY_KEY`,
`DEPLOY_BRANCH`).

Pre-build the demo library once (a small batch of AI calls) so those pages never need the AI:
```bash
ssh ladanio 'docker exec discover-study python scripts/prewarm_cache.py'
```

## 4. What visitors see when something goes wrong

| Situation | App behaviour | Visitor sees |
|---|---|---|
| Daily allowance used up (your cap) | no call made | banner + calm message on AI actions; **pre-built maps/timelines, facts, tiles and flashcards already made keep working** |
| Venice 402 (key limit / no balance) | not retried; app stops calling until midnight | same message |
| Venice overloaded / rate-limited / timeout | 3 retries (2 s, 5 s, 10 s) | “The AI is busy right now. Please try again in a minute.” |
| Key missing / invalid (401, 403) | status logged | “The study tools are temporarily unavailable.” |
| Content-policy refusal (422) | not retried | “I can't help with that one. Try rephrasing your question.” |
| Library-only mode, topic not pre-built | no call | “That one isn't in the demo's ready-made library yet…” |
| Session / visitor limit | refused before any call | “You've reached the study limit for this session…” |

No tracebacks, provider text or keys ever reach a visitor. Logs (`docker logs discover-study`) hold HTTP status and
reason only.

## 5. Operations

- **Logs:** `ssh ladanio 'docker logs -f --tail 100 discover-study'`
- **Restart:** `ssh ladanio 'docker restart discover-study'`
- **Change a cap or key:** edit `/etc/discover-study.env` on the server, then `docker restart discover-study`
  (the env file is read when the container is created: if a restart doesn't pick it up, run `deploy/deploy.sh`).
- **Rotate the key:** new capped key in Venice → env file → redeploy → delete the old key.
- **Undo the Caddy block:** delete the lines between `# discover-study (added by…` and `# /discover-study`, then
  `sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile && sudo systemctl reload caddy`
  (or restore the `.bak-*` file).
- **Remove the app entirely:** `docker rm -f discover-study; docker rmi discover-study:latest discover-study:prev`,
  remove the Caddy block, `rm -rf /opt/discover-canada-semantic-tutor /var/lib/discover-study`.

## 6. Checks after deploying

```bash
curl -sI https://ladan.io/discover_can_embedding_demo/                       # 200
curl -s  https://ladan.io/discover_can_embedding_demo/_stcore/health         # ok
curl -sI https://ladan.io/discover_can_embedding_demo/app/static/mascots/goose.png   # 200
curl -sI https://ladan.io/thriftstore/ | head -1                             # still works
curl -sI https://ladan.io/ | head -1                                         # main site still works
```
Then open the page, ask a question, and confirm the mascot shows. “Blank page / keeps reconnecting” means WebSockets
aren't passing through Caddy (it handles them automatically, so look for another proxy/CDN in front).

## 7. Security checklist

- [ ] Key only in `/etc/discover-study.env` (mode 600), never in git or the image.
- [ ] Venice key has its own USD limit; account balance is small.
- [ ] The container listens on `127.0.0.1:8502` only; Caddy is the only public entry (HTTPS already on).
- [ ] Container memory/CPU capped (`--memory 1100m --cpus 1`); the site's containers are untouched.
- [ ] Look at the usage ledger after the first day and tune the caps.
- [ ] Plan to upgrade/replace Ubuntu 18.04.

## 8. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `deploy.sh` says “only N MB RAM” | the resize in step 0 hasn't been done |
| 502 from Caddy | container not running: `docker ps -a`, `docker logs discover-study` |
| “Study tools temporarily unavailable” | key missing/invalid in the env file, or a 401 in the logs |
| Mascots missing | the Caddy block uses `handle_path` (strips the prefix); it must be `handle` |
| Build fails on `pip install` | out of disk (`df -h`); `docker system prune -f` |
| Everything says “allowance used up” early | prices unset so the token cap hit first; check the ledger, raise the cap or set prices |
