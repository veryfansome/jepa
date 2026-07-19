#!/bin/bash
# cloud/runpod_score.sh — idempotent RunPod scoring orchestrator for the evolve loop.
# Runs a batch of `evolve.cli score` jobs on one rented pod, serialized in a pod-side
# tmux session, each job skipped if its result JSON already exists; pulls the small
# result JSONs back; refuses to terminate until every result is verified local.
# Skeleton ported from sandbox/runs/runpod_stp_sft_sweep.sh (stage dispatch, state file,
# marker idempotency, transient-ssh-aware poll, verify-before-terminate).
#
#   bash cloud/runpod_score.sh                  # full pipeline over cloud/jobs.tsv
#   STAGE=score bash cloud/runpod_score.sh      # one stage: provision|bootstrap|syncdata|
#                                               #   jobs|verify|score|poll|download|verifydl|terminate
#   YES=1 AUTO_TERMINATE=1 bash cloud/runpod_score.sh    # unattended
#
# cloud/jobs.tsv — one job per line, whitespace-separated:
#   <gid> <mode:proxy|full> <split:inner|final> <data_root> <local_genome_json>
# e.g.:  r9-arch-foo proxy inner data/dockerfs-e5 /path/to/r9-arch-foo.json
#
# Results land in cloud/podresults/<gid>.<mode>.<split>.json (the cli score stdout,
# cache-note lines included — parse the JSON object out). Ingest locally with:
#   uv run python -m evolve.cli ingest --genome <g.json> --result cloud/podresults/<...>.json
#
# ENVIRONMENT COMPARABILITY (methodology, not mechanics): pod (CUDA) fitness numbers are
# NOT comparable to local MPS numbers at the ±0.001 noise level, and the pod computes its
# own objective-independent baselines (base_cache is per-environment, never synced). Keep
# a round's selection comparisons within ONE environment, and re-baseline the incumbent
# (score the unchanged champion genome first) whenever you move environments.

set -euo pipefail
CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TJ_DIR="$(cd "$CLOUD_DIR/.." && pwd)"
RP="bash $CLOUD_DIR/runpod.sh"
JOBS_FILE="${JOBS_FILE:-$CLOUD_DIR/jobs.tsv}"
STATE_FILE="$CLOUD_DIR/.runpod_score.state"
CONCURRENCY="${CONCURRENCY:-1}"   # >1 runs jobs concurrently on the pod GPU; keep 1 for clean determinism

log() { echo "==> $*" >&2; }
die() { echo "ERROR: $*" >&2; exit 1; }
confirm() {
    [ "${YES:-0}" = "1" ] && return 0
    local ans; read -r -p "$1 [y/N] " ans < /dev/tty; [ "$ans" = "y" ] || die "aborted"
}

_load() { [ -f "$STATE_FILE" ] && . "$STATE_FILE" || true; }
_save() { { printf 'POD_ID=%q\n' "$POD_ID"; printf 'POD_IP=%q\n' "$POD_IP"; printf 'POD_PORT=%q\n' "$POD_PORT"; } > "$STATE_FILE"; }
_refresh() {  # ip:port change across pod restarts — always re-resolve
    _load; [ -n "${POD_ID:-}" ] || die "no pod in state ($STATE_FILE) — run provision"
    local ip="" port=""
    read -r ip port < <($RP host "$POD_ID") || true
    [ -n "$ip" ] || die "pod $POD_ID has no public SSH port (status: $($RP status "$POD_ID" | jq -r .desiredStatus))"
    POD_IP="$ip"; POD_PORT="$port"; _save
}
SSH_KEY="${RUNPOD_SSH_KEY:-}"
if [ -z "$SSH_KEY" ]; then
    for c in "$HOME/.ssh/lambda_cloud_ed25519" "$HOME/.ssh/id_ed25519" "$HOME/.ssh/id_rsa"; do
        [ -f "$c" ] && SSH_KEY="$c" && break
    done
fi
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30${SSH_KEY:+ -i $SSH_KEY}${RUNPOD_SSH_PIN:+ -o IdentitiesOnly=yes}"
_ssh() { ssh $SSH_OPTS -p "$POD_PORT" "root@$POD_IP" "$@"; }

_jobs() { grep -vE '^\s*(#|$)' "$JOBS_FILE"; }
_result_name() { echo "$1.$2.$3.json"; }   # gid mode split

stage_provision() {
    _load
    if [ -n "${POD_ID:-}" ] && [ "$($RP status "$POD_ID" 2>/dev/null | jq -r '.desiredStatus // empty')" = "RUNNING" ]; then
        log "reusing running pod $POD_ID"; _refresh; return
    fi
    [ -f "$JOBS_FILE" ] || die "no jobs file: $JOBS_FILE"
    local n; n=$(_jobs | wc -l | tr -d ' ')
    confirm "Deploy 1x ${RUNPOD_GPU_TYPE:-RTX 4090} for $n scoring job(s)? (check '$RP types --available' for the rate; terminate is manual-verified)"
    local id ip port
    read -r id ip port < <($RP launch | tail -1)
    [ -n "$id" ] || die "launch failed"
    POD_ID="$id"; POD_IP="$ip"; POD_PORT="$port"; _save
    log "provisioned $POD_ID @ $POD_IP:$POD_PORT"
}

stage_bootstrap() {
    _refresh
    if _ssh 'test -f ~/.tjepa_setup_done' 2>/dev/null; then log "bootstrap: already done"; return; fi
    $RP bootstrap "$POD_ID"
    _ssh 'touch ~/.tjepa_setup_done'
}

stage_syncdata() {
    _refresh
    local roots; roots=$(_jobs | awk '{print $4}' | sort -u)
    # shellcheck disable=SC2086
    $RP sync-data "$POD_ID" $roots
}

stage_jobs() {  # rsync the genome JSONs named in jobs.tsv up under cloud/podjobs/<gid>.json
    _refresh
    local tmp; tmp=$(mktemp -d)
    while read -r gid mode split data gpath; do
        [ -f "$gpath" ] || die "genome file missing for $gid: $gpath"
        cp "$gpath" "$tmp/$gid.json"
    done < <(_jobs)
    _ssh 'mkdir -p ~/jepa/terminal-jepa/cloud/podjobs ~/jepa/terminal-jepa/cloud/podresults'
    rsync -a -e "ssh $SSH_OPTS -p $POD_PORT" "$tmp/" "root@$POD_IP:~/jepa/terminal-jepa/cloud/podjobs/"
    rsync -a -e "ssh $SSH_OPTS -p $POD_PORT" "$JOBS_FILE" "root@$POD_IP:~/jepa/terminal-jepa/cloud/jobs.tsv"
    rm -rf "$tmp"
    log "jobs synced: $(_jobs | wc -l | tr -d ' ')"
}

stage_verify() {  # cheap preflight before any paid scoring: GPU, env, data roots present
    _refresh
    _ssh '
        set -e; export PATH="$HOME/.local/bin:$PATH" UV_NO_SYNC=1
        nvidia-smi -L
        cd ~/jepa/terminal-jepa
        uv run python - <<PY
import torch, sys
assert torch.cuda.is_available(), "no CUDA"
sys.path.insert(0, ".")
from evolve import genome as G
from realenv import seq_worldmodel as M
print("imports ok; device:", M.pick_device())
PY
    '
    local root
    for root in $(_jobs | awk '{print $4}' | sort -u); do
        _ssh "test -f ~/jepa/terminal-jepa/$root/emb-seq-train.pt && test -f ~/jepa/terminal-jepa/$root/emb-seq-val.pt" \
            || die "data root $root missing caches on pod — run STAGE=syncdata"
    done
    log "verify: ok"
}

stage_score() {
    _refresh
    if _ssh 'test -f ~/jepa/terminal-jepa/cloud/podresults/.done' 2>/dev/null; then log "score: done marker present"; return; fi
    if _ssh 'tmux has-session -t tjepa' 2>/dev/null; then log "score: tmux session already running"; return; fi
    # generated remote runner: serialize jobs (or xargs -P under CONCURRENCY>1), skip
    # finished ones (result JSON exists AND contains a fitness line), marker at the end.
    local script
    script=$(cat <<'REMOTE'
#!/bin/bash
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH" UV_NO_SYNC=1
cd ~/jepa/terminal-jepa
mkdir -p cloud/podresults
run_one() {
    local gid="$1" mode="$2" split="$3" data="$4"
    local out="cloud/podresults/$gid.$mode.$split.json"
    if [ -s "$out" ] && grep -q '"fitness"' "$out"; then echo "skip $gid.$mode.$split"; return 0; fi
    echo "[$(date -u +%H:%M:%S)] scoring $gid $mode/$split on $data"
    uv run python -m evolve.cli score --genome "cloud/podjobs/$gid.json" \
        --mode "$mode" --split "$split" --data "$data" > "$out" 2>&1
    grep -q '"fitness"' "$out" || { echo "FAILED $gid.$mode.$split — tail:"; tail -5 "$out"; return 1; }
}
export -f run_one
fail=0
if [ "${CONC:-1}" -gt 1 ]; then
    grep -vE '^\s*(#|$)' cloud/jobs.tsv | awk '{print $1" "$2" "$3" "$4}' \
        | xargs -P "${CONC:-1}" -L1 bash -c 'run_one $0 $1 $2 $3' || fail=1
else
    while read -r gid mode split data _; do
        run_one "$gid" "$mode" "$split" "$data" || fail=1
    done < <(grep -vE '^\s*(#|$)' cloud/jobs.tsv)
fi
[ "$fail" = "0" ] && touch cloud/podresults/.done
echo "runner exit fail=$fail"
REMOTE
)
    printf '%s' "$script" | _ssh 'cat > ~/jepa/terminal-jepa/cloud/.podscore_run.sh'
    _ssh "tmux new-session -d -s tjepa 'CONC=$CONCURRENCY bash ~/jepa/terminal-jepa/cloud/.podscore_run.sh 2>&1 | tee ~/jepa/terminal-jepa/cloud/podscore.log'"
    log "score: launched in pod tmux session 'tjepa'"
}

stage_poll() {
    _refresh
    local misses=0 total; total=$(_jobs | wc -l | tr -d ' ')
    while :; do
        if _ssh 'test -f ~/jepa/terminal-jepa/cloud/podresults/.done' 2>/dev/null; then
            log "poll: all jobs done"; return
        elif ! _ssh true 2>/dev/null; then
            # transient ssh (exit 255) is NOT evidence the run died — retry, don't count a miss
            log "poll: host unreachable (transient) — retry in 30s"; sleep 30; continue
        elif _ssh 'tmux has-session -t tjepa' 2>/dev/null; then
            misses=0
            local done_n; done_n=$(_ssh 'ls ~/jepa/terminal-jepa/cloud/podresults/*.json 2>/dev/null | wc -l' | tr -d ' ')
            log "poll [$(date -u +%H:%M:%S)] $done_n/$total results"
        else
            misses=$((misses+1))
            if [ "$misses" -ge 3 ]; then
                _ssh 'tail -30 ~/jepa/terminal-jepa/cloud/podscore.log' >&2 || true
                die "a job failed (tmux gone, no done marker). Fix + re-run — resumes unfinished jobs."
            fi
            sleep 30; continue
        fi
        sleep 60
    done
}

stage_download() { _load; $RP pull "$POD_ID"; }

stage_verifydl() {
    local missing=0
    while read -r gid mode split data _; do
        local f="$CLOUD_DIR/podresults/$(_result_name "$gid" "$mode" "$split")"
        if [ -s "$f" ] && grep -q '"fitness"' "$f"; then :; else log "MISSING/invalid: $f"; missing=1; fi
    done < <(_jobs)
    [ "$missing" = "0" ] || die "refusing: results not verified local (run download, or fix + re-run score)"
    log "verifydl: all $(_jobs | wc -l | tr -d ' ') results local"
}

stage_terminate() {
    stage_verifydl
    _load
    [ "${AUTO_TERMINATE:-0}" = "1" ] || confirm "Terminate pod $POD_ID?"
    $RP terminate "$POD_ID"
    : > "$STATE_FILE"
}

ALL="provision bootstrap syncdata jobs verify score poll download verifydl terminate"
if [ -n "${STAGE:-}" ]; then "stage_$STAGE"; else for s in $ALL; do "stage_$s"; done; fi
