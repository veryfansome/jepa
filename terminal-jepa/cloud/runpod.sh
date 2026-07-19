#!/bin/bash
# cloud/runpod.sh — RunPod provider wrapper for terminal-jepa (GraphQL + REST, pure
# curl+jq+ssh+rsync; no CLI install). Ported from the proven sandbox/runs/runpod.sh
# (its GraphQL/REST split, _await_ssh EOF guard, parseable-last-line convention, and
# ssh-identity pinning are kept verbatim); retargeted to this repo: 1x cheap GPU by
# default, no wandb, uv-synced env, data roots rsynced explicitly via sync-data.
#
# A pod is a Docker container: you get `root` over SSH on an EXPOSED PUBLIC PORT
# (not user@ip:22); the ssh public key is injected via the PUBLIC_KEY pod env var.
#
# Subcommands (each prints machine-parseable results on the last stdout line):
#   types [--available]              list GPU types + price/stock (filter: $RUNPOD_TYPES_FILTER)
#   launch                           deploy a pod; prints "<podId> <ip> <port>"
#   host       <podId>               re-resolve "<ip> <port>" (survives restarts)
#   ssh        <podId>               interactive ssh into the pod
#   bootstrap  <podId>               apt deps + rsync repo code + uv sync + CUDA smoke
#   sync-data  <podId> <root>...     rsync data root caches (emb-seq-*.pt + summary.json) up
#   pull       <podId>               rsync pod results (~/jepa/terminal-jepa/cloud/podresults/) down
#   terminate  <podId>               terminate (stops billing)
#   status     <podId> | list
#   datacenters | volumes | create-volume <name> <gb> <dc>   (network-volume path; unused by
#                                    default — our data roots are small enough to rsync)
#
# Auth: RUNPOD_API_KEY, auto-sourced from $RUNPOD_ENV_FILE (default ~/.runpod.env,
# falling back to ~/.lambda.env where the key already lives from the sandbox project).
# Env knobs (defaults tuned for scoring this repo's ~2M-param world models):
#   RUNPOD_GPU_TYPE     default "NVIDIA GeForce RTX 4090"
#   RUNPOD_GPU_COUNT    default 1
#   RUNPOD_CLOUD        SECURE | COMMUNITY | ALL   (default SECURE; COMMUNITY is cheaper)
#   RUNPOD_IMAGE        default runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#                       (uv sync installs the repo's own python+torch; the image just
#                       supplies CUDA drivers)
#   RUNPOD_DISK_GB      default 60 (repo + data caches + .venv + HF encoder models)
#   RUNPOD_POD_NAME     default tjepa-score
#   RUNPOD_TYPES_FILTER default "4090|5090|A100|H100|L40|A40|RTX 6000"
#   RUNPOD_PUBKEY_FILE / RUNPOD_SSH_KEY   ssh identity (auto-discovered otherwise)
#   RUNPOD_VOLUME_ID + RUNPOD_DATACENTER (+ RUNPOD_VOLUME_MOUNT)  optional volume deploy

set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # the jepa repo root
TJ_DIR="$REPO_DIR/terminal-jepa"
RUNPOD_ENV_FILE="${RUNPOD_ENV_FILE:-$HOME/.runpod.env}"
[ -f "$RUNPOD_ENV_FILE" ] || RUNPOD_ENV_FILE="$HOME/.lambda.env"
[ -f "$RUNPOD_ENV_FILE" ] && { set +u; . "$RUNPOD_ENV_FILE"; set -u; }
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY required (put it in ~/.runpod.env or ~/.lambda.env)}"

GQL="https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY"
GPU_TYPE="${RUNPOD_GPU_TYPE:-NVIDIA GeForce RTX 4090}"
GPU_COUNT="${RUNPOD_GPU_COUNT:-1}"
CLOUD="${RUNPOD_CLOUD:-SECURE}"
IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
DISK_GB="${RUNPOD_DISK_GB:-60}"
POD_NAME="${RUNPOD_POD_NAME:-tjepa-score}"
TYPES_FILTER="${RUNPOD_TYPES_FILTER:-4090|5090|A100|H100|L40|A40|RTX 6000}"
RUNPOD_SSH_KEY="${RUNPOD_SSH_KEY:-}"
if [ -z "$RUNPOD_SSH_KEY" ]; then
    for c in "$HOME/.ssh/lambda_cloud_ed25519" "$HOME/.ssh/id_ed25519" "$HOME/.ssh/id_rsa"; do
        [ -f "$c" ] && RUNPOD_SSH_KEY="$c" && break
    done
fi
# IdentitiesOnly: a multi-key agent trips MaxAuthTries; accept-new: every pod is a fresh host key
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30${RUNPOD_SSH_KEY:+ -i $RUNPOD_SSH_KEY -o IdentitiesOnly=yes}"

log() { echo "==> $*" >&2; }
die() { echo "ERROR: $*" >&2; exit 1; }

gql() {
    local q; q=$(jq -Rn --arg q "$1" '{query:$q}')
    curl -sS -X POST "$GQL" -H "Content-Type: application/json" -d "$q"
}

# REST (Bearer auth) — only needed for network volumes / volume-attached deploy
# (GraphQL podFindAndDeployOnDemand has no networkVolumeId field).
REST_BASE="https://rest.runpod.io/v1"
rest() {
    local method="$1" path="$2" body="${3:-}"
    if [ -n "$body" ]; then
        curl -sS -X "$method" "$REST_BASE$path" -H "Authorization: Bearer $RUNPOD_API_KEY" \
            -H "Content-Type: application/json" -d "$body"
    else
        curl -sS -X "$method" "$REST_BASE$path" -H "Authorization: Bearer $RUNPOD_API_KEY"
    fi
}

_pubkey() {
    local f="${RUNPOD_PUBKEY_FILE:-${RUNPOD_SSH_KEY:+$RUNPOD_SSH_KEY.pub}}"
    [ -n "$f" ] && [ -f "$f" ] || die "no ssh public key (set RUNPOD_PUBKEY_FILE, or RUNPOD_SSH_KEY with a .pub beside it)"
    cat "$f"
}

# resolve a running pod's public ssh "<ip> <port>" (the privatePort-22 mapping)
_host() {
    local id="$1" js
    js=$(gql "query{ pod(input:{podId:\"$id\"}){ id desiredStatus runtime{ ports{ ip isIpPublic privatePort publicPort type } } } }")
    echo "$js" | jq -r '.data.pod.runtime.ports[]? | select(.privatePort==22 and .isIpPublic==true) | "\(.ip) \(.publicPort)"' | head -1
}

_ssh_to() { local ip="$1" port="$2"; shift 2; ssh $SSH_OPTS -p "$port" "root@$ip" "$@"; }

cmd_types() {
    local sel=""
    [ "${1:-}" = "--available" ] && sel='| select(.lowestPrice.stockStatus != null)'
    gql "query{ gpuTypes{ id displayName secureCloud communityCloud lowestPrice(input:{gpuCount:$GPU_COUNT}){ uninterruptablePrice stockStatus } } }" \
        | jq -r ".data.gpuTypes[]? | select(.displayName|test(\"$TYPES_FILTER\")) $sel | \"\(.displayName) | ${GPU_COUNT}x stock:\(.lowestPrice.stockStatus // \"none\") \$\(.lowestPrice.uninterruptablePrice // \"-\")/hr | id=\(.id)\""
}

# poll a just-created pod id → RUNNING + a public SSH port, wait for sshd, then print
# "<id> <ip> <port>" (the parseable last line).
_await_ssh() {
    local id="$1"
    log "pod id: $id — polling for RUNNING + a public SSH port (2-5 min)"
    local ip="" port="" st="" tries=0
    while :; do
        st=$(gql "query{ pod(input:{podId:\"$id\"}){ desiredStatus } }" 2>/dev/null | jq -r '.data.pod.desiredStatus // "?"' 2>/dev/null || echo "?")
        # reset + `|| true`: before the port is ready _host emits nothing and `read` hits
        # EOF (non-zero) → under set -e that would abort launch and ORPHAN a billing pod.
        ip=""; port=""; read -r ip port < <(_host "$id") || true
        echo "    status=$st ssh=${ip:+$ip:$port}" >&2
        [ -n "${ip:-}" ] && [ -n "${port:-}" ] && break
        tries=$((tries+1)); [ "$tries" -gt 60 ] && die "pod never exposed a public SSH port (check the RunPod console for $id)"
        sleep 10
    done
    log "waiting for sshd at $ip:$port"
    for _ in $(seq 1 30); do _ssh_to "$ip" "$port" true 2>/dev/null && break; sleep 10; done
    echo "$id $ip $port"     # last line: parseable
}

_deploy_gql() {
    local pubkey; pubkey=$(_pubkey | tr -d '\n')
    log "deploying ${GPU_COUNT}x '$GPU_TYPE' ($CLOUD), disk=${DISK_GB}GB, image=$IMAGE (no volume)"
    local m id
    m=$(gql "mutation{ podFindAndDeployOnDemand(input:{
            cloudType:$CLOUD gpuCount:$GPU_COUNT gpuTypeId:\"$GPU_TYPE\"
            name:\"$POD_NAME\" imageName:\"$IMAGE\"
            containerDiskInGb:$DISK_GB volumeInGb:0
            ports:\"22/tcp\" supportPublicIp:true startSsh:true
            env:[{key:\"PUBLIC_KEY\" value:\"$pubkey\"}]
        }){ id } }")
    id=$(echo "$m" | jq -r '.data.podFindAndDeployOnDemand.id // empty')
    [ -n "$id" ] || { echo "$m" | jq -r '.errors[0].message // .' >&2; die "deploy failed"; }
    echo "$id"
}

_deploy_rest() {
    local pubkey; pubkey=$(_pubkey | tr -d '\n')
    : "${RUNPOD_DATACENTER:?RUNPOD_DATACENTER required with RUNPOD_VOLUME_ID}"
    local mount="${RUNPOD_VOLUME_MOUNT:-/workspace}"
    log "deploying ${GPU_COUNT}x '$GPU_TYPE' ($CLOUD) in $RUNPOD_DATACENTER, volume $RUNPOD_VOLUME_ID @ $mount"
    local body r id
    body=$(jq -n --arg name "$POD_NAME" --arg img "$IMAGE" --arg cloud "$CLOUD" \
        --argjson gc "$GPU_COUNT" --arg gt "$GPU_TYPE" --arg dc "$RUNPOD_DATACENTER" \
        --arg vol "$RUNPOD_VOLUME_ID" --arg mount "$mount" --argjson disk "$DISK_GB" --arg pk "$pubkey" \
        '{name:$name, imageName:$img, cloudType:$cloud, computeType:"GPU",
          gpuCount:$gc, gpuTypeIds:[$gt], dataCenterIds:[$dc],
          networkVolumeId:$vol, volumeMountPath:$mount, volumeInGb:0,
          containerDiskInGb:$disk, ports:["22/tcp"], env:{PUBLIC_KEY:$pk}}')
    r=$(rest POST /pods "$body")
    id=$(echo "$r" | jq -r '.id // empty')
    [ -n "$id" ] || { echo "$r" | jq -r '.error // .message // .' >&2; die "deploy (REST + volume) failed"; }
    echo "$id"
}

cmd_launch() {
    local id
    if [ -n "${RUNPOD_VOLUME_ID:-}" ]; then id=$(_deploy_rest); else id=$(_deploy_gql); fi
    _await_ssh "$id"
}

cmd_datacenters() {
    log "storage-capable datacenters + ${GPU_COUNT}x '$GPU_TYPE' stock:"
    local dc st
    for dc in $(gql 'query{ dataCenters { id storageSupport } }' | jq -r '.data.dataCenters[]? | select(.storageSupport==true) | .id'); do
        st=$(gql "query{ gpuTypes(input:{id:\"$GPU_TYPE\"}){ lowestPrice(input:{gpuCount:$GPU_COUNT, dataCenterId:\"$dc\"}){ stockStatus uninterruptablePrice } } }" \
            | jq -r '.data.gpuTypes[0].lowestPrice | "stock=\(.stockStatus // "none") $\(.uninterruptablePrice // "-")/hr"' 2>/dev/null)
        printf '  %-10s %s\n' "$dc" "$st" >&2
    done
}

cmd_volumes() {
    rest GET /networkvolumes | jq -r 'if type=="array" then (.[] | "id=\(.id)  name=\(.name)  size=\(.size)GB  dc=\(.dataCenterId)") elif .error then "ERROR: \(.error)" else . end'
}

cmd_create_volume() {
    local name="${1:?create-volume <name> <sizeGB> <dataCenterId>}" size="${2:?<sizeGB>}" dc="${3:?<dataCenterId>}"
    local body r id
    body=$(jq -n --arg n "$name" --argjson s "$size" --arg dc "$dc" '{name:$n, size:$s, dataCenterId:$dc}')
    r=$(rest POST /networkvolumes "$body")
    id=$(echo "$r" | jq -r '.id // empty')
    [ -n "$id" ] || { echo "$r" | jq -r '.error // .message // .' >&2; die "volume create failed"; }
    echo "$id"
}

cmd_host()   { local id="${1:?host <podId>}"; _host "$id"; }
cmd_status() { local id="${1:?status <podId>}"; gql "query{ pod(input:{podId:\"$id\"}){ id name desiredStatus machineId } }" | jq '.data.pod'; }
cmd_list()   { gql 'query{ myself{ pods{ id name desiredStatus machine{ gpuDisplayName } } } }' | jq '.data.myself.pods'; }

cmd_ssh() {
    local id="${1:?ssh <podId>}" ip port
    read -r ip port < <(_host "$id"); [ -n "${ip:-}" ] || die "no public SSH port for $id (running?)"
    log "ssh root@$ip -p $port"
    exec ssh $SSH_OPTS -p "$port" "root@$ip"
}

cmd_bootstrap() {
    local id="${1:?bootstrap <podId>}"
    local ip port; read -r ip port < <(_host "$id"); [ -n "${ip:-}" ] || die "no SSH port for $id"

    # the base image may lack rsync/git/tmux; install before the first rsync
    log "ensuring rsync/git/curl/tmux on the pod"
    _ssh_to "$ip" "$port" '(command -v rsync >/dev/null && command -v git >/dev/null && command -v tmux >/dev/null) || (apt-get update -qq && apt-get install -y -qq rsync git curl tmux)'

    log "rsync repo code → root@$ip:~/jepa/  (code only; data goes via sync-data)"
    rsync -a --delete --exclude=.venv --exclude=__pycache__ --exclude='*.pyc' --exclude=.git \
        --exclude='data/' --exclude='cloud/podresults/' --exclude='cloud/*.state' --exclude='cloud/*.log' \
        --exclude='evolve/archive/base_cache.json' \
        -e "ssh $SSH_OPTS -p $port" "$REPO_DIR/" "root@$ip:~/jepa/"

    log "uv sync + CUDA smoke on the pod"
    _ssh_to "$ip" "$port" '
        set -e
        command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        cd ~/jepa/terminal-jepa && uv sync
        uv run python -c "import torch; assert torch.cuda.is_available(), \"no CUDA\"; print(\"cuda ok:\", torch.cuda.get_device_name(0))"
    '
    log "bootstrap complete for $id"
}

# sync-data <podId> <data_root>... — rsync a data root's encoder caches + summary up.
# Only emb-seq-*.pt + summary.json are needed to score (cached_encode short-circuits);
# raw *.jsonl stay local.
cmd_sync_data() {
    local id="${1:?sync-data <podId> <data_root>...}"; shift
    local ip port; read -r ip port < <(_host "$id"); [ -n "${ip:-}" ] || die "no SSH port for $id"
    local root
    for root in "$@"; do
        [ -d "$TJ_DIR/$root" ] || die "no such data root: $TJ_DIR/$root"
        log "sync-data $root → pod"
        _ssh_to "$ip" "$port" "mkdir -p ~/jepa/terminal-jepa/$root"
        rsync -a --include='emb-seq-*.pt' --include='summary.json' --exclude='*' \
            -e "ssh $SSH_OPTS -p $port" "$TJ_DIR/$root/" "root@$ip:~/jepa/terminal-jepa/$root/"
    done
}

cmd_pull() {
    local id="${1:?pull <podId>}" ip port; read -r ip port < <(_host "$id"); [ -n "${ip:-}" ] || die "no SSH port for $id"
    local dest="$TJ_DIR/cloud/podresults"; mkdir -p "$dest"
    log "rsync pod cloud/podresults/ → $dest/"
    rsync -a -e "ssh $SSH_OPTS -p $port" "root@$ip:~/jepa/terminal-jepa/cloud/podresults/" "$dest/"
}

cmd_terminate() {
    local id="${1:?terminate <podId>}"
    log "terminating $id"
    gql "mutation{ podTerminate(input:{podId:\"$id\"}) }" | jq -r 'if .errors then .errors[0].message else "terminated" end' >&2
}

case "${1:-}" in
    ""|-h|--help) sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 0 ;;
esac
CMD="$1"; shift
case "$CMD" in
    types) cmd_types "$@";; datacenters) cmd_datacenters "$@";;
    volumes) cmd_volumes "$@";; create-volume) cmd_create_volume "$@";;
    launch) cmd_launch "$@";; host) cmd_host "$@";;
    ssh) cmd_ssh "$@";; bootstrap) cmd_bootstrap "$@";; sync-data) cmd_sync_data "$@";;
    pull) cmd_pull "$@";;
    terminate) cmd_terminate "$@";; status) cmd_status "$@";; list) cmd_list "$@";;
    *) die "unknown command: $CMD";;
esac
