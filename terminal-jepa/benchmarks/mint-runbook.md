# dockerfs2 v2.0 mint runbook (the ONE command sequence; constitution §1/§3/§9)

Pre-conditions: review-loop CONVERGED-GO recorded in the prereg; working tree clean at the
runbook's commit; `data/dockerfs2` absent or empty.

```sh
# 1. THE MINT (single collection event; ~1h at measured ~90ms/step, 6 workers)
uv run python -m realenv.collect_docker --out data/dockerfs2 --policy v2 \
  --pin-digests --expect-digests benchmarks/dockerfs2-digests.json \
  --seqs-per-image 600 --seq-len 24 --seed 0 --workers 6 \
  --train-images "alpine:latest,ubuntu:latest,debian:stable-slim,python:3.12-slim,redis:7-alpine,nginx:stable-alpine,postgres:16-alpine,node:22-slim" \
  --val-images "fedora:latest,rockylinux:9,mariadb:latest,httpd:2.4"

# 2. PUBLICATION SCAN (must exit 0; fail-closed on vacuous scans)
uv run python -m benchmarks.scan_publish --root data/dockerfs2

# 3. ENCODE the primary scoring root (copies summary.json = version identity)
uv run python -m evolve.reencode --perception enc_e5_base --src data/dockerfs2 --out data/dockerfs2-e5

# 4. VERIFY version binding + record encoded-artifact shas (append to summary via amendment)
uv run python -c "import sys; sys.path.insert(0,'.'); from evolve import bench_versions as BV; s=BV.resolve('data/dockerfs2-e5'); assert s['version']=='dockerfs2-v2.0', s; print(s)"

# 5. HF PUBLICATION (dockerfs2/ + dockerfs2-e5/ subtrees; HF is canonical thereafter)
#    (uses HF_TOKEN from ~/.runpod.env; scan must have passed in step 2)
```

ABORT DISPOSITION (pre-committed, Amendment 5): on ANY failure in step 1, delete
`data/dockerfs2` wholesale and re-run step 1 in full. The completed run is the version's
single collection event. No splicing, no resume, no --train-only patching.
