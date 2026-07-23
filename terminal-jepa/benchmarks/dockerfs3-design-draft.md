# dockerfs3 (v3) — unified design draft

STATUS: synthesis draft, integrating the 12 design-fleet memos with the 11 adversarial verdicts. The cross-lens integration verdict's decisions (D-B1..D-B5, D-S1..D-S14) are BINDING and are applied throughout; every per-memo blocker/serious finding is resolved here or listed OPEN (§2, §17). Provenance cited inline as (lens:<memo>) / (verdict:<target> <finding>). Successor document: `benchmarks/dockerfs3-prereg.md`, assembled from §14–§16 plus the explicit incorporation list — §8.5 promotion rules + fallback fork, §8.2 exit vocabulary, §10.3 seven arms, §10.4 information-parity disclosure, §6.1 BNF — before any collection code is trusted.

---

## 1. Executive summary

dockerfs3 is the third mint of the Docker-filesystem world-model benchmark. The evidence trigger: the R13–R16 evolve plateau showed the WORLD, not the training recipe, is the binding lever (charter). v3 makes three qualitative shifts, each entering at measured minority mass over a preserved v2-style majority:

1. **STATE MUTATION** — mkdir/touch/echo-redirect/rm/mv/ln act on a two-tier namespace (a per-sequence `/tmp/w` workspace seeded by copying real image files, plus a denylist-guarded pool of expendable real files). Every mutation journals into ONE shell-state tracker; a revisit controller steers reads back onto mutated state so observations become functions of the mutation history (lens:world-dynamics, revised per verdict:cross-lens D-B2/D-B4).
2. **TIME & PROCESSES** — a virtual step-clock on the existing per-command `docker exec` substrate: background jobs via an installed `after` helper fired deterministically by a tick-prologue, canonical PIDs, signal-based job control (TERM/STOP/CONT/KILL/-0 with a pending-TERM state), `ps` as a deterministic state read (lens:time-processes, repaired per verdict:time B1/S1–S3). **No PTY in v3** (verdict:cross-lens D-B1); jobs/fg/bg/ctrl-keys defer to a v4 spike.
3. **COMPOSITION** — a frozen depth-1 grammar G3: pipes (`ls -1|cat` producers × `head/tail/grep -F` filters), workspace redirection (`>`/`>>`), one conditional form `[ TEST P ] && READ P`. Redirection CREATES linkable filesystem state; all payload bytes are image-grounded (verdict:cross-lens D-B4 / pre-mortem DG-1).

**Fitness shape is unchanged**: pooled content-verb top-1 retrieval margin over the honest-baseline max — now seven arms including a frozen symbolic shell-state tracker (SST) and its composite (verdict:cross-lens D-S8/D-S9). Multi-channel (exit, fs-delta) and multi-step (ROLL@k) enter as data channels and measured instruments, never fitness, with pre-registered v3.1 promotion rules (lens:prediction-targets). Verb classes are measured per (signature, mode, state_scope) cell on four axes with ONE axis-3 rule: SST exact-match ≥ 0.90 ⇒ class `sim`, excluded (verdict:cross-lens D-B3).

**What it costs**: ~302K steps (12 images × 900 seqs × 28±4 steps, 1.75× v2) + a half-size ablate train arm; ~1.7h mint + ~0.85h ablate at 6 workers on the measured exec substrate; ~11–15h encode (incl. per-step delta texts and the full-val SST/wtm precompute — §3.7); ~1–2 days re-baseline; ≤6 pilot iterations under 29 fail-closed pre-registered gates (§14). Publication ≈ 3.7–4.5 GB to HF. Total machine time ≈ 2–4 days on the single MPS host (§3.7, resolving verdict:completeness S1).

**The honesty spine**: one tracker module (`realenv/shell_state.py`) is simultaneously the collection-time state authority, the meta labeler, the axis-3 detector, and the strongest baseline arm — frozen pre-mint, sha-pinned, fidelity-gated at ≥0.995 (verdict:cross-lens D-B5, pre-mortem DG-4). The pre-mortem's aggregate-starvation gate DG-9 (mutation-affected content ≥10% of content steps, composed ≥6%) has a named owner and blocks the mint, so the expansion cannot silently ship empty.

---

## 2. USER DECISIONS REQUIRED

Deviations from the original v3 ask, plus genuinely open choices. Each has a recommendation; none is buried in the body.

**UD-1. Jobs/fg/bg + ctrl+c/ctrl+z are DEFERRED to a v4 PTY spike; v3 delivers the process automaton by signals.** The charter asked for `jobs`, `&` job control, and literal control keys. The infra memo's PTY design was empirically attacked: background-job "Done" notification placement is wall-clock-nondeterministic across replays (measured spread >200 steps on identical seeds — verdict:infra B1), and the timeout-recovery ladder was demonstrated unreachable (verdict:infra S1). The time memo shows the learnable content — a process automaton with suspend/resume/kill — is deliverable race-free on plain `docker exec` via `kill -STOP/-CONT/-TERM` on `after`-launched jobs. The integration verdict ruled exec wins (verdict:cross-lens D-B1). Consequences the user must sign off (verdict:cross-lens M12): `jobs`, `fg`, `bg`, `%n` jobspecs, `wait`, literal `^C`/`^Z`, and bare `sleep N &` are OUT of v3; `kill -INT` is also dropped (empirically a silent no-op on background children of a non-interactive shell — verdict:time B1); `<<<` herestrings are out (non-POSIX; dash/ash reject them — lens:infra, verdict:baselines S3). *Recommendation: accept; the v4 PTY spike charter inherits the infra memo + its verdict's measured fixes.*

**UD-2. Pipes/redirection/conditionals enter at minority mass under a frozen depth-1 grammar.** The charter's open-ended composition becomes G3: exactly one operator per command, 2 producers × 3 filters, workspace-only redirection, ONE conditional form `[ TEST P ] && READ P`; `||`, `;`, `if/then`, depth-2 pipes, `REDIR_IN`, and find-producers are excluded (lens:composition; pruned further by verdict:cross-lens D-S2/D-S12). Rationale: the class protocol's ≥30 cross-image-pair floor makes family count the binding constraint; depth-2 squares it past measurability. Composition mass ≈15% of steps (§7.2 table). *Recommendation: accept; depth-2 exists only as an out-of-corpus eval-side probe battery (report-only).*

**UD-3. Multi-channel and multi-step predictions ship as data + instruments, not fitness.** Every step carries `exit_cls` and a tracker-computed `fs_delta` embedding in the cache/batch contract; two frozen linear probes (EXIT, DELTA) and the ROLL@{2,4} write-back rollout instrument report ledger columns. Promotion to fitness is a v3.1 decision whose rule is frozen in the v3.0 prereg (§8.5), keyed to a baseline bracket + noise band, not raw sign (lens:prediction-targets, repaired per verdict:targets S3). Rationale: R10/R11 recorded that training-signal auxiliaries did not move full-budget outcomes; probe-first makes v3.1 evidence-driven. *Recommendation: accept.*

**UD-4. Counterfactual pre-mutation foils enter the v3.0 metric on the mutation slice (fitness-shape choice).** The eval-split verdict demonstrated (verdict:eval-split B1) that with v2 foil construction, stale within-trajectory retrieval still wins top-1 on mutation-probe steps (the one changed line barely moves the embedding relative to random other-directory foils), so the mutation-slice margin is pinned ≈0 by metric construction and evolution gets no selective signal from the headline slice. Options: (a) keep counterfactual foils report-only (eval memo's original stance — continuity, but a structurally signal-free headline slice); (b) inject forced pre-mutation-twin foils into the foil rows of state_scope∈{mutated,created} content steps at v3.0 (m=8 of 64 slots, only where pre/post renders differ; gated by the DG-2 resolution floor). *Recommendation: (b). This is a version-boundary eval change, legal under constitution §5, and the only way the slice can discriminate. The pooled aggregate and v2-style foils on native steps are unchanged.*

**UD-5. Images: 12 re-pinned v2 images; up to 3 new TRAIN-ONLY images contingent on the P0 capability audit.** The integration verdict ruled 12-only (verdict:cross-lens D-S10) on identity-risk grounds; the completeness verdict (B1) requires an images decision with a ps/procps audit. Reconciliation: val/final stay EXACTLY the v2 four (fedora, mariadb / rockylinux, httpd) — continuity preserved; new images, if any, go to TRAIN ONLY, which removes the split-identity risk D-S10 feared. Trigger: if P0 shows <3 of 8 train images with a usable `ps` (expected: ubuntu procps + alpine-family busybox-T2; debian-slim/python-slim/node-slim absent), the process arm is train-starved. Candidates in §12. *Recommendation: adopt `debian:stable` (full, procps) and `opensuse/leap:15.6` train-only iff the audit trips; else ship 12.*

**UD-6. Constitutional amendment: the constitution §6 probe-ladder gate is enforced on the native-continuity slice only.** Ladder rank order on mutation/time/composition slices is reported, not gated (demanding rank preservation on phenomena no rung trained for asserts the conclusion) (lens:eval-split D8; verdict:eval-split M7 requires this be a dated amendment, not a reinterpretation). *Recommendation: approve the amendment text in §14.*

**UD-7. Expansion abort criterion.** One-mint discipline needs a dignified exit (verdict:completeness M4): if the pilot loop exceeds **6 iterations** or **3 calendar weeks** without all gates green, v3 is shelved, recorded as a negative in README.md, and dockerfs2 remains the active bench. *Recommendation: accept these caps.*

**UD-8. Publication scope.** Publish raw + primary e5 roots AND the ablate raw root (it is part of the version's identity); exempt instrument tensors (`sst-val.pt`, `wtm-val.pt`, ROLL anchors) as sha-pinned regenerables (verdict:completeness M2). *Recommendation: accept.*

**Notation (codes used throughout)**: **F1** = host-executor artifact — an output produced by the host tooling, not the world; never a recorded observation, and its occurrence aborts the image (§3.1). **F5** = trajectory-abort semantics — an infra barrier/timeout failure discards and re-collects the whole trajectory; no partial sequence is ever stored (§3.3). **F8** = recoverable-from-record discipline — any flag the eval consumes must be recomputable from the stored step record {cmd, output, exit, cwd} alone; meta may cache it, never define it (§9.1, §13.1). **F6** = train-only collection mode — the ablate arm collects no val.jsonl (collect_docker.py's --train-only guard); its derived root serves training only (§11.5, §16). **D3** = the eval-split verdict's miss-mode exclusion rule — error/absent-outcome observations are excluded from fitness and carried report-only (§4.6, §11.2).

**OPEN (pilot-frozen, rule committed, number pending)**: canonical-PID scheme (100+10j vs seeded-stable arbitrary — watermark risk priced by the DG-8b position probe; freeze at prereg); DG-7 reference constant (v2 cross-split near-dup base rate — MUST be measured on dockerfs2 BEFORE v3 work starts); exact axis-2′ threshold (rule committed: max-margin midpoint recalibrated on v2 known cases — verdict:composition S3); uptime's slot and the hard-link LN-CONTRAST motif (pilot-measured, drop if invisible); G-SEP's separation margin δ (tracker-vs-within_traj gap under counterfactual foils — measured at P3, frozen by amendment).

---

## 3. Substrate & infra

### 3.1 Exec model (winner: time memo; verdict:cross-lens D-B1)

Per-command `docker exec` survives as the ONLY executor. `realenv/docker_env.py` DockerBox is extended, not replaced; no `realenv/pty_env.py` in v3 (shelved to the v4 spike with its verdict's fixes attached). Surviving infra-memo contributions re-anchored onto exec: fresh-container-per-trajectory, probe-container split, digest gates, cost/HF sizing method, ps-availability probing (lens:infra).

- **Fresh container per trajectory**: `docker run -d --rm --network none --init --hostname box-<image> -m 512m --cpus 1 --label tj3-mint=<seed> <digest-ref> sleep 86400`; torn down after each sequence. Mutation pollution structurally dies with the container (charter; lens:world-dynamics). `--init` (tini) guarantees orphan reaping (lens:time). The time memo's cross-sequence reset protocol is DELETED, not hedged (verdict:time M5).
- **exit codes**: exact from subprocess returncode; 124 = timeout (mint gate: zero occurrences), 125 = host executor error (aborts the image — it is an F1-class host artifact, never a world observation; verdict:targets M2).
- **stderr folding**: `output = stdout + stderr` (docker_env.py:82) is retained world behavior; the pipe mode rule is designed around it (§6.4).

### 3.2 Tick prologue and the cd branch

The existing unrecorded prologue (`cd <cwd> 2>/dev/null; <cmd>`) extends to: `<fire-script for jobs due at this vt>; <post-signal barrier if pending>; cd <cwd> 2>/dev/null; <recorded cmd>`. Sequencing inside one `sh -c` guarantees every due delayed effect commits before the recorded command runs — race window structurally zero (lens:time). **Prologue scaffolding is UNRECORDED by definition** (the v2 `cd <cwd>` prologue precedent): fire-scripts and barriers embed real PIDs and polling loops and are never part of the recorded command string, so recorded≡executed applies to the recorded string only — PID canonicalization (§5.2) remains the single declared bend of that law. The `cd` special-case branch (docker_env.py:74–80) gets the SAME prologue integration so a job due at a cd step fires on time (verdict:time M1). The `echo <vt> > /tmp/.tj/vt` tick-carrier is dropped as dead mechanism — firing is collector-decided (verdict:time M6); vt is collector state only.

### 3.3 The `after` helper (repaired per verdict:time S1/S2/S3)

Installed once per container in-band at `/usr/local/bin/after` (unwritable ⇒ the bgjob verb is unavailable for that image, v2 skip-redistribution). Container bootstrap (in-band, unrecorded, once per container): `mkdir -p /tmp/.tj` runs BEFORE the helper install and before any first `mkfifo`; the bootstrap also runs an in-band PID probe that learns the real PIDs of tini and the keeper (`sleep 86400`) to seed the canonical-PID map (§5.2) — job real PIDs enter the map via the recorded launch step's `echo $!`. Fixes over the memo spec:
- **fd hygiene**: the stdio redirects live in ONE place — the helper's line 1 (`exec </dev/null >/dev/null 2>&1`), which closes the inherited pipe fds within milliseconds of the fork. The RECORDED launch string is exactly the §6.1 BNF form `after j K 'effect' & echo $!` — no redirects — so the frozen BNF, the SST parser-totality assert, and recorded≡executed all hold with one canonical shape (round-5 M2 ruling; supersedes the earlier both-places wording). Measured 0.12s vs 2.11s per launch without the helper-internal exec (verdict:time S1). Cost model uses 0.15s/launch.
- **explicit job index**: the collector passes `j` in argv — recorded form `after <j> <K> '<effect>' & echo $!` (recorded ≡ executed; j and K are deterministic semantics, not nonces). No `/tmp/.jn` self-assignment race.
- **registration barrier**: the next step's prologue blocks (bounded) on fifo `/tmp/.tj/g<j>` existing before proceeding; timeout ⇒ abort trajectory (F5 semantics).
- **post-signal barrier** (verdict:time S2): after any `kill` step, the NEXT recorded step's prologue carries a bounded in-band wait for the expected `/proc/<pid>` state (gone after TERM-on-running/KILL; `T` in stat after STOP; running after CONT) before the recorded command runs — the `<post-signal barrier if pending>` slot of the §3.2 prologue, composed by the collector and injected through the §16 prepend-only seam. Placement ruling: the barrier NEVER lives in the kill step's own script (it is unrecorded prologue scaffolding, §3.2); the barrier-bearing NEXT step declares the +5s extra_timeout and its dur_ms absorbs the wait. Determinism by construction, not exec-overhead luck.
- **schedule-aware launch guard** (verdict:time S3): the collector only draws (launch_vt, K) satisfying launch_vt + K + max_possible_deferrals + δ ≤ L−1; K ∈ {2,3,5,8} remains but K=8 draws only when the bound holds. "Every fire is observable" becomes arithmetic, not hope.
- **watchdog**: `after` self-kills at 5s real-time if fired while wedged (deadlock valve under the 8s exec timeout); the self-kill lines are IN the normative script below.

Concrete shapes (normative):

```sh
# helper /usr/local/bin/after (installed in-band once per container; the bootstrap
# has already run `mkdir -p /tmp/.tj`, so the mkfifo below cannot precede its dir):
#!/bin/sh
# usage: after <j> <K> <effect...>   (K is display semantics; firing is collector-driven)
exec </dev/null >/dev/null 2>&1
j="$1"; shift 2
mkfifo "/tmp/.tj/g$j" 2>/dev/null
read _ < "/tmp/.tj/g$j"          # block until the collector's fire-script writes the gate
( sleep 5; kill -9 $$ ) &         # watchdog: self-kill at 5s real time if the effect wedges
w=$!
sh -c "$*"                        # run the effect
: > "/tmp/.tj/d$j"                # done-marker
kill "$w" 2>/dev/null             # disarm the watchdog

# recorded launch step — the ONE canonical shape, identical to the §6.1 BNF (fd hygiene
# lives inside the helper's line-1 exec, never in the recorded string; round-5 M2 ruling):
after 1 3 'echo <mined-token> >> /tmp/w/task1.log' & echo $!

# fire-script fragment injected into the prologue of the step at which job j is due:
echo go > /tmp/.tj/g<j>; t=0
while [ ! -e /tmp/.tj/d<j> ] && [ $t -lt 50 ]; do sleep 0.1; t=$((t+1)); done
while [ -d /proc/<realpid> ] && [ $t -lt 50 ]; do sleep 0.1; t=$((t+1)); done   # reap barrier
[ $t -lt 50 ] || exit 97    # 97 = barrier timeout -> collector aborts the trajectory

# post-signal barrier — injected into the NEXT recorded step's prologue after any kill step
# (placement ruling §3.3: never in the kill step's own script; unrecorded scaffolding):
#   TERM-on-running / KILL: wait for /proc/<pid> gone
#   STOP: wait for `grep -q '^State:.T' /proc/<pid>/status`
#   CONT on stopped-pending-TERM: wait for /proc/<pid> gone; else wait for running state
```

Worker layout unchanged from v2: ThreadPoolExecutor, one thread per image, trajectories sequential within a thread, 6 workers × (--cpus 1, 512m) containers. Teardown: `docker rm -f` per trajectory; a `finally` sweep collects orphans by the `tj3-mint=<seed>` label.

Infra artifacts as world state (verdict:time M2): control files live under ONE dotdir `/tmp/.tj/` (fifos `g<j>`, done-markers `d<j>`); `/usr/local/bin/after` is a real inspectable file. They are deterministic policy-constants, so they are NOT render-filtered (honesty: the world is what it is); they ARE (a) excluded from all mining pools, (b) untouchable by every mutation arm (denylist §4.4), (c) known to the DG-3 nondeterminism scanner, (d) expected to fail axis-1 if any observation is dominated by them (constant across images ⇒ echo/const, excluded).

### 3.4 Timeouts, dur_ms, shadow probes

- Timeout T = 8s + declared `extra_timeout`. `extra_timeout` owners: sleep-bearing commands, AND every barrier-bearing step — a step whose PROLOGUE carries a fire-script/reap barrier or a post-signal barrier (the only barrier home, §3.3 placement ruling) declares +5s per barrier, computed by the collector's step assembler (each barrier can legitimately consume up to ~5s; without the declaration a spurious 124 would trip the zero-timeout gate and abort the image). **Zero-timeout mint gate: abort wiring installed at collect() entry (the daemon_errs mirror), asserting zero timeout events at completion** (Amendment-7 lesson; pre-mortem DG-10b), not a runbook step, measured against T + extra_timeout. Any timeout at mint aborts the image.
- `dur_ms` is NEW instrumentation (nothing in v2 times commands): `DockerBox.run` gains a monotonic-clock capture around the exec subprocess and returns `dur_ms` in the step dict; the collector strips it from the recorded step and appends `{seq_idx, step, dur_ms}` to the side-channel file `timing-<split>.jsonl` (gitignored, excluded from replay byte-diffs) — wall-clock in the diffed bytes is irreproducible by construction (verdict:infra S2). It never enters renders. Both the docker_env capture and the writer are §16 diff items.
- Shadow/audit probes (`--audit-shadow`): pilot-only, and **fenced**: class-measurement and determinism pilots run shadow-OFF; shadow-ON collections are a separate audit arm that never feeds class freezing (verdict:infra S3). At-mint in-band probe count for verification purposes = zero (verdict:cross-lens D-S4); fs_delta comes from the tracker (§8.2).

### 3.5 Probe container & probe cache

One legacy exec DockerBox per image, pristine, run before its trajectories (lens:infra). Probe products (immutable per-image dict): availability (`command -v` over the full v3 verb set), **ps tier probe** (T2 `ps -o pid,stat,args` / T1 `ps -o pid,args` / T0 plain / ABSENT — the explicit absent tier per verdict:time M3; note modern busybox ≥1.31 supports `-o stat`, so alpine-family lands T2 — verdict:time S5 corrects the memo's stale claim), `/usr/local/bin` writability, `/bin/sh` identity (readlink -f — bash/dash/ash; version identity), sizes/wc/K-pools/find-pairs as v2 **with match counts recorded** (fixes verdict:composition S2 even though find-producers are pruned), ls-child-count batch for pipe floors, and the **error-template harvest**: the closed template battery (cat/ls/cd/rm/mv/head/tail/grep on a guaranteed-absent path) captures each image's exact error strings per verb; recorded in the per-image report and frozen into summary.json. The SST consumes this frozen per-image table instead of a hand-authored {gnu,busybox} enum — resolving the shell-dialect blindness (bash/dash/ash emit different cd/redirect error text) and the opener-detection fragility (verdict:baselines S2c, S4).

**Probe cache key = (image digest, policy content-hash via lexicon_hashes(), collector version)** with a cold-vs-cached dict-equality twin check in pilot (pre-mortem DG-10c; verdict:cross-lens M9; verdict:world-dynamics S5).

### 3.6 Per-sequence seeding & replay unit

RNG stream per sequence: `random.Random(f"dockerfs:{seed}:{image}:{arm}:{seq_idx}")` (v2's single per-image stream made per-sequence replay ill-posed — verdict:eval-split S4; verdict:world-dynamics S5). Sequence k is a self-contained replay unit: fresh container + per-seq seed + probe dict ⇒ (digest, seed, arm, seq_idx) → bytes. The L5/L6 replay audits become cheap by construction.

### 3.7 Cost model (one budget table; owner: the runbook; resolves verdict:completeness S1, verdict:cross-lens D-S11)

Per-step: ~90ms exec (v2 measured) + ~30ms amortized container churn (~0.85s spawn+teardown / 28 steps) + launch/signal barriers ≈ **~120ms/step**; pilot-measured on 2 images before the runbook freezes.

| item | size | wallclock (6 workers) |
|---|---|---|
| mint, full arm | 12 img × 900 seqs × 28 steps ≈ 302K steps | ≈ 1.7h |
| mint, ablate arm (train-only, half seqs) | 8 img × 450 × 28 ≈ 101K steps | ≈ 0.85h (8 images on 6 workers ⇒ 2 serial rounds on two threads; matches §1) |
| container spawns | ~14.4K (10,800 full + 3,600 ablate; one per trajectory) at ~1.6/s aggregate | inside the above — pilot spawns budgeted in the pilot rows |
| pilots P1/P2 (≤6 iters × ~24 seqs/img × 12) | ~8K steps each | ~3 min each (same ~120ms/step model that validates the mint figure) |
| pilots P3/P4 (60 seqs/img × 12; P4 P3-sized) | ~20K steps each | ~7 min each |
| twin-mint determinism doubles (DG-3a) | pilot scale only, 2× two pilot runs | ~12 min total |
| e5 encode (mint + ablate obs/cmd texts + per-step delta texts) | ~1.2M texts (806K obs/cmd + 403K delta) | ≈ 9–13h, once |
| precompute_baselines (SST/wtm predicted texts, FULL val = inner + final) | ~200K texts | ≈ 2h, once |
| re-baseline: 6 rungs × 3 seeds × 4000 steps + the subsampled-train champion arm (§11.5: champion on the full root subsampled to 450 seqs/image, 3 seeds) | 21 full trainings | ≈ 1–2 days |
| instruments (probes, ROLL@k) | full-score only | minutes per score |

Publication: raw ≈ 1.2–1.6 GB, e5 ≈ 2.0–2.4 GB, ablate raw ≈ 0.5 GB ⇒ ≈ 3.7–4.5 GB total, max file ~1.6 GB — inside HF norms (lens:infra §6). Mint budget gate: ≤3h collection (full 1.7h + ablate 0.85h = 2.55h against it); total program ≈ 2–4 days machine time. The infra memo's 350–450K floor arithmetic error is corrected (12×800×28=268.8K; the committed 900 seqs/image gives 302K — verdict:infra M4).

---

## 4. The world: verbs, mutation system, workspace & tiers, provenance law, motifs & revisit controller

### 4.1 Verb inventory (v3 toolset)

v2's nine (uname, cd, ls, cat, head, tail, stat, find, grep) + new: **pwd, echo (redirect producer + ~1% bare), rm, mv, ln/ln -s, readlink, mkdir, touch, ps, kill (signal family), after (bgjob launch), uptime, sleep** + the G3 composed families (§6). `jobs`/`fg`/`bg`/`wait` excluded (UD-1).

### 4.2 Two-tier namespace

- **Tier W — workspace `/tmp/w`** (ONE arena; verdict:cross-lens D-S3; the `/w`-with-fallback design lost — its fallback broke cross-image command identity, verdict:world-dynamics M1). Setup is in-band/unrecorded. Seeding = the sequencing memo's Zone-W mechanism, ADOPTED as the DG-1-compliant design (verdict:cross-lens D-B4): copy a crc32-selected, image-specific sample of ~12 small real files into `/tmp/w` under **normalized names from one sha-hashed filename lexicon** (merged from the two memos' lexicons) — command identity is cross-image (the class protocol needs identical command strings), content is image-specific (the held-out-image split keeps biting) (verdict:sequencing M2). Manifest hash into step-0 meta.
- **Tier S — expendable real files**: enumerate()-derived, regular files ≤64KB, minus the denylist (§4.4). Ops: `rm <f>`, `mv <f> <f>.bak|.old|.orig`, `mv <f> /tmp/w/`. **Destination absence is verified** by enumerate()'s list PLUS an in-band `[ -e ]` probe at generation (the ledger alone cannot know image-shipped `.bak` files — verdict:world-dynamics S2).

Caps: ≤8 mutations/seq, ≤4 Tier-S; interventions/seq 6±2 (pacer); no motif starts in the last 5 slots.

### 4.3 Provenance law (revised T3; verdict:cross-lens D-B4, pre-mortem DG-1)

**Every byte written into the workspace is image-grounded**: echo payloads are mined from THIS trajectory's render-visible observation prefix (V2_MINE_CAP=500 discipline; the visibility-parity unit test asserts nothing past the render cap enters any pool — pre-mortem DG-4c). A small audited lexicon arm (≤5% of writes) carries `meta.payload_src="lexicon"`; all others `"mined:<step>"`. Payload charset excludes backslashes and quotes beyond `_sq()` handling (busybox echo interprets escapes; verdict:baselines S3). No wallclock/PID/nonce may enter a payload (grep audit at pilot). The payload-mining regex is PINNED to v2's `_TOKEN_RE` form (`[A-Za-z_][A-Za-z0-9_.\-]{3,23}`, collect_docker.py) — the leading `[A-Za-z_]` structurally prevents a mined token being eaten as an `-n`/`-e`/`-E` flag by busybox/dash echo. The time lens's job-effect vocabulary switches from lexicon tokens to mined tokens (`echo <mined-token> >> /tmp/w/task<j>.log`) — the process arena is thereby image-grounded too (verdict:time S4). DG-1 pilot gate: delayed-copy predictor top-1 on revisit reads < a pilot-measured per-family threshold, frozen by amendment (a TOP-1 accuracy bound — NOT the axis-2′ containment number, which is a different statistic); cross-sequence payload duplication < 5%.

### 4.4 Denylist (union; verdict:cross-lens D-S6)

LOAD_BEARING = {/bin, /sbin, /usr/bin, /usr/sbin, /lib*, /usr/lib, /usr/libexec, /usr/local/bin} ∪ every `command -v`-resolved tool path ∪ loader/musl/busybox globs ∪ **NSS/identity databases: /etc/passwd, /etc/group, /etc/shadow, /etc/nsswitch.conf** (mutating passwd flips every later `ls -l`'s uid→name resolution image-wide — an observation dependency no path-keyed ledger can represent; verdict:world-dynamics S1) ∪ infra artifacts (/usr/local/bin/after, /tmp/.tj/*). MutGuard re-validates every mutation string against templates + denylist; violation ⇒ collection abort (collector bug). MutGuard is homed in `collect_docker.py` (collection-side only, never eval-path; §16 property tests).

### 4.5 Pool routing through the tracker (verdict:world-dynamics S3)

ALL v2 draw pools — seen_paths linkage, cat band-cyclers, the CONFIG_FILES arm, probe-verified find pairs, grep small_files/transplant pools — filter through the collection-mode tracker's live-state view (tombstones for deleted/moved-away paths; find pairs re-verified by glob-match against the tracker's touched set). Absent-path probes exist ONLY as explicit meta-labelled revisit arms. The expected-vs-realized mismatch counter gates the pilot; its threshold is set AFTER measuring the benign base rate (target ≤1%, but pilot-measured first — verdict:world-dynamics M3).

### 4.6 Motifs & revisit controller (revised mix; verdict:cross-lens D-B2)

Motif templates (normative; probes queued at scheduled offsets, all probe targets DG-2-scoped):

| motif | intervention | scheduled probes (content-bearing branch bolded) |
|---|---|---|
| CUD chain | `echo '<mined>' > /tmp/w/<name>` | **`cat /tmp/w/<name>`** (payload only in an earlier cmd text — SST-covered, credit zeroed; blind-capture variants use `PROD > WSF`); later `rm`; `cat` again (error, report-only) |
| MV-displacement | `mv <tierS-F> <F>.bak` | **`cat <F>.bak`** (content transported — the deep dynamics fact), `cat <F>` (error, report), **`ls -1 <small-parent>`** (listing delta) |
| RM-listing | `rm /tmp/w/<name>` \| `rm <tierS-F>` | **`ls -1 <parent≤8 entries, ≥1 survivor>`**, `cat <target>` (error, report), `[ -f <target> ] && cat <target>` (COND miss, excluded/report) |
| LN chain | `ln -s <F> /tmp/w/<link>` | **`readlink /tmp/w/<link>`**, **`cat /tmp/w/<link>`** (= F's content); chain ext p=.35: `rm <F>` → `cat <link>` (dangling error, report) |
| LN-CONTRAST (low rate) | hard `ln` twin of the above | **`cat`-through-hardlink after `rm`** (content survives — the ln vs ln -s discrimination); pilot-retained-or-dropped |
| APPEND chain | `echo '<mined>' >> /tmp/w/<name>` ×1–2 | **`cat`/`head -n K` post-append** (content = ordered function of ≥2 history steps) |
| JOB chain | `after <j> <K> '<effect>' & echo $!` (the §6.1 canonical form) | pre-fire check p=.3 (**absence-then-presence must be timed**), **`cat /tmp/w/task<j>.log`** at fire+δ, `ps` after launch/kill/fire (need_ps boost) |
| UNMUT control | none | re-read of an untouched path after unrelated mutations, expected=persist |

Revisit controller targets (realized-rate gated ±5%):
- ≥35% of post-first-mutation read steps are revisits;
- **revisit mix ≈ 15% absence/error (report-only channel — they are miss-mode under the eval-split D3 exclusion rule (§2 notation) and carry no fitness) / 55% displaced-or-created CONTENT (the fitness-bearing branch: `cat B` after `mv A B`, readbacks of created files, listing deltas) / 20% indirection (through symlink; ls of parent) / 10% unmutated-persistence controls** (revisit mix re-weighted from the memo's 40/40/20 because the error branch is structurally excluded by D3 — verdict:cross-lens B2/D-B2);
- ≥60% of Tier-S targets pre-observed (before/after contrast), ~40% deliberately unobserved (the slice where tracker+copy hybrids cannot transport content — the honest-margin surface);
- probe-outcome mix per DG-8: {hit .70, miss-reverted .20, miss-never-mutated .10} across ALL motif arms (the grep-miss lesson applied to probes; kills "probe ⇒ hit" schedule priors). Denominator note (the 15%-vs-30% reconciliation): the 15% error band is over ALL post-first-mutation revisit reads; the {70/20/10} mix is over SCHEDULED PROBES only — and misses are not all errors: a miss-reverted probe renders as an error only when the revert deleted the path (rm-revert), while mv-revert and miss-never-mutated probes render as ordinary reads of restored/untouched content, so the render-as-error share of scheduled probes is ~15%, jointly satisfiable with the revisit band; both realized rates are gated (G-RATE, DG-8);
- **probe scoping (DG-2 corollary, policy law)**: listing probes target dirs ≤8 entries OR probe the mutated path directly, and probe dirs keep ≥1 surviving entry so post-mutation listings are never empty (an empty-output ls would be D3-excluded; verdict:eval-split M1). The naturalism cost of small-dir probing is flagged for review B (pre-mortem risk 2).

Timestamp policy for probes: revisit-ls draws only TIME_FREE_LS_OPTS {"", -1, -a, -1a}; `readlink` replaces `ls -l` for link observation; `stat` pinned to the time-free v2 format `%n %s %F %a` (verdict:sequencing S2). Full field policy in §5.5.

---

## 5. Time & processes

### 5.1 Virtual step-clock

vt = index of the recorded step; one tick per step, carried by collector state and the fire-script prologue (§3.2). Foreground `sleep {0,1}` only, weight ~0.005, decorative. `uptime` renders `up <vt>` + fixed masks (clock→00:00:00, users→0, load→0.00 0.00 0.00), weight ~0.01, expected excluded-or-thin (measured; note /proc/uptime is host-wide, not container-reset — verdict:cross-lens M6 corrects the memo). Jitter-invariance gate (§14 G-J): two pilot collections, one with injected 0–300ms inter-step delays ⇒ byte-identical jsonl.

### 5.2 Canonical PIDs (bidirectional virtualization)

Real PIDs are replay-chaotic; the world presents virtual PIDs: reserved tini→1, keeper→2, job j→100+10j (scheme OPEN, §2; the DG-8b position-probe prices the step-index watermark risk — verdict:time M4). The real→canonical map is seeded by the §3.3 in-band bootstrap probe (tini/keeper real PIDs) and extended per job via the recorded launch step's `echo $!`. Applied to observations before storage AND in reverse to commands (`kill -STOP 110` translates at exec, records canonically). Raw PIDs never stored; `meta.n_pid_renames` kept. This is the one declared bend of recorded≡executed; unit-tested + covered by the jitter gate.

### 5.3 The kill-signal process automaton (repaired per verdict:time B1)

Signal set: `kill <cpid>` (TERM), `kill -STOP`, `kill -CONT`, `kill -9` (hard kill; works on stopped jobs), `kill -0` (liveness probe). **`kill -INT` is dropped** — POSIX sets SIGINT to SIG_IGN in async children of non-interactive shells; it is a verified silent no-op (verdict:time B1a). Automaton states: **{waiting, stopped, stopped-pending-TERM, fired, killed}** — TERM on a stopped job does NOT kill it; it stays `T` and dies at CONT (verified; verdict:time B1b). The collector's fire-deferral bookkeeping, meta labels, and the SST job simulator all share this 5-state machine from the ONE tracker module (§10), so mislabeling-by-design is structurally impossible. Deliberate-miss arm (~20% of kills): kill a cpid already fired/killed ⇒ "No such process", exit 1, meta.intended_miss — a read of mutation history.

### 5.4 ps handling

Per-image tier from the P0 probe (§3.5): T2/T1/T0/ABSENT; recorded per-step (`meta.ps_tier`) and per-image; class constancy measured per tier group (cross-image command identity is tier-conditional, like v2's stat fallback). Render rules (frozen canonicalizer): drop self/transient rows (exec wrapper + the ps invocation; unit-tested against adversarial recorded commands containing the marker strings — lens:time risk 3), virtualize PID column, TIME→0:00, no etime/elapsed ever requested. A job row displays the pending effect and delay — process structure is both predictable and predictive. DG-5 entropy gates (§14) prevent the find-97.6%-empty reincarnation: distinct renders ≥25/image, max single render ≤25%, job-state mixture {running, stopped, terminated, mixed} each ≥15% of ps reads (controller-steered), majority-class exit accuracy recorded as the probe baseline.

### 5.5 Per-field nondeterminism policy (ONE table; owner: time lens; verdict:cross-lens D-S7, verdict:verb-classes B1)

Three layers, and the flip-rate statistic is measured **across replayed collections (the twin-mint), not same-session twins** — back-to-back twins are blind to exactly the replay-random fields (mtimes of created files flip 0% within a session and ~100% across replays; verdict:verb-classes B1):

| field | layer | policy |
|---|---|---|
| PIDs (everywhere) | store-time | virtualize (canonical, bidirectional) |
| uptime elapsed | store-time | virtualize → vt |
| mtimes of trajectory-mutated/created paths | store-time | virtualize → `T+<vt_of_mutation>` |
| mtimes/dates of untouched shipped files | — | leave raw (image-constant facts) |
| runtime-mount mtimes (resolv.conf, hostname, hosts) | render-canon | mask (fresh per container start ×14.4K containers — verdict:world-dynamics B2a) |
| wall clock, load, users, cpu TIME | render-canon | mask to fixed tokens |
| parent-dir mtimes of mutated dirs (`ls -l` family) | render-canon | mask time fields on `-l`-family renders of tracker-touched dirs (verdict:world-dynamics B2b, DG-3d) |
| inodes, size-tiebreak orderings | template avoidance | `-i`,`-lt`,`-lS` dropped from LS_OPTS for mutation-adjacent steps; TIME_FREE opts for revisits |
| etime/ELAPSED, scheduling order | excluded | never requested / closed by construction |

**Mask placement ruling** (resolves verdict:completeness S3): store-time virtualization happens in the collector (raw never stored). Measured render masks live in a NEW frozen module `realenv/render_canon.py`, applied inside `reencode.py`/`encode_split` BEFORE any perception chunk's render — sha-pinned eval-path code under the §7 Darwin-Gödel guard. Input contract: `render_canon.canon(step, state) -> step'`, where `state` is the collection-mode tracker state at that step — reencode RE-FOLDS `realenv/shell_state.py` over the RAW jsonl per sequence before any render, so the touched-set the `-l`-family mask needs is recomputed at encode time from the stored records themselves (no meta flag is trusted — F8). A genome-selected perception impl cannot skip the masks. Any change to render_canon is version identity.

Fields that still differ in the twin-mint after this table are enumerated into the mask list or their (sig, mode, opt-family) cell goes excluded-nondet — **scoped per opt-family, never per verb** (flunking `ls -l` on /tmp/w must not eject all of ls; verdict:eval-split S3).

---

## 6. Composition grammar G3 (revised per verdict:composition)

### 6.1 The grammar (frozen at prereg; exactly one operator; no recursion)

```
COMPOSED ::= PIPE | REDIR_W | COND
PIPE     ::= PROD "|" FILT
PROD     ::= "ls -1" D | "cat" F                  # find-producers pruned (D-S2); ls -l banned
FILT     ::= "head -n" K | "tail -n" K | "grep -F -m 8" TOK
REDIR_W  ::= PROD (">"|">>") WSF | "echo" 'PAYLOAD' (">"|">>") WSF
COND     ::= "[" TESTOP P "]" "&&" READ P         # TESTOP ∈ {-e,-f,-d,-s}; READ ∈ {cat, ls -1, head -n K}
```
Excluded: `<<<` (non-POSIX), `||`, `;`, `if/then`, depth-2 pipes, REDIR_IN, `wc -l`/count filters, `cd` in any composed string (docker_env cd-tracking), `ps` in pipes (D-S12; `ps | grep sleep` deleted from m_job). The prereg BNF's totality universe is the FULL command grammar — the v2 atomic templates (the ≥60% majority mass) + this G3 grammar + the audited process forms (`after j K 'effect' & echo $!`, kill family); the SST parser asserts totality against that whole universe (verdict:cross-lens D-S13/S14; verdict:baselines S3).

### 6.2 Families and floors

The complete signature inventory (meta.sig vocabulary; `composed_verb()` emits exactly these plus first-token verbs for simple commands):

```
pipe:ls|head  pipe:ls|tail  pipe:ls|grep   pipe:cat|head  pipe:cat|tail  pipe:cat|grep
redir:echo>   redir:prod>   cond:cat       cond:ls        cond:head      echo (bare, ~1%)
```

11 measurable composed families — 6 pipe + 2 redir + 3 cond (down from ~18 after pruning REDIR_IN + find-producers — verdict:cross-lens D-S2); bare `echo` is a simple verb (§4.1), not a composed family. Modes per family: pipe-grep {hit, miss}; cond {hit, miss}; others {ok}. Coverage: ≥30 cross-image pairs per (sig, mode, state_scope) cell else the fixed cross-image battery. Pipe-grep tokens: ~60% linked (mined from THAT producer's earlier render-visible output, v2 mine rules, meta.linked/query_src), ~20% transplant misses (meta.intended_miss, never -i), remainder lexicon. K = v2_k_for(target) for real files; **K for workspace files derives from tracker-known content line counts** (the probe container is pristine and cannot verify created files — verdict:cross-lens M11, verdict:sequencing M6); 2K+1 floor enforced from tracker content for workspace targets and from wc/child-count probes for real targets. Syntactic-equivalence pairs (`cat F | head -n K` vs atomic `head -n K F`, same F, both in-corpus) capped at ~10% of the pipe arm — a standing surface-invariance report slice.

### 6.3 Redirection state

Writes route MutGuard + register in the one tracker; observation empty+exit0 ⇒ measured ack/excluded; payoff is the readbacks. Readback controller: ≥70% of writes get ≥1 readback; append chains p≈0.35 (content becomes a function of ordered history); readbacks keep atomic families with `meta.ws_target=True`; **readback ls restricted to names-only forms (-1/-a)** (fresh mtimes otherwise break replay — verdict:composition B3). ws_observed stratification (observed-capture vs blind-capture ≈ 50/50): the SST saturates observed-capture; blind-capture (`ls -1 /usr/lib > f` where /usr/lib was never listed) is provably tracker-blind — the honest composed-margin surface. Review-B replay audit confirms the SST actually fails on the blind slice (lens:composition risk 5).

### 6.4 Mode rules for composed steps (repaired per verdict:composition S1)

Pipeline exit = last stage (POSIX, no pipefail). Because docker exec folds stderr into output, a failed producer + succeeding filter yields exit 0 + non-empty error text — NOT detectable from (exit, emptiness) alone. Therefore: **pipe producers are drawn only from tracker-live, existence-verified paths**; misses exist only via the explicit intended-miss arm on the FILTER token (transplanted grep tokens), which the (exit==0 ∧ empty) rule labels correctly. Producer failure at collection is an expected-vs-realized mismatch ⇒ pilot gate. COND: hit (exit 0, non-empty READ output) ⇒ content; miss ⇒ excluded, reported.

### 6.5 The echo-guard, at slice level (repaired per verdict:composition B1)

Axis-2′ (history containment: rendered obs 3-grams vs the union of earlier COMMAND texts) is measured **per family AND per the ws_target/readback slice** — family medians cannot dilute the echo channel. Threshold: the committed RULE is the max-margin-midpoint procedure recalibrated on v2 known cases (linked-grep hits = content side, cd = echo side); 0.656 is the prior, the pilot freezes the number by amendment (verdict:composition S3). DG-6 derivability (obs is a substring/line-subset of earlier same-trajectory RENDERED observations) is a second column; the SST's mandatory G3 evaluation over belief state (SST-G3 exact-match ≥0.95 on derivable composed steps, else family ⇒ sim/excluded) makes string-slicing credit structurally impossible (verdict:composition B2). Both columns live in class_measure.py (verdict:cross-lens D-S13). The aggregate-max weakness (a slice-perfect tracker never becoming the pooled max) is closed by the **sst_composite arm** (§10.3): composite = SST-where-determined else within_traj_mut, so slice-local SST perfection zeroes slice credit step-wise.

### 6.6 Corrected claims

"echo never earns fitness itself — its readbacks do" is amended: echo-readbacks whose payload transited command text die at the slice-level axis-2′; the echo arm's honest payoff is the excluded-class record, tracker calibration, and blind-capture readbacks (verdict:composition M4). Curriculum stays training-side (Role-A batcher chunks over meta.family); the world ships the full grammar at once (one-mint + paired-collection).

---

## 7. Sequencing policy gen_sequence_v3 (ONE number set)

### 7.1 Architecture

The v2 flat verb-mixture wrapped in an event scheduler: the collection-mode tracker + a probe event queue; motifs emit interventions immediately and schedule probes at future offsets; interleaving produces long-range causal dependencies (lens:sequencing). All arms run on exec; m_job rewritten onto `after`; m_ctrl deleted; m_cond re-templated to G3 COND (verdict:cross-lens D-B1/D-B2).

### 7.2 The reconciled numbers

- **seq-len 28±4, max 32 steps = 64 tokens** — fits pos_emb(64) and the constitution §6 ASSERT; no version-identity table change (verdict:cross-lens D-S1). Delay mixture compressed: mutations by step ~14, probes to step ~27; delays {1–2: .40, 3–8: .35, ≥9 (≤14): .25}.
- **900 seqs/image × 12 images ≈ 302K steps**; ablate arm 450 seqs/image train-only.
- **Weight budget (owner: sequencing lens; verdict:cross-lens D-S2)** — the per-arm start-weight table below is the SINGLE SOURCE OF TRUTH; every aggregate here is derived from it. Old-verb plain (unpiped, atomic v2 verbs incl. the 3 openers) **≥60%**; new mass ≈39%. Booking rule: m_redirect (.050) is G3 REDIR_W and is booked under composition ONLY — the earlier draft double-counted it under mutation:

| arm | mass |
|---|---|
| mutation atomic (mkdir/touch/rm/mv/ln/readlink + revisit probes' interventions) | ~10.4% |
| composition (pipes ~7.5%, REDIR_W ~5% = m_redirect echo>/prod>+appends, COND ~2.5%) | ~15% |
| time/process (after-launch ~3%, ps ~3%, kill family ~2%, uptime ~1%, sleep ~0.5%) | ~9.5% |

uptime appears ONCE (double-count removed). pwd follows ~30% of realized cds (predicted sim — axis-3, rule 4 — capped). All realized rates gated ±5% v2-style.

Full per-arm start weights (loop steps after the 3 v2-style openers; controller-steered, realized ±5%):

| arm | weight | | arm | weight |
|---|---|---|---|---|
| flat cd | .105 | | m_rm | .035 |
| flat ls (plain/TIME_FREE) | .140 | | m_mv | .035 |
| flat cat | .100 | | m_ln (+readlink probes) | .022 |
| flat config-cat | .025 | | m_redirect (echo>/prod> + appends; booked under composition = REDIR_W) | .050 |
| flat head | .048 | | mkdir/touch | .012 |
| flat tail | .042 | | pipe families (6) | .075 |
| flat stat (time-free fmt) | .042 | | cond families (3) | .025 |
| flat find | .048 | | redir readbacks (atomic, ws_target) | in atomic mass |
| flat grep | .060 | | after-launch | .030 |
| pwd (follows cd) | .031 | | ps | .030 |
| uptime | .010 | | kill family (TERM/STOP/CONT/9/-0 + miss arm) | .020 |
| sleep {0,1} | .005 | | echo-bare | .010 |

Derived from the table (the inline check): atomic v2-verb plain mass = .105+.140+.100+.025+.048+.042+.042+.048+.060 = **.610 ≥ the .60 floor** ✓; mutation atomic = .035+.035+.022+.012 = **.104**; composition = .075+.050+.025 = **.150** (pipes .075 + redir .050 + cond .025 — m_redirect booked here ONLY, it is G3 REDIR_W); time = .030+.030+.020+.010+.005 = **.095**; pwd .031 (≈30% of cd's .105); echo-bare .010. **Total = .610 + .104 + .150 + .095 + .031 + .010 = 1.000 exactly.** Motif pre-emption draws against these arms' budgets (a due motif action replaces the weight draw), so realized ≈ start within the ±5% gates; skip-redistribution (unavailable verbs) flows to flat cat, recorded per image.
- interventions/seq 6±2; probe coverage ≥0.75 of mutations; pending-probe cap 4; flush zone last 5 slots; error-outcome obs rate 0.10–0.20 (report channel); jobs ≤3/seq with the §3.3 launch guard; chain-depth d2 target 0.15±0.05 — COMMITTED (removed from the §2 OPEN list; start-weight arithmetic honest — verdict:sequencing M5; chain extensions count against N_mut).

### 7.3 DG-8 probe-outcome arm and the schedule-leak guard

Every scheduled probe carries `meta.intended_outcome ∈ {hit, miss-reverted, miss-never-mutated}` at the {70/20/10} realized mix; per-step delay randomization; the **axis-1+ measurement** (cmd + step-index + steps-since-last-mutation probe, no history content) runs at P3 with the hard ceiling: axis-1+ top-1 < 0.5× within_traj on the probe slice, else delays/coverage re-randomize and the pilot reruns (pre-mortem DG-8; verdict:sequencing S5).

### 7.4 Kill-boundary determinism

The memo's "deterministic by construction" job-set claim was false at the kill boundary (verdict:sequencing S1); resolved structurally: after every kill step, the NEXT recorded step's prologue carries the post-signal /proc barrier (§3.3 placement ruling), so the post-kill state IS a pure function of history by the time the next command runs. No `wait`, no jobspecs (UD-1).

### 7.5 Meta schema (one owner: sequencing lens; verdict:cross-lens meta-merge)

Per step, superset of v2: {verb, sig (≡family, one field — M3), mode, state_scope∈{native,mutated,created} (derived from mut_dep), mut_affected, mut_dep:[steps], arm, role, mut_id, probe_of, delay, chain_depth, zone, victim_observed, ws_target, ws_observed, payload_src, intended_outcome (one vocabulary — M2's `expected` enum merged), expected (tracker outcome class), signal (renamed from time's meta.sig — M1), job fields {j, K, launch_vt, fire_vt_planned, phase}, ps_tier, k/tok/band as applicable, sched:{due,issued}, pre_obs_step (int|-1, the step whose observation last rendered this step's target pre-mutation — feeds §8.1 counterfactual foils and the §13.1 cache column), delta_text (the collection-time tracker's canonical fs-delta AUDIT copy — recomputed and F8-asserted at derived-root build, §8.2)}. Touched-set propagation semantics are written law (§10.2). The render-inert-meta unit test (L4) is the safety condition; every new field joins its whitelist review.

---

## 8. Prediction targets & instruments

### 8.1 Fitness (unchanged shape)

Primary fitness = pooled content top-1 margin: `content_top1(WM) − max(aggregate arms)` over the measured content cells, on inner-val; pooled aggregation stays (harness continuity; verdict:cross-lens D-S8); per-verb margins and per-slice maxes are standing ledger columns; ≥500 eval steps per content verb per split enforced by mint stratification, bootstrap CIs in the re-baseline report. Foils: same-verb backbone + counterfactual injection on the mutation slice per UD-4(b) — the recommended option, pending user sign-off (§2). **Injection mechanism (the ONE v3 retrieval-code change, version identity under the constitution's Darwin-Gödel guard)**: `retrieval()` gains an optional `forced_foils` parameter — an [N, m] index tensor (m=8; −1 = none) built by `_data_tensors` from `pre_obs_step`; forced indices are expressed in FULL-array step positions; `content_retrieval` owns the SUBSET-SEAM TRANSLATION (round-6 S1): after computing the content-cell index list `ii` it (a) row-subsets forced_foils to `ii`, (b) remaps each forced VALUE from full-array position to subset position via the full→subset map, and (c) DROPS any forced index whose target step is outside the content subset (its embedding is absent from the subset `true` tensor — e.g. a pre-mutation render in an excluded cell); dropped slots fall back to sampled foils and are counted in a `cf_dropped` ledger column; a pilot floor on the REALIZED injected-slot rate over the mutation slice joins G-CF (if pre-obs steps land mostly in excluded cells, the injection must not silently evaporate — round-7 M3). Injection applies to the same-verb foil arm ONLY (the random-foil arm stays pure). The surviving forced indices REPLACE the first m of the 63 SAMPLED foil slots (64 candidates = 1 true + 63 foils) in EVERY round before `_rank_stats` (which is unchanged); UD-4's slot arithmetic reads the same way. Sha-pinned eval-path code; the archived-margin regression (§16) proves v1/v2 scoring is bit-identical with the parameter absent.

### 8.2 Side-channels in data (target-only; repaired per verdict:targets B1)

`exit_cls` [n] (frozen vocab at prereg: {0 ok, 1 err, 2 usage, 126 not-executable, 127 not-found; 124 timeout retained for pilots but zero at mint; 125 = host artifact ⇒ abort, never a class} — verdict:cross-lens M4, verdict:targets M2. 130, 137, 143, and 148 are ALL REMOVED under one no-producer rule (UD-1/§5.3), applied symmetrically: SIGINT has no source once `^C` and `kill -INT` are out; stopped jobs produce no exit at all (a `T` job stays alive; its state is visible only via the ps channel); every kill-family signal (TERM/KILL) targets background cpids whose exits are never a recorded step's exit; the watchdog's `kill -9` hits the `after` helper; and an OOM-137 on a recorded `docker exec` is a container-death abort under DG-10a, never a class) and `z_delta` [n,768] (canonical fs-delta text). **Emission ownership (one rule)**: the collection-mode tracker computes the delta text at generation and stores it in meta as an AUDIT copy (`meta.delta_text`, feeding DG-4a; verdict:cross-lens D-S4); the derived-root builders RECOMPUTE delta text, `exit_cls`, and `pre_obs_step` at build time from the raw jsonl via the same shell_state re-fold that powers render_canon (§5.5), ASSERT equality against the audit copy (F8 — recompute, don't trust), and encode the recomputed delta texts into `z_delta` — through the root's perception OBSERVATION render path (same convention as the SST predicted texts, e.g. e5's "passage: " prefix; one pinned convention shared with DELTA-PROBE's reference embeddings, round-6 M4); the CELL-DEFINING columns (sig via verbsig §13.3, mode — recomputable from exit+empty per §9.1, state_scope, mut_affected, ws_target, ws_observed) are ALL recompute-ASSERTED from the same re-fold; the SCORE-TIME recompute defines the cell, the cached column only accelerates it (round-7 S1) (they define fitness cells/slices — meta may cache them, never define them, F8; round-6 M2); the remaining §13.1 columns (payload_src, intended_outcome, exit, empty) are copied from the record verbatim. `exit_cls` mapping is TOTAL with SPECIFICS-FIRST precedence: {124→pilot-only class, mint abort; 125→abort; any value ≥128 (incl. bare 128)→build-time ABORT (the §8.2 no-producer argument would be falsified — fail closed, never classify); negative host-side returncodes (docker CLI signal-killed)→F1/abort family, never a class; then 0→ok; 1,2,126,127→their classes; any remaining nonzero→1 err catch-all}. The §16 reencode/mv_encode entries own this emission. Canonical delta format (sorted paths, ≤8 entries + `(+N more)`, size field for append checkability — verdict:targets M8):

```
delta: none
delta: removed /etc/foo.conf
delta: created /tmp/w/notes.txt(42B), moved /etc/a -> /etc/a.bak
delta: appended /tmp/w/task1.log(+18B)
```

`mv a b` where b existed is impossible by construction (destination-absence verification §4.2), so the `replaced` ambiguity (lens:targets open q4) does not arise; the guard failing is a collection abort.

**Enforcement — the harness strip seam (function-level contract)**: the harness owns `_strip_target_only(seqs)` — a shallow per-sequence copy with the `exit_cls`/`z_delta` keys REMOVED — applied before EVERY `stream.collate(...)` and `stream.flatten_predictions(...)` call (the only two places genome stream code receives seq dicts: harness.py's train loop and its scoring path). Batcher ruling: the batcher chunk receives the train-side fit only and returns batch indices; the fit passed to the batcher call is stripped too — so "structurally never sees the tensors" is literally true across every genome-code entry point. Aux targets attach inside `_train` from the harness-held ORIGINAL seq dicts, indexed by the same batch indices the batcher returned, via the immutable aux-target plumbing. The leakage assert moves INTO the immutable eval path: it perturbs exit_cls/z_delta in the harness-held originals at all positions and re-runs the stripped-copy pipeline — cmd_t predictions must be bit-unchanged — closing both the acausal-read exploit and the self-authored-guard hazard (verdict:targets B1). Oracle-annotation inputs (causal consumption of delta/exit as input tokens) are RULED OUT for v3.0 — pre-registered. The aux-target plumbing is DORMANT in v3.0 (no sanctioned consumer: multi-channel training targets are a v3.1 promotion decision, §8.5); it ships so the v3.1 flip is a config change, not a harness change (round-7 M5).

### 8.3 Probe instruments (report-only)

- **EXIT-PROBE**: linear probe on frozen cmd-position h; verbs enter iff pilot exit entropy ≥0.5 bits (threshold calibrated on v2 first — lens:targets open q6). Baselines: majority-per-verb, no-history z_cmd probe, tracker exit prediction, **and the position-aware probe (cmd + step-index + steps-since-last-mutation)** — DG-8b adopted (verdict:targets S2).
- **DELTA-PROBE**: h→768 retrieval vs {tracker delta-text, cmd-echo-assume-success, no-history}; ACTIVE only if state-dependent-effect arms (`>>` appends, rm of tracked dirs) ≥20% of mutation steps.
- Both: 3 seeds, balanced accuracy with absent-class treatment pinned in the prereg (verdict:targets M9). Both probes are implemented in `evolve/instruments.py` alongside ROLL@k (§8.4) — one immutable-eval-path module owns all report-only instruments.

### 8.4 ROLL@k (k ∈ {2,4})

Write-back rollouts at ~2000 content anchors; **indexing pinned: real history through obs_t; imagined obs at t+1..t+k−1; scored at obs_{t+k}** (ROLL@2 = 1 imagined step — verdict:targets M6); ASSERT prefix+horizon ≤32 steps. Information-mask rule: baselines see obs ≤ t and commands ≤ t+k, nothing else; arms = retrieve_by_cmd(cmd_{t+k}), within_traj(≤t), copy-last-real, tracker-rolled-symbolically (the strongest k-step arm). Companion columns: pred-vs-true sqL2 vs random-pair (the R10 off-manifold diagnostic). Cmd-corruption sensitivity is an aggregate batch check, not per-step (verdict:eval-split M4). Implemented once in `evolve/instruments.py` (immutable eval path); runs at full-budget scoring only.

### 8.5 v3.1 promotion rules (frozen in the v3.0 prereg; repaired per verdict:targets S3)

A channel may be PROPOSED as v3.1 fitness iff, on the v3.0 champion at 3/3 seeds: probe margin over the FULL baseline bracket (incl. tracker and position probe) ≥ **0.05 absolute** AND above the pre-measured probe noise band. A ~0 reading is interpreted via the bracket: tracker-saturated ⇒ "info present, channel covered" (no training-level proposal); genuinely low with weak baselines ⇒ the separately-embedded multi-channel training target earns the v3.1 proposal. ROLL@4: margin over the tracker arm ≥ noise band + 0.02. Mechanism note corrected: failed-rm renders ARE separable in fused space; exclusion is class-driven, not embedding-drowning (verdict:targets M1).

**Pre-registered fallback fork (verdict:targets S1 — a v3.0-BLOCKING path, not a v3.1 option)**: if DG-2's family exclusions drive the mutation-slice eligible mass below the DG-9 floor, the redesign path is fixed IN ADVANCE: the separately-embedded delta channel (the design-1.1 alternative rejected for v3.0 primary) becomes the v3.0-blocking redesign proposal — not a v3.1 afterthought. The mint stays blocked until either redesigned probes pass DG-2 with DG-9 satisfied, or the separately-embedded channel is ratified by dated amendment. Referenced from the DG-9 row (§14).

---

## 9. Verb classes & measurement

### 9.1 Unit and axes

Classification unit: **(signature, mode, state_scope)** cell — signature from the frozen `composed_verb()` labeler — ONE shared implementation in a NEW module `realenv/verbsig.py`, imported by the collector (writes meta.sig), the harness (re-derives from cmd text and ASSERTS equality), and `class_measure.py`; v1/v2 keep first-token verb_of bit-identically, mode from per-verb rules keyed on (exit, output-emptiness) **recoverable from the step record alone** (F8 discipline; any meta flag entering a mode rule gets a tracker recomputation cross-check — verdict:verb-classes M3), state_scope from mut_dep.

Four axes, all on RENDERED text (post-render_canon) in e5 space, frozen tool `benchmarks/class_measure.py`:
- **Axis-1** no-history predictability; θ1 = 0.59 (v2 midpoint rule; recalibrated numbers frozen by amendment).
- **Axis-2** cmd↔obs containment (own step), 0.656; all stages count as cmd text. Plus a **render-prefix containment column** so cwd-echo channels (pwd) are measurable (verdict:eval-split M5).
- **Axis-2′** history containment + **DG-6 derivability** — the two new columns (§6.5), measured per family AND per readback/probe slice.
- **Axis-3 (ONE rule; verdict:cross-lens D-B3, denominator per verdict:baselines B2)**: statistic = render-parity SST text exact-match **over ALL eval steps of the cell, ⊥ counted as mismatch**; ≥0.90 ⇒ class `sim`, excluded. Coverage (determined share) and accuracy-on-determined are separate ledger columns. Calibration anchors (achievable under this denominator): pwd ≈1.0 and echo ≈1.0 must exceed 0.90; v2 cat/ls must stay <0.90 (non-vacuous — the all-steps denominator makes the check real, closing the 0/0 vacuous pass); cd anchored on its ascend/reset sub-cell. The classes memo's separate state-echo class is dropped; its max(within_traj, SST) statistic survives inside the sim measurement as calibration (D-B3). Cells that remain content carry the tracker same-verb-foil top-1 as a standing credit-ledger column.
- **Axis-4** replay stability: the ceiling C from back-to-back twins (restricted to read-only, state-neutral steps — verdict:cross-lens D-S5, verdict:verb-classes S2); the **masking decision keys on cross-replay flip rates from the twin-mint**, not twin flips (verdict:verb-classes B1; §5.5). C ≥0.80 content-eligible / 0.50–0.80 reported / <0.50 excluded, post-masking, re-measured.

### 9.2 Six classes, ordered precedence (borderline ⇒ excluded side)

Rules applied in ORDER per cell; any statistic inside its pilot noise band goes to the excluded side (constitutional default). **Cell-unit extension (round-6 B1 ruling)**: for cells with `state_scope=created`, the classification unit splits on `ws_observed` — the harness rewrites observed-capture steps to the sub-cell pseudo-verb `"sig|mode|created-obs"` and blind-capture steps to `"sig|mode|created"` (the same step-level rewrite machinery as `"<verb>-miss"`; `ws_observed` is a §13.1 cache column, F8-recomputed under the §6.3 semantics: the created file's content transited a prior COMMAND's text OR a prior RENDERED OBSERVATION (the §10.1 ws.observed flag) — the command-text-only reading is wrong (an `ls -1 /etc > f` after an earlier listing of /etc is observed-capture: the SST determines it via edit-replay; round-7 S2)). This is what lets the worked example below assign semi-echo to the observed-capture sub-cell while the blind-capture sub-cell survives as content — without it, rule 1 at cell level would kill the whole created-scope readback surface and gut the DG-9 mass floor. classes.json carries rows for both sub-cells; native/mutated scopes are NOT split (bounded blowup).

1. axis-2 ≥ thresh OR axis-2′ ≥ thresh (slice-level) → **semi-echo** (excluded, reported)
2. success-mode near-dup ≥0.95 ∧ median OUTPUT-field ≤5 chars (field-scoped — verdict:verb-classes M6) → **ack** (excluded; mutation acks still drive downstream state)
3. axis-1 ≥ θ1 → **echo/const** (excluded)
4. axis-3 SST exact-match ≥0.90 → **sim** (excluded; the model's sim-cell accuracy is a standing bookkeeping-diagnostic column, never gated)
5. axis-4 C <0.80 post-masking → **noisy-excluded** (0.50–0.80 → reported-only)
6. else → **content** (enters fitness; carries the tracker credit-ledger column)

Predictions (hypotheses, never assertions): mutation own-steps ack/sim; pwd/echo/sleep sim; error modes semi-echo (report); masked-ps content iff DG-5+axis-4 clear; uptime excluded-or-thin; pipes/cond hits content; readbacks content minus the slice-level echo deaths. Conditionals predicted ONCE (ack removed from the double-booking — verdict:verb-classes M6). Worked example of the load-bearing case: `cat /tmp/w/app.conf` after `echo '<mined>' > /tmp/w/app.conf` — axis-1 low (payload varies), axis-2 low (path≠payload), axis-2′ HIGH at the ws_target slice (payload transited the echo command's text) ⇒ semi-echo for the observed-capture sub-cell; the blind-capture sub-cell (`ls -1 /usr/lib > f` → `cat f`) survives as content with SST ⊥ — exactly the intended honest surface.

### 9.3 Under-floor cells (resolves verdict:verb-classes S1 vs D-S12)

Below the 30-pair floor, the fixed cross-image probe battery (v2 rule, extended with the scripted mutation mini-battery) supplies the measurement — the battery GOVERNS. Terminal-verb inheritance is permitted only where the battery has confirmed terminal-dominance, and NEVER for families with argument-bearing producer stages (echo-source pipes are banned from the grammar anyway). Inheritance is recorded in the table, never silent.

### 9.4 The mutation slice is measured, not assumed (resolves verdict:verb-classes B2)

Because the cell unit includes state_scope, all axes ARE computed on mutated/created cells — a slice-confined trivial channel can no longer hide in pooled per-signature statistics. The false claim that retrieve_by_cmd only retrieves pre-mutation observations is corrected: cross-trajectory rbc CAN hit policy-templated created-file reads, which is exactly what DG-1 (image-grounded payloads) + DG-7 (cross-split near-dup gate) neutralize. Slice honesty rests on: DG-1, DG-7, the counterfactual foils (UD-4), delayed-copy/derivability columns, and the fidelity-gated tracker in the max — five measured instruments, zero assumptions. The gating floor is DG-9 (§14, the ONE mutation-content floor): mut-affected content ≥10% of content steps, composed ≥6% of content steps, per-image ≥ half these floors, with ≥30-pair coverage (G-COV). The policy additionally AIMS for ~15% mut-affected content at pilot — an explicitly NON-GATING pilot target (headroom above the DG-9 floor), never a gate.

### 9.5 The frozen table

`benchmarks/dockerfs3-classes.json` (machine-readable: rows = (sig, mode, state_scope), columns = class + all axis statistics + coverage), sha-pinned in the prereg; `bench_versions.resolve()` loads it and asserts summary.json matches verbatim (replacing v2's inline dict — verdict:completeness S4). The executable plumbing: `bench_versions.VERSIONS` gains a `"dockerfs3-v3.0"` entry whose class table maps each (sig, mode, state_scope) CELL to a fitness role {content, semi-echo, ack, echo/const, sim, noisy-excluded, report-only} — the `content=` tuple the harness consumes becomes the set of content CELLS, not verbs (incl. the §9.2 created-scope sub-cells `"sig|mode|created"` / `"sig|mode|created-obs"`, split on ws_observed — round-6 B1). Cell pseudo-verb strings are ATOMIC KEYS — never parsed by splitting (sigs themselves contain `|`, e.g. `pipe:ls|head`); every consumer treats them as opaque equality keys (round-7 M2). Cell pooling rides the existing verb-string machinery: the harness rewrites each step's verb string to the cell pseudo-verb `"sig|mode|scope"` (generalizing the v2 `"<verb>-miss"` rewrite), so `content_retrieval` and the same-verb foil pools pool by cell automatically with zero retrieval-code change FOR POOLING — the single v3 retrieval-code change is the `forced_foils` parameter (UD-4(b) injection, §8.1; version identity), and the two are independent (§13.3). Class freeze by dated amendment on post-fix pilot data, after render_canon and the tracker freeze, before mint (§16 order).

---

## 10. Honest baselines

### 10.1 One tracker (verdict:cross-lens D-B5; pre-mortem DG-4b)

**`realenv/shell_state.py`** — the SINGLE shell-state authority. Two declared visibility modes:
- **Collection mode** (full visibility): used by gen_sequence_v3 for pool filtering, meta.expected, state_scope, mut_affected, fs_clock, fs_delta. The FsLedger journal schema (path → state, fs_clock, provenance, tombstones) is the adopted core; the composition ws dict and the 5-state job table are sub-structures.
- **Eval mode (SST)** (render-parity): consumes ONLY OBS_CAP renders of the history + command strings + frozen static knowledge; total parser over the frozen BNF; closed rules; emits (pred_step_dict, determined).

Unified state schema (normative):

```python
Overlay = {
  "cwd": str,                       # "/" at t=0 (protocol constant, documented)
  "fs": { path: VNode },            # VNode = {kind: dir|file|symlink, content: known(bytes)|partial|unknown,
                                    #          deleted: bool (tombstone/whiteout), link_target: str|None,
                                    #          fs_clock: int, provenance: "image"|"mut:<op>@<step>"|"redirect@<step>",
                                    #          payload_sha256: str|None, trailing_nl_known: bool}
  "ws": { wsf: {producer, observed: bool, appends: int, content_lines: int} },   # composition sub-structure
  "jobs": { j: {cpid, K, effect, state: waiting|stopped|stopped_pending_term|fired|killed,
                launch_vt, deferrals: int} },                                     # 5-state machine (§5.3)
  "fs_clock": int,
  "touched": set,                   # propagation law per §10.2
}
```

Collection mode is a fold over raw step records; SST mode is a fold over (render_obs strings, cmd strings) only — the two share transition code, differ only in the evidence extractor. A second independent implementation exists only as the pilot cross-check; disagreement gate = 0. The tracker is model-free eval code: sha-pinned, frozen pre-mint, never a genome chunk (constitutional ruling per pre-mortem open q5).

### 10.2 SST scope (rules amended per verdict:baselines S2)

R1 pwd; R2 cd (entailed targets only; cwd at t=0 is the protocol constant "/" — documented hand-coded knowledge, verdict:baselines M5); R3 mutation own-steps (entailed existence; errors from the probe-harvested per-image template table §3.5 — no hand-authored dialect enum); R4 echo (BNF bans backslashes ⇒ literal join is safe); R5 reads of known content — **appends are determined only for trajectory-created files whose full byte content including trailing newline is known from the writing commands; appends onto merely-observed files ⇒ ⊥** (docker_env rstrips trailing newlines — verdict:baselines S2b); R6 ls edit-replay — **requires an uncapped prior render** (capped listings have unknowable totals — verdict:baselines S2a); one-per-line classes fully determined; `-l` splices ledger-only; R7 find replay — uncapped-prior precondition, dropped if pilot shows <2% replays; R8 errors on known-dead paths via the per-image template table; R9 process: the shared 5-state job simulator; ps predictions in canonical form where structurally determined. **SST-G3**: full evaluation of the composed grammar over belief state (mandatory — verdict:composition B2, pre-mortem DG-6).

Touched-set propagation law (verdict:eval-split M3): a mutation touches its target path, the target's parent (listing changes), all descendant paths for dir moves, and the link node for ln; symlink-mediated reads touch link + target. Written into the module docstring and unit-tested.

### 10.3 Encoding & arms (repaired per verdict:baselines B1)

SST/wtm predictions are rendered and encoded **via the ROOT'S perception module** — resolved by a NEW `reencode.load_perception_for_root(root)`: it reads the root's perception stamp `{perception: {impl, model, content_sha}}` from summary.json (written by reencode/mv_encode at build time, §13.1) and calls the existing `load_perception(impl)` (which takes an impl NAME, not a root), FAILING CLOSED if the stamp is absent on a v3 root — including the e5 "passage: " prefix — with the root's recipe, standardized with train stats — never via realenv render_obs. **Render-parity gate**: on determined steps whose predicted text exact-matches the collected text, the SST embedding must match the cached true embedding at cos ≥ 0.9999 (near-zero, not zero — batch-composition float noise; verdict:baselines M3). Tie-rule honesty corrected: `_rank_stats` counts only strictly-closer foils, so an exact prediction KEEPS top-1 against identical foils; near-constant cells are removed because every exact-text arm and trivial memorization bank 1.0 there — degeneracy, not tie-refusal (verdict:baselines S1, verdict:targets M4).

**Seven arms, frozen in the prereg (verdict:cross-lens D-S9)**: retrieve_by_cmd, no_history, copy_prev, within_traj (exact ratified v2 definition, continuity), **within_traj_mut** (retrieved stale render patched with the overlay delta, re-encoded; patcher reuses the one tracker; mut-vs-plain is a reported diagnostic, not an assert — verdict:baselines M2), **sst** (⊥→zeros), **sst_composite** (SST-where-determined else within_traj_mut — expected to dominate; the aggregate-max fix). All as constitution §5 aggregate columns + solo ledger columns per verb.

### 10.4 Fidelity & floors (verdict:cross-lens D-B5; verdict:baselines S5; verdict:eval-split S1)

- **DG-4a SST-fidelity**: rendered-TEXT exact-match ≥0.995 per (sig, mode, state_scope) determined cell on pilot; any cell below ⇒ fix or move out of the determined set BEFORE class freeze. (The existence/exit-only audit was the wrong channel.)
- **⊥-share floor**: ≥40% SST-undetermined eval steps per content verb, plus ≥15% determined per content verb (the G-⊥ scope — ONE denominator here and in the §14 row) — controller-steered realized rates with a **mint-scale assert** (pilot-only floors can drift; verdict:baselines S5). Joint satisfiability (⊥-floor × determined-floor × ≥500 steps/verb) arithmetic-checked at pilot (verdict:baselines M6).
- **Information-parity disclosure**: the SST reads OBS_CAP=1600-char renders; the encoder's 256-token window covers ~1000 chars — the asymmetry is conservative (strengthens the baseline) and is DISCLOSED in the prereg rather than claimed away (verdict:baselines S6).
- Freeze-then-audit: rules + template tables authored from train-pilot renders + docs only, content-hashed, frozen pre-mint; post-freeze edits = dated amendment + re-baseline. Calibration-on-v2 (not v1) is a dated deviation note (verdict:baselines M1).

### 10.4b Precompute interface

`evolve/precompute_baselines.py` (deterministic, seed-free, cached once per root): reads `<root>/val.jsonl` — the FULL val split, all four held-out images (inner AND final; chosen explicitly so final-test champion scoring has the new arms — an inner-only precompute would break it); folds the SST per trajectory; renders predicted step-dicts — with `render_canon.canon` applied to each predicted step-dict BEFORE the perception render, exactly as on real steps (render parity; G-EMB would catch the omission but the spec states it) — via the root's perception module (resolved by `load_perception_for_root`, §10.3); encodes; writes `sst-val.pt` [steps,768] + determined-mask, `wtm-val.pt` [steps,768] (within_traj_mut patched texts re-encoded), aligned to `_data_tensors` step order — the full-val tensors are sliced by the image-contiguous `val.jsonl` sequence order that `split_val` filters, so applying the same image filter to the precomputed rows reproduces `_data_tensors` step order exactly; sha256s → summary.json + prereg. `harness._base_for` loads them and adds the three new arms to the aggregate table; its signature extends to `_base_for(split, seed, steps, fit, evaldata, device, data_root, stats=(mo, so))` — mo/so are score_genome locals today, so the train-stats tuple must be passed through (a listed §16 signature change); `base_cache.json` keying gains the arm set + classes_sha + the root artifact sha (§13.2). Cost: full val ≈ 4×900×28 ≈ 100K steps ⇒ ~200K predicted texts (sst + wtm) ≈ 2h — budgeted as its own §3.7 row, not "negligible".

### 10.5 Margin definition

fitness = pooled content top-1 margin (§8.1). The honest definition of world-model surplus in a mutable world: reads of never-observed content on unseen systems, post-mutation reads of never-listed dirs, blind-capture readbacks, composition with ⊥ stages — cross-system priors + inference under partial observability, never replayable bookkeeping (lens:honest-baselines §5). Process-domain disclosure (verdict:time S4): on the time/process arm the headline margin is honestly expected ≈0 — the cells the 5-state automaton fully determines (ps reads, kill outcomes, job-log readbacks with known payloads) classify `sim` and are excluded, so the process arm's v3.0 payoff is diagnostic (DG-5 entropy, probe columns, the class table), not headline margin; process-cell margin can come only from cells the SST cannot determine.

---

## 11. Eval & split integrity

### 11.1 Split policy

Held-out IMAGE, identical 8/2/2 (fedora+mariadb inner / rockylinux+httpd final), digests re-pinned into `benchmarks/dockerfs3-digests.json`. Mutation patterns are NOT structurally held out (held-out-tool prior; the constitution's one-policy/paired-collection rule); pattern generalization is measured via slices (lens:eval-split D1). New train-only images (UD-5) never touch the val/final sets.

### 11.2 Leakage battery (L1–L7, all pre-registered)

- L1 exit channel: structural (the eval-split D3 exclusion rule, §2 notation) + miss-render near-dup audit.
- L2 delta channel: targets namespace the history renderer never reads.
- L3 generalized causality sweep: corrupt obs_j ⇒ all channels' predictions at cmd t ≤ j bit-unchanged; **includes exit_cls/z_delta perturbations at every position** (verdict:targets B1); horizon rules for ROLL; cmd-corruption sensitivity as an aggregate check (verdict:eval-split M4).
- L4 render-inert meta: pinned unit test — the render is a pure function of {cmd, output, exit, cwd} + whitelist.
- L5 replay determinism: re-collect 20 sampled sequences per split (cheap under per-seq seeding §3.6); **fail-closed byte-equality on jsonl-as-stored: 100% or every differing field enumerated into the mask list and the diff re-run clean** (DG-3 adopted over the 99.5% tolerance — verdict:cross-lens M10, M14).
- L6 cross-sequence pollution: fresh-container replay reproduces sequence k exactly (structural under §3.1 + §3.6).
- L7 = **DG-2 foil/encoder resolution** (the missing gate — verdict:eval-split B2): per probe family, median d(pre-render, post-render embedding) > 3× the twin-replay noise floor AND ≥90% of probe steps above the floor; failing families are redesigned (probe-the-entry, small dirs) or excluded — decided at pilot, frozen by amendment.

Plus: DG-7 cross-split contamination (val mutation-slice obs vs ALL train renders, cos>0.995 or exact ≤ the v2-measured reference constant; companion: cross-image vs within-image rbc top-1 on the slice — indistinguishable ⇒ slice image-independent ⇒ excluded, prereg rule); DG-1/DG-6 columns; the nondeterminism scanner (fail-closed, planted-canary self-tested) covering epoch/date/nonce/pid patterns + `/tmp/.tj` + `after` names.

### 11.3 Foils

Same-verb backbone; mode/const exclusions before pooling; counterfactual pre-mutation foil injection per UD-4(b) (recommended, pending sign-off; m=8 slots, meta.pre_obs_step, only where pre/post renders differ, DG-2-gated); dup-after-render audit per (sig, mode, state_scope) slice; foil-pool composition slice for the v2-continuity report (workspace steps sliced out — verdict:world-dynamics M2).

### 11.4 Re-baseline ladder

6 rungs (mse, InfoNCE, hippo, fastweights, chunked, r13-co-syscond-cuedrecall), 3 seeds, full budget, one environment; plus the **subsampled-train champion arm** (champion trained on the full root subsampled to 450 seqs/image, 3 seeds — the §11.5 ablate comparison arm; the "+3" completing §3.7's 21 trainings). The quantified constitution §6 gate enforced on the **native-continuity slice** — computed under BOTH v3 and v2 mode rules (the generalized miss rule changes the slice; the v2-mode-rule column preserves comparability — verdict:eval-split M1), with the dated amendment per UD-6. New-slice ladder order reported. Bench-discrimination checks: (a) tracker > within_traj on the mutation slice under counterfactual foils; (b) incumbent-vs-tracker mutation margin recorded as the declared headroom line. Positional audit of every rung (hippo clamps silently) is a checklist item even at 28±4 (verdict:completeness S7). sanity.py's gen-twin/history-ablation arms re-validated against v3 step shapes (no exit:null partials exist on exec — moot hazard; verdict:completeness M8).

### 11.5 Paired ablate arm (verdict:eval-split D9; plumbing per verdict:completeness S2)

Same mint run, same digests/seeds: **dockerfs3-ablate** = identical policy with mutation/time/composition arms off, weights renormalized, 450 seqs/image, TRAIN-ONLY. Subsampling is TRAIN-side: the comparison arm is the champion trained on the FULL root subsampled (seeded, recorded in the ledger) to 450 seqs/image — matched to the ablate root's train size (the §3.7/§11.4 **subsampled-train champion arm**, 3 seeds). Causal claims ("what did the new phenomena buy") = model(subsampled-full-train) − model(ablate-train), BOTH arms scored UNCHANGED on the ONE frozen v3 inner val, standardized with the canonical full-root train stats for both (verdict:completeness M5). The causal contrast is explicit: same train quantity, phenomena present vs ablated. **Plumbing**: `score_genome` gains `--val-data` (train root ≠ val root) AND `--stats-root` — `standardize_stats` currently derives from the `data` root inside score_genome (harness.py), so an ablate run would otherwise standardize with ablate stats; the ablate arm passes the canonical full root so both arms standardize identically. The train-side subsample is implemented in the harness train assembly (seeded, recorded in the ledger; owner: the §16 harness diff). `bench_versions.resolve()` recognizes the ablate root by an explicit `"ablate": true` summary flag (never val.jsonl sniffing); scan_publish inventories it; the root lives at `data/dockerfs3-ablate` and is published (UD-8).

---

## 12. IMAGES (new section; resolves verdict:completeness B1)

### 12.1 The set

**Core = the 12 v2 images, re-pinned fresh** (the :latest tags have drifted since dockerfs2-digests.json; the drift's effect on the continuity slice is measured and reported at P0 as a continuity note):

| split | images |
|---|---|
| train (8) | alpine:latest, ubuntu:latest, debian:stable-slim, python:3.12-slim, redis:7-alpine, nginx:stable-alpine, postgres:16-alpine, node:22-slim |
| inner-val (2) | fedora:latest, mariadb:latest |
| final-test (2) | rockylinux:9, httpd:2.4 |

Digests → `benchmarks/dockerfs3-digests.json`; collection runs by digest ref with the v2 `--pin-digests`/`--expect-digests` entry gate.

### 12.2 Capability audit (P0, a pilot gate)

Per image, before any policy pilot: `command -v` for {pwd, uptime, sleep, ps, echo, kill, rm, mv, ln, readlink, mkdir, touch} (most are builtins/busybox; **ps is the decision variable**); ps tier T2/T1/T0/ABSENT; `/usr/local/bin` writability (the `after` gate); `/bin/sh` identity; error-template harvest; fifo support; fractional-sleep support (`sleep 0.1` — the fire/barrier scripts depend on it; busybox FANCY_SLEEP / coreutils expected on all 12, but probed, never assumed). Expected ps landscape (corrected facts: ubuntu:latest SHIPS procps — verdict:infra M2; modern busybox supports `-o stat` — verdict:time S5): train coverage = ubuntu (procps T2) + 4 alpine-family (busybox T2/T1); absent on debian-slim, python-slim, node-slim; val fedora likely procps, mariadb ubuntu-based procps; final rocky procps, httpd debian-slim-based likely ABSENT. Absent ⇒ v2 skip-and-redistribute, recorded.

### 12.3 New candidates (TRAIN ONLY; contingent per UD-5)

| candidate | rationale | risk |
|---|---|---|
| debian:stable (full) | procps + rich /usr tree; near-zero distribution shift from the slim sibling | low |
| opensuse/leap:15.6 | new family (zypper world), procps, distinct /etc layout | moderate (new error templates) |
| archlinux:base | rolling glibc, procps, distinct filesystem conventions | moderate |

RHEL-family candidates (alma/oracle/amazon) are EXCLUDED — fedora/rocky are the held-out family; seeding train with their relatives would weaken the transfer claim. New images pull digest-pinned, enter TRAIN only, and are declared in the version tuple. If adopted, seqs/image may drop to keep the step budget (e.g. 14 img × 780 ≈ 306K).

---

## 13. META-THREADING & CACHE VERSIONING (new section; resolves verdict:completeness B2, verdict:verb-classes M5)

### 13.1 What the encoded cache carries (v3 format)

The v3 cache format, emitted per sequence by the derived-root builders `reencode.py`/`mv_encode.py`:

```
{ z_obs [n,768], z_cmd [n,768], cmds [n], image,
  # v3 meta columns (exactly what the eval consumes; nothing render-visible):
  sig [n] str, mode [n] str, state_scope [n] str, mut_affected [n] bool,
  ws_target [n] bool, ws_observed [n] bool, payload_src [n] str|None, intended_outcome [n] str|None,
  pre_obs_step [n] int|-1, exit [n] int, empty [n] bool,
  exit_cls [n] long, z_delta [n,768],
  # format guard (root-level, NOT inside the .pt payload):
  } + root-level cache_meta.json: { cache_format: 3, bench_version, policy_sha, classes_sha }
      # written beside the emb-seq-*.pt caches by reencode.py/mv_encode.py; the harness-owned wrapper reads
      # it fail-closed BEFORE loading any .pt (the torch payload shape — a plain list of seq
      # dicts — is unchanged, so v1/v2 cache loading is untouched). The perception stamp
      # {perception: {impl, model, content_sha}} is written into the derived root's
      # summary.json at build time; read by load_perception_for_root(root) (§10.3),
      # which FAILS CLOSED if absent on a v3 root.
```

**Raw-root closure (ruling)**: v3 roots are scorable ONLY as derived roots built by `reencode.py`/`mv_encode.py` — render_canon interposed, perception-stamped. `bench_versions.resolve()` and the cached_encode wrapper RAISE on a v3-policy root lacking the perception stamp. Raw-root scoring (`seq_worldmodel.encode_split` directly over a raw jsonl root) stays a v1/v2-only path; `encode_split` gains no v3 branch (render_canon lives only in the reencode path).

**The v2 ok-bit is replaced**: `encode_split`'s hard-coded `exit==0 ∧ non-empty` bit cannot express v3's per-(verb,mode) rules (mutation successes are empty-exit-0). v3 caches store raw `exit` + `empty`; the harness computes mode masks from the frozen spec at score time — recoverable-from-record, F8-honest. v1/v2 caches keep the ok bit untouched (bit-identical regression, §16).

### 13.2 Fail-closed staleness (the Amendment-4 precedent, made structural)

The harness-owned wrapper around `cached_encode` (the §16 layering ruling — realenv stays evolve-free): a v3 cache is loadable ONLY if the root-level `cache_meta.json` beside it (§13.1) carries {cache_format, bench_version, policy_sha, classes_sha} matching what `bench_versions.resolve(root)` expects — any mismatch or absence RAISES (never silently loads), including a v3-policy root lacking the §13.1 perception stamp. collect() unlinks stale `summary.json`/`emb-seq-*.pt` at entry (Amendment-6 discipline — this is what the code does today; it does NOT and cannot clean `base_cache.json`, which lives in `evolve/archive/` keyed `data_root|split|seed|steps`). base_cache staleness is closed by KEYING, not unlinking: the v3 key extends with the arm set + classes_sha + the root's ARTIFACT SHA (sha256 over summary.json + the emb-seq caches) + a TRAIN-SET DESCRIPTOR (`full` or `sub<seqs-per-image>:<subsample-seed>`) + (when `--val-data`/`--stats-root` are set) THEIR roots' artifact shas — the frozen-val embeddings and standardization stats are inputs to every cached baseline number, so a re-encode of the full root that leaves the ablate root untouched must miss the cache (round-6 S3 — the baselines are fit-dependent, so the §11.5 subsampled-train champion arm must never share a cache row with the full-root scores on the same root/split/seed/steps), so a re-mint into the same path can never be served stale baseline maxes; this keying subsumes and replaces the current `|v2` vtag (harness.py:68, keyed on `within_traj_in_max`) — the explicit arm set generalizes that single bit; owner: the §16 harness diff. reencode.py copies summary.json (already in code, Amendment 4) AND stamps the format block + the perception stamp; `mv_encode.py` (the third encode path, live for the stream chunk axis) gets the SAME stamping — without it every multi-vector root would either raise at the fail-closed gate or bypass it entirely. **Stamp-target ruling**: the mv root PROPAGATES the src root's perception stamp UNCHANGED — its eval space (z_obs/z_cmd) is copied verbatim from the single-vector src root (mv_encode.py:81), so `load_perception_for_root` must resolve the src's dual-surface impl, never mv's own `--perception` recipe (which may expose only `render_obs_multi`/`pool` and would break the SST/wtm precompute); the mv recipe is recorded separately as `mv_recipe` in the mv root's summary. A v3 root scored against a v2-era or partial cache is impossible by construction; the planted-violation test ships a deliberately stale cache and asserts the raise (DG-10d).

### 13.3 Harness consumption

`_data_tensors` reads the meta columns; re-derives sig from cmd text via the frozen labeler (`realenv/verbsig.py`) and ASSERTS equality (F8); rewrites each step's verb string to its cell pseudo-verb `"sig|mode|scope"` per the v3 VERSIONS cell table (generalizing the v2 `"<verb>-miss"` rewrite) so `content_retrieval(content=...)` and `_foils_sameverb` pool by cell with no pooling-code change (§9.5); builds mode masks, slices (state_scope, mut_affected, ws_target), the counterfactual `forced_foils` index tensor (from pre_obs_step, consumed by `retrieval()`'s §8.1 injection parameter), the determined-mask alignment for SST/wtm tensors (from `evolve/precompute_baselines.py`: `sst-val.pt`, `wtm-val.pt` + determined mask, sha-recorded; the full-val tensors are sliced by the image-contiguous `val.jsonl` sequence order that `split_val` filters, matching `_data_tensors` step order — §10.4b). exit_cls/z_delta reach training ONLY through the §8.2 strip seam: `_strip_target_only(seqs)` before every stream.collate/flatten_predictions call, aux targets attached in `_train` from harness-held originals by batch index. `base_cache.json` keying extends with the arm set + classes_sha + the root artifact sha (§13.2).

---

## 14. Pre-registered gates (consolidated)

All fail-closed. The `when` column is binding: most gates measure on pilot data before class freeze; DG-9 measures at P5 (post-freeze, pre-mint) by design; MINT rows assert at mint; G-LAD at re-baseline; G-SEP is measured at P3 (both arms training-free) and enforced at re-baseline; G-BUDGET is rolling. Each gate is ratified only after a planted-violation self-test (DG-10d, a table row below — no gate counts until it has rejected a planted failure). Owners are lenses-as-roles in the runbook. **29 gates total.**

| gate | what | threshold | when | owner |
|---|---|---|---|---|
| DG-1 | payload provenance: delayed-copy predictor on revisit reads; cross-seq payload dup | top-1 < pilot-measured per-family threshold, frozen by amendment (a top-1 accuracy bound — not the axis-2′ containment number); dup <5%; lexicon arm ≤5% w/ payload_src | pilot P3 | world-dynamics |
| DG-2 (=L7) | foil/encoder resolution on mutation probes | median d_pair >3× noise floor; ≥90% steps above floor, per family | pilot P3 | eval-integrity |
| DG-3a | twin-mint byte-diff (2× pilot, same seeds) | 100% or enumerated-mask + clean re-run | pilot P1 | infra |
| DG-3b (=G-J) | jitter invariance (injected 0–300ms delays) | byte-identical jsonl | pilot P1 | time |
| DG-3c | nondeterminism scanner (renders: dates/nonces/pids/sentinels/helper names) | zero hits; planted-canary self-test | pilot+MINT | infra |
| DG-3d | -l-family field policy on mutated paths | per §5.5 table; twin-mint clean after masks | pilot P1 | time |
| DG-4a | SST fidelity (rendered text) | exact-match ≥0.995 per determined cell | pilot P3, pre-freeze | baselines |
| DG-4b | one-tracker rule + independent cross-check | disagreement = 0 | pilot P1 | baselines |
| DG-4c | visibility parity (mining ≤ render cap) | unit test green | always | sequencing |
| DG-5 | process-arm entropy: ps distinct renders; single-render mass; job-state mixture | ≥25/image; ≤25%; {run,stop,term,mixed} ≥15% each | pilot P2/P3 | time |
| DG-6 | composed derivability + SST-G3 | SST-G3 exact-match ≥0.95 on derivable steps, else family sim | pilot P3 | composition |
| DG-7 | cross-split contamination (val mutation slice vs train) | near-dup rate ≤ v2 reference constant (measure FIRST) | pilot P3 + MINT | eval-integrity |
| DG-8 | probe-outcome mix + axis-1+ schedule leak | {70/20/10}±5%; axis-1+ top-1 < 0.5× within_traj on probe slice | pilot P2/P3 | sequencing |
| DG-8b | position-probe: the position-aware predictor (cmd + step-index + steps-since-last-mutation) as watermark pricer — prices the canonical-PID / step-index watermark (§2 OPEN) and joins the §8.3 probe baseline bracket | axis-1+ top-1 < 0.5× within_traj on the probe + process slices; the cpid scheme freezes only after its watermark price is measured | pilot P3 | sequencing |
| DG-9 | **aggregate eligible mass** (post-freeze, pre-mint; breach ⇒ the §8.5 pre-registered fallback fork) | mut-affected content ≥10% of content steps; composed ≥6% of content steps; per-image ≥ half these floors | pilot P5 | **eval-integrity (named owner — verdict:cross-lens M13)** |
| DG-10a | trajectory-death fail-fast (container dies mid-seq ⇒ abort image) | zero survivors; planted-fault test | MINT | infra |
| DG-10b | zero-timeout / zero-125 asserts at collect() entry | 0 events | MINT | infra |
| DG-10c | probe-cache key (digest, policy-hash, collector version) + cold/cached twin | dict-equal | pilot P1 | infra |
| DG-10d | gate-ratification meta-gate: every gate in this table must reject a PLANTED violation before it counts as ratified | one recorded planted-failure rejection per gate, pre-GO | pilot (rolling, before review C) | eval-integrity |
| G-EM | expected-vs-realized mismatch | ≤ pilot-measured base rate (target ≤1%) | pilot P1 | world-dynamics |
| G-RATE | all realized policy rates (linkage, revisit, probe mix, weights) | ±5% of targets | pilot P2 + MINT | sequencing |
| G-⊥ | ⊥-share ≥40% / determined ≥15% per content verb | mint-scale assert | pilot + MINT | baselines |
| G-EMB | SST render-parity embedding gate | cos ≥0.9999 on exact-text determined steps | pilot P3 | baselines |
| G-CF | counterfactual-pair floor | ≥30 differing pre/post pairs per image (ls-class) AND the §8.1 realized injected-slot-rate floor on the mutation slice (pilot-measured) | pilot P2 | eval-integrity |
| G-COV | ≥30 cross-image pairs per cell (else battery); ≥500 steps/content verb/split | as stated | pilot P3 + MINT | classes |
| G-DIG | digest entry gate (--pin/--expect) | exact match | MINT | infra |
| G-LAD | ladder gate: mse < InfoNCE < evolved rungs on native-continuity slice (v2-mode-rule column) | strict, no non-adjacent inversions | re-baseline | eval-integrity |
| G-SEP | tracker/within_traj separation under counterfactual foils on mutation slice | tracker >> within_traj (pre-registered δ) | measured pilot P3 (both arms training-free); enforced re-baseline | eval-integrity |
| G-BUDGET | mint ≤3h; pilot iterations ≤6 / ≤3 weeks (UD-7) | as stated | rolling | runbook owner |

Constitutional amendments required (dated, in the prereg): axis-3 sim class + the (sig, mode, state_scope) unit (constitution §4 extension); seven-arm max (constitution §5, version boundary); native-slice scoping of the constitution §6 ladder gate (UD-6); constitution §6 instrument arithmetic unchanged (28±4 fits); calibration-on-v2 note.

---

## 15. Scope: v3.0 vs v3.1 vs v4

**v3.0 (ships now)**: everything in §3–§13. Exec substrate; mutation + revisit world; kill-signal automaton + ps + after-jobs; G3 composition; 7-arm baseline max; fused-render pooled fitness + counterfactual foils per UD-4(b) (recommended, pending sign-off); exit/delta data channels (target-only); EXIT/DELTA probes + ROLL@{2,4} as instruments; ablate arm; full gate battery.

**v3.1 (pre-registered forks, no new collection)**: promotion of exit or ROLL channels into fitness per §8.5; counterfactual top-1 as a fitness component (if UD-4(a) is chosen instead, promotion is re-proposed here); per-stage pipeline exits as meta; miss-prediction report slice as a candidate channel; depth-2 probe battery results informing a grammar extension proposal.

**v4 (separate charter)**: the PTY spike — jobs/fg/bg/%n, literal ctrl-keys, `wait` — inheriting the infra memo PLUS its verdict's mandatory fixes (bg-job closure semantics: every job closed by a blocking wait-class step or outliving the trajectory; Tier-1 resends the bookkeeping line; sentinel regex carries the nonce + trailing newline; shadow fencing). Trigger condition (lens:time open q6): the v3 process-dynamics slice shows the model saturating the signal-delivered automaton. Also v4-or-later: `<<<`, variable expansion/globbing in payloads (BNF freeze forces a boundary), multi-channel training targets if the v3.1 probe fork selects them.

**Retired framings stay retired**: no held-out-tool, no synthetic-ontology substrate, no training-signal-auxiliary re-propose (R10/R11 recorded priors; retry only under the §8.5 evidence fork).

---

## 16. Pilot protocol & mint runbook deltas

**Freeze order (resolves verdict:completeness S6)**: (1) substrate decision = exec (done, D-B1) → (2) P0 images/capability audit + DG-7 reference-constant measurement on dockerfs2 → (3) tracker module + render_canon implemented, unit-tested → (4) P1 mechanism pilot (determinism: DG-3a/b/c/d, DG-4b (the independent cross-check, disagreement = 0), DG-10c, G-EM) → (5) P2 rate tuning (G-RATE, G-CF, DG-5, DG-8 mix) → (6) P3 class + honesty measurement (all four axes + axis-2′/DG-6 columns, DG-1, DG-2, DG-4a, DG-5, DG-7, DG-8b, G-EMB, G-COV; G-SEP measured here — both arms training-free — enforcement stays at re-baseline) → (7) SST freeze-then-audit; render_canon freeze; class table freeze BY AMENDMENT → (8) P4 post-fix verify (re-verify every frozen number) → (9) P5 = DG-9 aggregate mass check → (10) reviews A/B/C (adversarial, convergence-ruled) → (11) prereg GO → (12) the one mint (full + ablate arms, digest-gated, gates as code asserts) → (13) encode + publish (scan_publish extended for .pt inventory/exemptions) → (14) re-baseline (§11.4) + plangoals-v3 harvest → (15) README/insights record.

Pilot sizes: P0 = probe-only (no trajectories); P1/P2 = 24 seqs/image × 12 (~8K steps, ~3 min at the §3.7 per-step model); P1's twin arms double it; P3 = 60 seqs/image × 12 (~20K steps — class measurement needs the coverage floors); P4 = P3-sized rerun; P5 = analysis over P3/P4 data (no new collection). Shadow-audit collections (§3.4), where used, are separate arms outside P1/P3.

**Runbook deltas vs v2** (`benchmarks/dockerfs3-runbook.md`, owner: the prereg assembler): two-arm mint invocation; per-seq seeding; fresh-container lifecycle + orphan sweep (`docker ps -aq --filter label=tj3-mint=<seed> | xargs docker rm -f`); probe-container step; capability-audit step; timing side-channel handling; mint-host environment + uv.lock freeze recorded in summary (verdict:completeness M6); budget table §3.7 with measured 2-image pilot numbers; the UD-7 abort rule.

**Review ladder content** (A/B/C, convergence-ruled as in v2):
- **Review A** (design/prereg): the amendment set (§14 tail), the axis definitions, the UD decisions as ratified, the BNF ≡ G3 identity, the gate thresholds vs their planted-violation tests.
- **Review B** (pilot evidence): SST fidelity table both directions (under- AND over-prediction — verdict:eval-split risk 7); tracker-coverage per verb; blind-capture replay audit (SST actually fails on ws_observed=False — lens:composition risk 5); mismatch-counter triage log; DG-2 naturalism cost review (small-dir probing as a policy fingerprint — pre-mortem risk 2); realized-rate tables; mask-list size review (>10 masked fields is itself a finding to surface — pre-mortem risk 4); per-image skip/availability table.
- **Review C** (pre-mint GO): DG-9 mass figures; the frozen class table; the headroom ledger (incumbent-vs-tracker per slice); plangoals-v3 harvest plan; the budget table against measured pilot ms/step; the UD-7 clock.

**v2→v3 harness/CLI diff list** (implementation checklist): `bench_versions.py` v3 entry (the (sig, mode, state_scope) cell table + fitness-role map, classes-file loader) + ablate flag; `evolve/archive.py` — `_bench_of` recognizes dockerfs3 data roots (bench assignment); `ACTIVE_BENCH` flips to `"v3"` only at the re-baseline commit; `evolve/cli.py` — `--bench` choices gain `"v3"`; argparse wiring for `--val-data` and `--stats-root`; `harness.py` — `_data_tensors` meta columns + slices + counterfactual foil rows + the cell pseudo-verb rewrite (`"sig|mode|scope"`, generalizing `"<verb>-miss"`) + 3 new arms + the `_strip_target_only` seam (§8.2) + eval-path leakage assert incl. new tensors + `_base_for(..., stats=(mo, so))` signature change + base_cache artifact-sha keying; `score_genome(--val-data, --stats-root, --subsample-seqs, --subsample-seed)` — the last two wire the train-side seeded subsampling (full root → 450 seqs/image) for the ablate comparison arm and feed the §13.2 train-set descriptor; `realenv/verbsig.py` (NEW — the ONE `composed_verb()`/sig labeler, imported by collector, harness, and class_measure.py); `realenv/seq_worldmodel.py` — the ONE v3 retrieval-code change: `retrieval(forced_foils=...)` + `content_retrieval`'s subset-seam translation (§8.1; version identity under the Darwin-Gödel guard; regressed bit-identical for v1/v2 by the archived-margin test) — `encode_split`/`_rank_stats` and everything else in the module UNCHANGED; `cached_encode` fail-closed (raises on a v3-policy root lacking the format block/perception stamp; raw-root scoring via `seq_worldmodel.encode_split` stays a v1/v2-only path — `encode_split` itself is UNCHANGED, §13.1 ruling; **layering: the gate lands in a harness-owned wrapper around `cached_encode` — it consults `bench_versions.resolve` and `cache_meta.json`, both evolve-side concepts, so `realenv/seq_worldmodel.py` stays free of evolve imports**); `reencode.py` render_canon interposition (re-folds shell_state over the raw jsonl per sequence, §5.5) + §13.1 meta-column emission from that same re-fold (`exit_cls` computation, delta-text recompute + F8 equality assert against `meta.delta_text`, `pre_obs_step`) + delta-text encode into `z_delta` + `cache_meta.json`/summary stamping + the perception stamp `{perception: {impl, model, content_sha}}` + the `load_perception_for_root(root)` resolver (fail-closed on stamp-less v3 roots); `mv_encode.py` — the SAME emission + stamping + fail-closed path (the third encode path), with render_canon interposed on the multi-vector SEGMENT renders too (render_obs_multi consumes canon'd steps — the §5.5 law applies to every perception surface, round-5 M4); reencode.py/mv_encode.py — TRAIN-ONLY-root tolerance (the ablate raw root has no val.jsonl by F6; the split loop skips absent splits instead of raising — required to build dockerfs3-ablate-e5, round-5 M1); `evolve/precompute_baselines.py` — full-val precompute resolving perception via `load_perception_for_root`; `docker_env.py` — the prologue-injection seam (`DockerBox.run` signature extends to accept a collector-composed prologue fragment — fire-script + post-signal barrier — prepended inside the same `sh -c`, with the cd special-case branch (docker_env.py:74–80) integrated identically, §3.2), per-step monotonic `dur_ms` capture, `--init` + `--label tj3-mint=<seed>` container-run flags, per-step `extra_timeout` plumbed into the exec timeout; `collect_docker.py` — gen_sequence_v3, per-seq seeding, fresh-container collect_image, container bootstrap (`mkdir -p /tmp/.tj` + `after` helper install + the tini/keeper PID probe, §3.3) with the bidirectional canonical-PID map applied at exec and before storage (§5.2), store-time virtualization (uptime→`up <vt>` + fixed masks, mutated-path mtimes→`T+<vt>`, and the §5.4 ps canonicalizer — all applied before storage; render_canon owns only the measured render-side masks, §5.5), `MutGuard` (homed here, §4.4, with its §16 property tests), the `--audit-shadow` CLI flag (§3.4), probe split, gates-as-asserts, two-arm mint, the `timing-<split>.jsonl` side-channel writer; `benchmarks/scan_publish.py` — .pt inventory rows + the UD-8 exemption list (regenerable instrument tensors: sst-val.pt/wtm-val.pt/ROLL anchors) + ablate-root inventorying (§11.5, §16 step 13); `evolve/sanity.py` (+ calib_bench.py, path_battery.py, plan_eval.py) — ALL direct `M.cached_encode` callers route through the harness-owned cache wrapper on v3 roots (sanity's champion-validation arms are part of the promotion story; a stamp-less v3 cache must raise there too — round-7 S5); `realenv/plan_env.py` — the run side resolves perception via `load_perception_for_root` and goes through the harness-owned cache gate on v3 roots (today `Enc` hardcodes enc_e5_base + raw cached_encode, plan_env.py:48–59 — round-5 M5); the plangoals-v3 harvest mode (a NEW input path: reads the mint jsonl + folds shell_state per trajectory rather than live-probing a pristine container; goals referencing filesystem-mutated state replay via the §3.6 unit (re-bootstrap + re-seed, then recorded commands); job-dependent goals excluded in v3.0 — §16 plangoals paragraph); new modules per §18. Each lands with its §16 test.

**plangoals-v3** (verdict:completeness M3): goals harvested from realized trajectories with mutation-aware semantics — a goal references the world state AT ITS HARVEST STEP (post-history), stratified by depth × first path component × state_scope; sha-pinned; review-C precondition. Owner: `realenv/plan_env.py` gains a v3 harvest mode — a NEW input path, not an extension of the stratifier alone: current harvest live-probes a pristine container (plan_env.py:93–146) and consumes no trajectory data, whereas v3 harvest reads the mint jsonl and folds shell_state per trajectory; goals referencing FILESYSTEM-mutated state carry their trajectory-prefix, and the run side replays it through the §3.6 SELF-CONTAINED REPLAY UNIT — collector-side re-bootstrap (`/tmp/.tj` + `after` install) and Tier-W re-seed REGENERATED from the §3.6 per-seq seed and verified against the step-0 manifest HASH, FIRST, then the prefix's recorded commands (recorded commands alone cannot recreate in-band seeding/bootstrap; round-7 S3). JOB-DEPENDENT goals are EXCLUDED from the v3.0 harvest (round-6 S2 ruling): job firing is unrecorded prologue scaffolding and kill steps carry canonical PIDs, so recorded-command replay cannot reproduce process state or fired-effect files — the harvest filter drops any goal whose state closure touches a job-effect path or process state; reconstructing fire schedules from meta job fields is a v3.1 instrument. In the §16 diff list.

**Tests** (`tests/test_collect_v3.py`; resolves verdict:completeness S5): the fakeable exec DockerBox seam survives (no PTY seam needed); gate-at-entry tests (timeout/125/digest/mismatch); planted-fault tests per DG-10d (kill a container mid-trajectory ⇒ image aborts; stale cache ⇒ raise; planted scanner canary; planted DG-1 lexicon payload; planted DG-7 duplicate); MutGuard property tests; render_canon unit tests incl. adversarial marker strings; tracker rule tests R1–R9 on hand-built trajectories; visibility-parity test; render-inert-meta test; **archived-margin regression**: the r13 champion's recorded 0.4781 final-test and the v1 0.5848 reproduce bit-identically under the v3 harness (per-verb columns, new arms, meta reading, v1/v2 caches).

---

## 17. Blocker-resolution ledger

Every blocker (B) and serious (S) finding from every verdict → where resolved, or OPEN. (Minors are folded inline where structural; the consequential ones are cited throughout.)

| finding | resolution |
|---|---|
| XL-B1 substrate fork | §3 exec-only; PTY→v4 (§15); UD-1 |
| XL-B2 mutation fitness mass excluded | §4.6 revisit re-mix; §6.1 COND re-template; DG-9 owned (§14) |
| XL-B3 three axis-3 definitions | §9.1 one rule (SST ≥0.90 ⇒ sim), denominator defined |
| XL-B4 payload provenance | §4.3 DG-1 law; §5 job effects mined |
| XL-B5 five trackers | §10.1 one module, two modes |
| XL-S1 seq-len | §7.2: 28±4 |
| XL-S2 weight budget | §7.2 one table, ≥60% floor |
| XL-S3 workspace fragmentation | §4.2 /tmp/w, one lexicon, G3 owns redirection |
| XL-S4 fs_delta probes at mint | §3.4/§8.2: tracker computes; probes pilot-only |
| XL-S5 twin-replay double-apply | §9.1 axis-4 read-only twins |
| XL-S6 denylist divergence | §4.4 union + infra helpers |
| XL-S7 timestamp triplication | §5.5 one table, three layers |
| XL-S8 fitness aggregation | §8.1 pooled; per-verb ledger |
| XL-S9 arm-list divergence | §10.3 seven arms frozen |
| XL-S10 image set | §12 (12 core; UD-5 train-only extension) |
| XL-S11 cost omissions | §3.7 aggregated budget |
| XL-S12 pipes ownership | §6 G3 first-class; ps banned; signature labeler |
| XL-S13/S14 axes + BNF | §9.1 columns added; §6.1 BNF ≡ G3 |
| CP-B1 images lens missing | §12 |
| CP-B2 meta threading/cache | §13 |
| CP-S1 budget owner | §3.7 + G-BUDGET (runbook owner) |
| CP-S2 ablate plumbing | §11.5 (--val-data; ablate flag) |
| CP-S3 mask placement | §5.5 ruling (store-time + frozen render_canon pre-perception) |
| CP-S4 bench_versions v3 spec | §9.5 + §13.2 (classes.json, sha, fail-closed) |
| CP-S5 test plan | §16 tests |
| CP-S6 runbook/freeze order | §16 |
| CP-S7 positional audit | §11.4 (28±4 moots the extension; rung audit kept) |
| IN-B1 Done-notification nondeterminism | mooted by exec (§3); mandatory-closure rule attached to the v4 spike (§15) |
| IN-S1 Tier-1 dead code | mooted (exec timeouts are subprocess-side); fix attached to v4 spike |
| IN-S2 dur_ms vs byte-diff | §3.4 side-channel |
| IN-S3 shadow pilot fencing | §3.4 fenced |
| TM-B1 kill -INT no-op; pending-TERM | §5.3 (-INT dropped; 5-state automaton) |
| TM-S1 launch stall/race | §3.3 (fd hygiene, explicit j, fifo barrier) |
| TM-S2 no post-signal barrier | §3.3 /proc barrier |
| TM-S3 launch-cutoff arithmetic | §3.3 schedule-aware guard |
| TM-S4 lexicon payloads / process-domain honesty | §4.3 mined effects; §5.4 DG-5; domain margin honestly expected ≈0 on sim cells, disclosed §10.5 |
| TM-S5 stale busybox-stat claim | §3.5/§12.2 (tier probe; busybox lands T2) |
| WD-B1 workspace echo loop | §4.3 DG-1 + §11.2 DG-7 + §14 DG-9 |
| WD-B2 mtime replay leaks | §5.5 table (runtime mounts, parent dirs, /w) + DG-3a/d |
| WD-S1 NSS-file mutations | §4.4 denylist extension |
| WD-S2 mv destination clobber | §4.2 [ -e ] probe |
| WD-S3 pool routing incomplete | §4.5 all pools tracker-filtered |
| WD-S4 SST un-gated, duplicated | §10.1/§10.4 (one tracker, DG-4a) |
| WD-S5 per-seq seeding; probe cache key | §3.6; §3.5 DG-10c |
| CM-B1 family-median echo dilution | §6.5 slice-level axis-2′ + sst_composite |
| CM-B2 pipe derivability | §6.5/§10.2 SST-G3 + DG-6 |
| CM-B3 readback ls mtimes | §6.3 names-only readback ls |
| CM-S1 stderr-folding mode rule | §6.4 existence-verified producers |
| CM-S2 find line counts | find-producers pruned (§6.1); counts recorded anyway (§3.5) |
| CM-S3 0.656 transplant | §6.5 rule-not-number, recalibrated |
| CM-S4 lexicon payload contamination | §4.3 (≤5% audited arm) + DG-7 |
| SQ-S1 kill-boundary race | §7.4 + §3.3 barriers |
| SQ-S2 scheduled ls -l timestamps | §4.6 TIME_FREE + §5.5 + stat format pinned |
| SQ-S3 redirect payload provenance | §4.3 |
| SQ-S4 probe observability (OBS_CAP + resolution) | §4.6 scoping rule + DG-2 |
| SQ-S5 schedule leak | §7.3 DG-8 + axis-1+ |
| SQ-S6 no mass floor | §14 DG-9 (owned) |
| TG-B1 oracle tensors genome-visible | §8.2 target-only + eval-path leakage assert |
| TG-S1 DG-2 precondition unstated | §11.2 L7 + the §8.5 pre-registered fallback fork (family exclusion → DG-9 breach → separately-embedded channel becomes the v3.0-blocking redesign proposal), referenced from the §14 DG-9 row |
| TG-S2 position-blind probe baselines | §8.3 position-aware arm |
| TG-S3 v3.1 rule on raw sign | §8.5 bracket + noise band + effect size |
| VC-B1 twin-flip masking wrong statistic | §5.5/§9.1 cross-replay flip rates; twins → ceiling only |
| VC-B2 mut slice unmeasured; rbc claim false | §9.4 per-cell axes + DG-1/7 + counterfactual foils; claim corrected |
| VC-S1 inheritance vs battery | §9.3 battery governs |
| VC-S2 twin-replay unscoped | §9.1 read-only/state-neutral only |
| VC-S3 axis-3 conservatism inverted | §9.1 merged rule; borderline ⇒ excluded side |
| HB-B1 encoding misbinding (passage prefix) | §10.3 root-perception binding + G-EMB |
| HB-B2 axis-3 denominator | §9.1 all-steps denominator + achievable anchors |
| HB-S1 tie-rule inverted | §10.3 corrected derivation |
| HB-S2 wrong-but-determined (cap marker, newline, dialect) | §10.2 amended R5/R6/R7 + §3.5 template harvest |
| HB-S3 BNF `<<<` / echo backslash | §6.1 BNF; §4.3 charset ban |
| HB-S4 dialect-detection openers | §3.5 probe-harvested templates (no opener reliance) |
| HB-S5 numeric gates missing | §10.4 DG-4a/b + mint-scale ⊥ assert |
| HB-S6 information-parity overclaim | §10.4 disclosed asymmetry |
| ES-B1 metric blind to mutation | UD-4 counterfactual foils in-metric (recommended) + G-SEP |
| ES-B2 no encoder-resolution gate | §11.2 L7 = DG-2 |
| ES-S1 tracker gate wrong channel | §10.4 DG-4a (rendered text) |
| ES-S2 central degeneracy unaudited | §11.2 DG-7 numeric |
| ES-S3 G1 vs ls -l; per-verb nondet scoping | §5.5 + opt-family scoping |
| ES-S4 seed protocol vs replay | §3.6 per-seq seeds |
| ES-S5 no eligible-mass floor | §14 DG-9 |

All 21 blockers and 61 serious findings across the 11 verdicts are resolved above; none dropped. Items that remain genuinely open are decisions, not defects, and live in §2 (UD-4, UD-5, UD-7, UD-8, and the OPEN list: cpid scheme, DG-7 constant, axis-2′ number, uptime/LN-CONTRAST retention, G-SEP δ; the d2 target is COMMITTED at 0.15±0.05 in §7.2).

---

## 18. Doc-sync obligations (same-commit; per root CLAUDE.md update triggers)

| doc | change |
|---|---|
| `terminal-jepa/CLAUDE.md` | new modules (realenv/shell_state.py, realenv/render_canon.py, realenv/verbsig.py, benchmarks/class_measure.py, evolve/instruments.py, evolve/precompute_baselines.py), new data roots (dockerfs3, dockerfs3-ablate, dockerfs3-e5, dockerfs3-ablate-e5 — the ablate arm's derived root, required by the §13.1 raw-root closure and budgeted in the §3.7 encode row), regen recipe |
| `terminal-jepa/README.md` | module/file inventory + reproduction commands |
| `terminal-jepa/evolve/CLAUDE.md` | fitness arms (7), scoring CLI (--val-data, --stats-root), target-only tensor contract, instruments, bench-version notes |
| `.claude/skills/evolve/SKILL.md` | threshold/CLI references kept consistent with evolve/CLAUDE.md |
| root `README.md` | post-mint: the v3 finding record (or the UD-7 negative) |
| `benchmarks/dockerfs3-prereg.md` | assembled from §14–§16 plus the explicit incorporation list (§8.5 promotion rules + fallback fork, §8.2 exit vocabulary, §10.3 seven arms, §10.4 information-parity disclosure, §6.1 BNF); amendments dated |
| `benchmarks/dockerfs3-runbook.md` | §16 deltas; prose rules that gate become CODE asserts |
| `benchmarks/dockerfs3-digests.json`, `benchmarks/dockerfs3-classes.json` | new tracked artifacts, shas in the prereg |
| `evolve/bench_versions.py` | v3 entry (the (sig, mode, state_scope) cell table + fitness-role map, classes-file loading, fail-closed resolve, ablate flag) — ACTIVE_BENCH does NOT live here |
| `evolve/archive.py`, `evolve/cli.py` | archive.py owns `ACTIVE_BENCH` (flips to "v3" only at the re-baseline commit) + `_bench_of` recognition of dockerfs3 roots; cli.py `--bench` choices gain "v3" + argparse wiring for `--val-data`/`--stats-root` |
| `tests/test_collect_v3.py`, tests for tracker/render_canon/meta-inertness | §16 suite |
| HF dataset card (`veryfansome/terminal-jepa-dockerfs`) | v3 roots, ablate root, artifact policy (UD-8) |
| bench-constitution.md | the dated amendments listed in §14 |
| auto-memory `evolve-insights` | re-baseline stats, neutrally |

New CLAUDE.md files (none planned) would need AGENTS.md symlinks; no CLAUDE.md deletions.
