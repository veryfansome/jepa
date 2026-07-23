# dockerfs3 (v3.0) mint — pre-registration

STATUS: ACTIVE (assembled 2026-07-22 from `benchmarks/dockerfs3-design-draft.md` per its STATUS-header assembly rule). Governed by `bench-constitution.md`; predecessor: `benchmarks/dockerfs2-prereg.md`. This prereg = draft §14 (gates) + §15 (scope) + §16 (freeze order, pilots, diff list, tests, reviews, runbook deltas) + the explicit incorporation list (§8.5 promotion rules + fallback fork, §8.2 exit vocabulary + emission/strip-seam rules, §10.3 seven arms, §10.4 information-parity disclosure, §6.1 BNF) + the version-identity declaration + the OPEN pilot-frozen numbers. No collection code is trusted before review A of this document; the mint proceeds only at the GO entry (§11) appended by dated amendment.

**User decisions (recorded)** — all eight UD-1..UD-8 DECIDED 2026-07-22, accepted as recommended (user sign-off; draft §2):
- **UD-1**: jobs/fg/bg/%n, `wait`, literal `^C`/`^Z`, bare `sleep N &`, `kill -INT`, `<<<` are OUT of v3; the process automaton ships by signals on plain `docker exec`; the PTY spike is a v4 charter (recorded in `BACKLOG.md`, inheriting the infra memo + its verdict's measured fixes).
- **UD-2**: composition enters at minority mass under the frozen depth-1 grammar G3 (§4.1); depth-2 exists only as an out-of-corpus eval-side probe battery (report-only).
- **UD-3**: multi-channel + multi-step ship as data + instruments, not fitness; promotion is a v3.1 decision under the frozen §4.3 rules.
- **UD-4**: option (b) — counterfactual pre-mutation-twin foils enter the v3.0 metric on the mutation slice (m=8 of 64 slots, same-verb foil arm only, only where pre/post renders differ, DG-2-gated). A version-boundary eval change, legal under constitution §5.
- **UD-5**: 12 re-pinned v2 images; up to 3 new TRAIN-ONLY images contingent on the P0 capability audit (§1.2); val/final stay exactly the v2 four.
- **UD-6**: constitutional amendment approved — the constitution §6 probe-ladder gate is enforced on the native-continuity slice only; ladder rank order on mutation/time/composition slices is reported, not gated (dated amendment, §3.1).
- **UD-7**: expansion abort criterion — pilot loop >6 iterations OR >3 calendar weeks without all gates green ⇒ v3 is shelved, recorded as a negative in README.md, dockerfs2 remains the active bench.
- **UD-8**: publish raw + primary e5 roots AND the ablate raw root; instrument tensors (`sst-val.pt`, `wtm-val.pt`, ROLL anchors) exempt as sha-pinned regenerables.

**Notation** (from draft §2): **F1** = host-executor artifact (never a recorded observation; occurrence aborts the image). **F5** = trajectory-abort semantics (barrier/timeout failure discards + re-collects the whole trajectory; no partial sequence stored). **F8** = recoverable-from-record (any eval-consumed flag recomputable from {cmd, output, exit, cwd}; meta may cache, never define). **F6** = train-only collection mode (ablate arm collects no val.jsonl). **D3** = miss-mode exclusion (error/absent-outcome observations excluded from fitness, report-only).

---

## 1. Version identity

- **bench_version = `dockerfs3-v3.0`.** Substrate: per-command `docker exec` only (no PTY; UD-1/D-B1). Fresh container per trajectory; per-sequence RNG stream `random.Random(f"dockerfs:{seed}:{image}:{arm}:{seq_idx}")`; sequence = self-contained replay unit.
- **Budget**: full arm 12 images × 900 seqs/image × **seq-len 28±4 (max 32 steps = 64 tokens — fits pos_emb(64) and the constitution §6 ASSERT; no version-identity table change)** ≈ 302K steps; paired **ablate arm** 8 train images × 450 seqs/image, TRAIN-ONLY (F6) ≈ 101K steps. Same mint run, same digests, same seeds (paired collection); mint seed recorded in summary.json and the container label `tj3-mint=<seed>`.
- **One-mint rule** (constitution §1/§3): both splits and both arms, one run, one policy. Abort disposition pre-committed (v2 Amendment-5 precedent): on ANY mint abort, delete the output dir wholesale and re-run the single one-command mint; no splicing, no resume; the completed run is the version's sole collection event. Mint aborts on any skipped image; digest entry gate in code (`--pin-digests` + `--expect-digests` required at collect() entry).
- **Toolset (enumerated)**: v2's nine (uname, cd, ls, cat, head, tail, stat, find, grep) + pwd, echo (redirect producer + ~1% bare), rm, mv, ln/ln -s, readlink, mkdir, touch, ps, kill (signal family), after (bgjob launch), uptime, sleep + the 11 G3 composed families (§4.1). jobs/fg/bg/wait/^C/^Z/`<<<`/kill -INT excluded (UD-1).
- **Policy identity**: `gen_sequence_v3` — the v2 flat verb-mixture wrapped in an event scheduler + collection-mode tracker + probe event queue; the per-arm start-weight table below is the SINGLE SOURCE OF TRUTH (copied VERBATIM from draft §7.2; the draft is superseded on this table by this prereg):

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

Derived (inline check): atomic v2-verb plain .610 ≥ the .60 floor; mutation atomic .104; composition .150 (m_redirect booked here ONLY); time .095; pwd .031; echo-bare .010; **total = 1.000 exactly**. Motif pre-emption draws against these budgets; skip-redistribution flows to flat cat, recorded per image. Interventions/seq 6±2; ≤8 mutations/seq, ≤4 Tier-S; jobs ≤3/seq; probe coverage ≥0.75; error-outcome obs rate 0.10–0.20 (report channel); chain-depth d2 target 0.15±0.05 (COMMITTED). Policy content-hash (`lexicon_hashes()`) keys the probe cache and is stamped as `policy_sha` in `cache_meta.json`. The ablate arm is the identical policy with mutation/time/composition arms off, weights renormalized.
- **Classes-file authority**: `benchmarks/dockerfs3-classes.json` — machine-readable rows = (sig, mode, state_scope) cells (plus the created-scope `ws_observed` sub-cells `"sig|mode|created"` / `"sig|mode|created-obs"`), columns = class + all axis statistics + coverage. Sha-pinned in this prereg by the class-freeze amendment; `bench_versions.resolve()` loads it and asserts summary.json matches verbatim. The `content=` set the harness consumes is the set of content CELLS, not verbs. Cell pseudo-verb strings are ATOMIC KEYS, never parsed by splitting.
- **Version-identity code**: `realenv/render_canon.py` (any change is version identity), the ONE v3 retrieval-code change `retrieval(forced_foils=...)` + `content_retrieval`'s subset-seam translation (Darwin-Gödel guard; archived-margin regression proves v1/v2 scoring bit-identical with the parameter absent), and the seven-arm max (constitution §5 version boundary).

### 1.1 Images (draft §12)

Core = the 12 v2 images, re-pinned fresh (the :latest tags have drifted since dockerfs2-digests.json; the drift's effect on the continuity slice is measured and reported at P0 as a continuity note). Digests → `benchmarks/dockerfs3-digests.json`; collection runs by digest ref with the v2 `--pin-digests`/`--expect-digests` entry gate.

| split | images |
|---|---|
| train (8) | alpine:latest, ubuntu:latest, debian:stable-slim, python:3.12-slim, redis:7-alpine, nginx:stable-alpine, postgres:16-alpine, node:22-slim |
| inner-val (2) | fedora:latest, mariadb:latest |
| final-test (2) | rockylinux:9, httpd:2.4 |

**P0-contingent train-only candidates** (UD-5; trigger: P0 shows <3 of 8 train images with a usable `ps`): `debian:stable` (full; procps + rich /usr tree; low risk), `opensuse/leap:15.6` (new family, procps, distinct /etc layout; moderate risk), `archlinux:base` (rolling glibc, procps, distinct filesystem conventions; moderate risk). RHEL-family candidates (alma/oracle/amazon) EXCLUDED — fedora/rocky are the held-out family. New images pull digest-pinned, enter TRAIN only, never touch val/final, and are declared in the version tuple; if adopted, seqs/image may drop to keep the step budget (e.g. 14 img × 780 ≈ 306K).

### 1.2 P0 capability audit (pilot gate, per image, before any policy pilot)

`command -v` for {pwd, uptime, sleep, ps, echo, kill, rm, mv, ln, readlink, mkdir, touch}; ps tier T2/T1/T0/ABSENT (**ps is the decision variable**); `/usr/local/bin` writability (the `after` gate); `/bin/sh` identity; error-template harvest; fifo support; fractional-sleep support (`sleep 0.1` — probed, never assumed). Absent ⇒ v2 skip-and-redistribute, recorded.

---

## 2. OPEN pilot-frozen numbers (rule committed, number pending; each freezes by dated amendment on pilot data)

1. **Canonical-PID scheme** (100+10j vs seeded-stable arbitrary) — the watermark risk is priced by the DG-8b position probe; the cpid scheme freezes only after its watermark price is measured — by dated amendment on pilot data (P3).
2. **DG-7 reference constant** — the v2 cross-split near-dup base rate; MUST be measured on dockerfs2 BEFORE any v3 work starts (freeze-order step 2); frozen by dated amendment.
3. **Exact axis-2′ threshold** — rule committed: max-margin midpoint recalibrated on v2 known cases (linked-grep hits = content side, cd = echo side); 0.656 is the prior; the pilot freezes the number by dated amendment. **P3 pilot pins it: axis-2p ≈ 0.96** — above linked-grep-hits (native 0.847 / mutated 0.927 ≈ 0.93) and below the true echo-loop readbacks (`cat|hit|created-obs`, `readlink|hit|native` = 1.000). At 0.656 grep is wrongly semi-echo (a content-surface gap, not a leak); at ~0.96 grep returns to content while genuine workspace-echo readbacks stay excluded. Freeze the number here.
4. **uptime's slot and the hard-link LN-CONTRAST motif** — pilot-measured, dropped if invisible; retention decided by dated amendment on pilot data.
5. **G-SEP separation margin δ** — the tracker-vs-within_traj gap under counterfactual foils, measured at P3 (both arms training-free); frozen by dated amendment.

Additionally, every gate threshold written "pilot-measured" in §3 (DG-1 per-family top-1 bound, G-EM benign base rate, G-CF realized injected-slot-rate floor, the axis-1 θ1 recalibration) freezes the same way: by dated amendment on pilot data, before the class freeze it feeds.

---

## 3. Pre-registered gates (consolidated; draft §14)

All fail-closed. The `when` column is binding: most gates measure on pilot data before class freeze; DG-9 measures at P5 (post-freeze, pre-mint) by design; MINT rows assert at mint; G-LAD at re-baseline; G-SEP is measured at P3 (both arms training-free) and enforced at re-baseline; G-BUDGET is rolling. Each gate is ratified only after a planted-violation self-test (DG-10d — no gate counts until it has rejected a planted failure). Owners are lenses-as-roles in the runbook. **29 gates total.**

| gate | what | threshold | when | owner |
|---|---|---|---|---|
| DG-1 | payload provenance: delayed-copy predictor on revisit reads; cross-seq payload dup | top-1 < pilot-measured per-family threshold, frozen by amendment (a top-1 accuracy bound — not the axis-2′ containment number); dup <5%; lexicon arm ≤5% w/ payload_src | pilot P3 | world-dynamics |
| DG-2 (=L7) | foil/encoder resolution on mutation probes | median d_pair >3× noise floor; ≥90% steps above floor, per family | pilot P3 | eval-integrity |
| DG-3a | twin-mint byte-diff (2× pilot, same seeds) | 100% or enumerated-mask + clean re-run | pilot P1 | infra |
| DG-3b (=G-J) | jitter invariance (injected 0–300ms delays) | byte-identical jsonl | pilot P1 | time |
| DG-3c | nondeterminism scanner (renders: dates/nonces/pids/sentinels/helper names) | zero hits; planted-canary self-test | pilot+MINT | infra |
| DG-3d | -l-family field policy on mutated paths | per the §3.2 table below; twin-mint clean after masks | pilot P1 | time |
| DG-4a | SST fidelity (rendered text) | exact-match ≥0.995 per determined cell | pilot P3, pre-freeze | baselines |
| DG-4b | one-tracker rule + independent cross-check | disagreement = 0 | pilot P1 | baselines |
| DG-4c | visibility parity (mining ≤ render cap) | unit test green | always | sequencing |
| DG-5 | process-arm entropy: ps distinct renders; single-render mass; job-state mixture | ≥25/image; ≤25%; {run,stop,term,mixed} ≥15% each | pilot P2/P3 | time |
| DG-6 | composed derivability + SST-G3 | SST-G3 exact-match ≥0.95 on derivable steps, else family sim | pilot P3 | composition |
| DG-7 | cross-split contamination (val mutation slice vs train) | near-dup rate ≤ v2 reference constant (measure FIRST) | pilot P3 + MINT | eval-integrity |
| DG-8 | probe-outcome mix + axis-1+ schedule leak | {70/20/10}±5%; axis-1+ top-1 < 0.5× within_traj on probe slice | pilot P2/P3 | sequencing |
| DG-8b | position-probe: the position-aware predictor (cmd + step-index + steps-since-last-mutation) as watermark pricer — prices the canonical-PID / step-index watermark (§2 OPEN) and joins the probe baseline bracket (§4.3) | axis-1+ top-1 < 0.5× within_traj on the probe + process slices; the cpid scheme freezes only after its watermark price is measured | pilot P3 | sequencing |
| DG-9 | **aggregate eligible mass** (post-freeze, pre-mint; breach ⇒ the §4.3 pre-registered fallback fork) | mut-affected content ≥10% of content steps; composed ≥6% of content steps; per-image ≥ half these floors | pilot P5 | **eval-integrity (named owner)** |
| DG-10a | trajectory-death fail-fast (container dies mid-seq ⇒ abort image) | zero survivors; planted-fault test | MINT | infra |
| DG-10b | zero-timeout / zero-125 asserts at collect() entry | 0 events | MINT | infra |
| DG-10c | probe-cache key (digest, policy-hash, collector version) + cold/cached twin | dict-equal | pilot P1 | infra |
| DG-10d | gate-ratification meta-gate: every gate in this table must reject a PLANTED violation before it counts as ratified | one recorded planted-failure rejection per gate, pre-GO | pilot (rolling, before review C) | eval-integrity |
| G-EM | expected-vs-realized mismatch | ≤ pilot-measured base rate (target ≤1%) | pilot P1 | world-dynamics |
| G-RATE | all realized policy rates (linkage, revisit, probe mix, weights) | ±5% of targets | pilot P2 + MINT | sequencing |
| G-⊥ | ⊥-share ≥40% / determined ≥15% per content verb | mint-scale assert | pilot + MINT | baselines |
| G-EMB | SST render-parity embedding gate | cos ≥0.9999 on exact-text determined steps | pilot P3 | baselines |
| G-CF | counterfactual-pair floor | ≥30 differing pre/post pairs per image (ls-class) AND the realized injected-slot-rate floor on the mutation slice (pilot-measured) | pilot P2 | eval-integrity |
| G-COV | ≥30 cross-image pairs per cell (else battery); ≥500 steps/content verb/split | as stated | pilot P3 + MINT | classes |
| G-DIG | digest entry gate (--pin/--expect) | exact match | MINT | infra |
| G-LAD | ladder gate: mse < InfoNCE < evolved rungs on native-continuity slice (v2-mode-rule column) | strict, no non-adjacent inversions | re-baseline | eval-integrity |
| G-SEP | tracker/within_traj separation under counterfactual foils on mutation slice | tracker >> within_traj (pre-registered δ, §2) | measured pilot P3 (both arms training-free); enforced re-baseline | eval-integrity |
| G-BUDGET | mint ≤3h; pilot iterations ≤6 / ≤3 weeks (UD-7) | as stated | rolling | runbook owner |

### 3.1 Constitutional amendments required (dated, applied in bench-constitution.md in the same commit)

- Axis-3 `sim` class + the (sig, mode, state_scope) classification unit (constitution §4 extension).
- Seven-arm baseline max (constitution §5, version boundary).
- Native-slice scoping of the constitution §6 ladder gate (UD-6): the probe-ladder gate is enforced on the native-continuity slice only; ladder rank order on mutation/time/composition slices is reported, not gated.
- Constitution §6 instrument arithmetic unchanged (28±4 fits).
- Calibration-on-v2 (not v1) dated deviation note.

---


### 3.2 Per-field nondeterminism policy (copied VERBATIM from draft §5.5 — the frozen table DG-3d gates against; flip-rate statistic measured across replayed collections, not same-session twins)

| field | layer | policy |
|---|---|---|
| PIDs (everywhere) | store-time | virtualize (canonical, bidirectional) |
| uptime elapsed | store-time | virtualize → vt |
| mtimes of trajectory-mutated/created paths | store-time | virtualize → `T+<vt_of_mutation>` |
| mtimes/dates of untouched shipped files | — | leave raw (image-constant facts) |
| runtime-mount mtimes (resolv.conf, hostname, hosts) | render-canon | mask (fresh per container start) |
| wall clock, load, users, cpu TIME | render-canon | mask to fixed tokens |
| parent-dir mtimes of mutated dirs (`ls -l` family) | render-canon | mask time fields on `-l`-family renders of tracker-touched dirs |
| inodes, size-tiebreak orderings | template avoidance | `-i`,`-lt`,`-lS` dropped from LS_OPTS for mutation-adjacent steps; TIME_FREE opts for revisits |
| etime/ELAPSED, scheduling order | excluded | never requested / closed by construction |

Fields still differing in the twin-mint after this table are enumerated into the mask list or their (sig, mode, opt-family) cell goes excluded-nondet — scoped per opt-family, never per verb.

#### §5.5 AMENDMENT (2026-07-23): `-l` mtime mask moved render-canon → STORE-time; `-l` pseudo-fs / recursion exclusions

Rows 130–133 of the table above are amended by the pre-mint collector determinism review (twin-mint on real alpine + fedora, `collect_image_v3`, raw-jsonl byte-diff). `render_canon.py` is version identity, so this is a dated amendment + re-baseline; it lands pre-mint (no published data affected).

- **`-l`-family ls date/time triplet → STORE-time, UNCONDITIONAL.** The `-l` mtime mask (rows 131 & 133) MOVES from the render-canon (encode-time) layer to the collector's STORE-time path (`_V3Session.do`, before a step is stored), applied to EVERY `-l`-family long-listing row. Rationale: **DG-3a (and the twin-mint determinism invariant) diffs the RAW recorded jsonl BEFORE any encode**, so an encode-time render-canon mask cannot make the stored bytes deterministic; and the old touched-set scoping missed `/`, `/etc`, `/tmp`, whose dir mtimes are set at CONTAINER-CREATION time by docker bind-mounts / the unrecorded bootstrap (the SEVERE-1 leak). Consequence: row 130 ("leave raw" for image-constant shipped-file mtimes) is SUPERSEDED — those `-l` times are now masked too, the pre-registered determinism/fidelity trade. `predict()` is BOT for `-l` (R6), so store-time masking is golden-rule-parity-safe; the 3-token `LS_TIME_TOKEN` preserves the 9-field row the SST `-l` child-splice reads. `render_canon.canon()` re-applies the SAME mask via the shared `render_canon.canon_ls_l_text(output)` helper — now UNCONDITIONAL / state-independent — as defense in depth (a fixed point on collector-canonical bytes).
- **`-l` of `/` → names-only; `-R` dropped; `/proc`,`/sys`,`/dev` excluded from ls/find target pools (SEVERE-2).** `-R` is removed from the v3 ls opt pool (recursion into `/proc` renders live-PID dirs and risks the 8s cap → DG-10b). `/proc`,`/sys`,`/dev` are excluded from every flat-ls / find / cd target pool (they are volatile pseudo-filesystems; the world is the real image fs). An `-l`-family listing of `/` (the only dir holding those mounts) is downgraded to a names-only form, because a mount's row link-count (`/proc` = live subdir/PID count) is replay-volatile even after the time mask. These keep volatile pseudo-fs metadata out of every recorded byte; row 137's "enumerate or exclude" clause covers the residue.
#### §4.4 + DG-3c AMENDMENTS (2026-07-23, from the P1 determinism pilot)

- **DG-4b — SST quotearg soundness (sha-pinned SST change).** The P1 pilot's tracker cross-check found 1/1199 determined predictions divergent: GNU coreutils single-quote an error-message filename ONLY when it contains a shell-special char (`quotearg` shell-escape-if-needed), while busybox always-quotes — so a determined error-template prediction for a special-char path (the pilot case: debian `cat /var/lib/dpkg/info/libsystemd0:arm64.shlibs`, colon-quoted) is dialect-divergent and unknowable from the template alone. FIX: `shell_state._tmpl` now BOTs any error prediction whose path kwarg holds a `_QUOTEARG_SPECIAL` char (space/tab/nl/`:'"()[]{}|&;<>*?$`\\!#~`). Byte-deterministic gap (twin/jitter passed), sound-by-conservatism, ~1/56 recorded errors affected; DG-4a's 0.995 bar was already met (517/518). Standing test: `test_shell_state.test_quotearg_special_char_path_error_is_bot`; the differential suite's per-image templates exercise it live.
- **DG-3c exemption — image-constant date/time content.** The scanner's date/time/IP hits on the pilot are ALL image-constant, digest-pinned, twin-identical file *content* (not renders of live state): os-release `SUPPORT_END`, locale `PO-Revision-Date`, config-example IPs (127.0.0.1/192.168.x), md5 checksums, and — added here — `/var/log/apk.log` build-time stamps (`… at YYYY-MM-DD HH:MM:SS`, present in some alpine-derived images). These are pre-registered EXEMPT from DG-3c (they are facts of the frozen image, proven constant by DG-3a's twin byte-identity), distinct from the store-time-masked `<date>` in system_id and the dropped `/proc/version`. The DG-3c canary self-test scans for NON-exempt nonces (runtime PIDs, container-ids, per-run IPs, wall-clock) — zero hits across 4704 pilot steps.

- **Host-fingerprint exclusions (v3 cat/uname pools + system_id; determinism re-review).** Three host-specific / date-bearing sources are removed from the v3 identity arm so the raw jsonl is cross-host replayable and DG-3c date-scanner clean: (a) `/proc/version` and `/etc/resolv.conf` dropped from the v3 config-cat pool (`_V3_CONFIG_FILES`) — the first leaks the kernel build-date (a literal date → DG-3c false positive) and is a pseudo-fs path (SEVERE-2 consistency), the second carries the HOST DNS nameserver (flips on a different host); (b) `uname -a` and `-v` dropped from the v3 uname pool (`_V3_UNAME_OPTS`) — the only forms embedding the build-date; (c) the top-level `system_id` field (still `uname -a`+os-release) has its kernel build-date store-time-masked to `<date>` via `_canon_system_id`, preserving distro/arch/hostname identity (the syscond context) while removing the DG-3c-tripping literal. os-release/hostname/hosts/issue/passwd/group stay (image-constant, `--hostname`-pinned). v3-only constants; the shared `CONFIG_FILES`/`UNAME_OPTS` are untouched so v1/v2 records stay byte-identical.

## 4. Incorporated normative specifications

### 4.1 Composition grammar G3 (draft §6.1; frozen at this prereg — exactly one operator, no recursion)

```
COMPOSED ::= PIPE | REDIR_W | COND
PIPE     ::= PROD "|" FILT
PROD     ::= "ls -1" D | "cat" F                  # find-producers pruned (D-S2); ls -l banned
FILT     ::= "head -n" K | "tail -n" K | "grep -F -m 8" TOK
REDIR_W  ::= PROD (">"|">>") WSF | "echo" 'PAYLOAD' (">"|">>") WSF
COND     ::= "[" TESTOP P "]" "&&" READ P         # TESTOP ∈ {-e,-f,-d,-s}; READ ∈ {cat, ls -1, head -n K}
```

Excluded: `<<<` (non-POSIX), `||`, `;`, `if/then`, depth-2 pipes, REDIR_IN, `wc -l`/count filters, `cd` in any composed string, `ps` in pipes. The BNF's totality universe is the FULL command grammar — the v2 atomic templates (the ≥60% majority mass) + this G3 grammar + the audited process forms (`after j K 'effect' & echo $!`, kill family); the SST parser asserts totality against that whole universe. 11 measurable composed families — 6 pipe + 2 redir + 3 cond; the complete `meta.sig` vocabulary: `pipe:ls|head, pipe:ls|tail, pipe:ls|grep, pipe:cat|head, pipe:cat|tail, pipe:cat|grep, redir:echo>, redir:prod>, cond:cat, cond:ls, cond:head` plus first-token verbs for simple commands (bare `echo` ~1% is a simple verb). Modes: pipe-grep {hit, miss}; cond {hit, miss}; others {ok}.

### 4.2 Exit vocabulary, emission ownership, and the strip seam (draft §8.2)

**Exit vocabulary (frozen at this prereg)**: `exit_cls` classes = {0 ok, 1 err, 2 usage, 126 not-executable, 127 not-found}; 124 timeout retained for pilots but zero at mint; 125 = host artifact ⇒ abort, never a class. 130, 137, 143, and 148 are ALL REMOVED under one no-producer rule (UD-1/the 5-state automaton), applied symmetrically: SIGINT has no source once `^C` and `kill -INT` are out; stopped jobs produce no exit at all (a `T` job stays alive; its state is visible only via the ps channel); every kill-family signal (TERM/KILL) targets background cpids whose exits are never a recorded step's exit; the watchdog's `kill -9` hits the `after` helper; an OOM-137 on a recorded `docker exec` is a container-death abort under DG-10a, never a class. The mapping is TOTAL with SPECIFICS-FIRST precedence: {124 → pilot-only class, mint abort; 125 → abort; any value ≥128 (incl. bare 128) → build-time ABORT (fail closed, never classify); negative host-side returncodes (docker CLI signal-killed) → F1/abort family, never a class; then 0 → ok; 1, 2, 126, 127 → their classes; any remaining nonzero → 1 err catch-all}.

**Emission ownership (one rule)**: the collection-mode tracker computes the delta text at generation and stores it in meta as an AUDIT copy (`meta.delta_text`, feeding DG-4a). The derived-root builders (`reencode.py`/`mv_encode.py`) RECOMPUTE delta text, `exit_cls`, and `pre_obs_step` at build time from the raw jsonl via the same shell_state re-fold that powers render_canon, ASSERT equality against the audit copy (F8 — recompute, don't trust), and encode the recomputed delta texts into `z_delta` through the root's perception OBSERVATION render path (one pinned convention shared with DELTA-PROBE's reference embeddings, e.g. e5's "passage: " prefix). The CELL-DEFINING columns (sig via `realenv/verbsig.py`, mode — recomputable from exit+empty, state_scope, mut_affected, ws_target, ws_observed) are ALL recompute-ASSERTED from the same re-fold; the SCORE-TIME recompute defines the cell, the cached column only accelerates it. Remaining cache columns (payload_src, intended_outcome, exit, empty) are copied from the record verbatim. Canonical delta format (sorted paths, ≤8 entries + `(+N more)`, size field for append checkability):

```
delta: none
delta: removed /etc/foo.conf
delta: created /tmp/w/notes.txt(42B), moved /etc/a -> /etc/a.bak
delta: appended /tmp/w/task1.log(+18B)
```

**The strip seam (function-level contract)**: the harness owns `_strip_target_only(seqs)` — a shallow per-sequence copy with the `exit_cls`/`z_delta` keys REMOVED — applied before EVERY `stream.collate(...)` and `stream.flatten_predictions(...)` call (the only two places genome stream code receives seq dicts). The batcher chunk receives the stripped train-side fit and returns batch indices only. Aux targets attach inside `_train` from the harness-held ORIGINAL seq dicts, indexed by the batcher's indices, via the immutable aux-target plumbing — DORMANT in v3.0 (no sanctioned consumer; it ships so the v3.1 flip is a config change). The leakage assert lives IN the immutable eval path: it perturbs exit_cls/z_delta in the harness-held originals at all positions and re-runs the stripped-copy pipeline — cmd_t predictions must be bit-unchanged. Oracle-annotation inputs (causal consumption of delta/exit as input tokens) are RULED OUT for v3.0 — pre-registered.

### 4.3 v3.1 promotion rules + pre-registered fallback fork (draft §8.5; frozen now)

A channel may be PROPOSED as v3.1 fitness iff, on the v3.0 champion at 3/3 seeds: probe margin over the FULL baseline bracket (incl. tracker and the position-aware probe) ≥ **0.05 absolute** AND above the pre-measured probe noise band. A ~0 reading is interpreted via the bracket: tracker-saturated ⇒ "info present, channel covered" (no training-level proposal); genuinely low with weak baselines ⇒ the separately-embedded multi-channel training target earns the v3.1 proposal. ROLL@4: margin over the tracker arm ≥ noise band + 0.02.

**Pre-registered fallback fork (a v3.0-BLOCKING path, not a v3.1 option)**: if DG-2's family exclusions drive the mutation-slice eligible mass below the DG-9 floor, the redesign path is fixed IN ADVANCE: the separately-embedded delta channel (the alternative rejected for v3.0 primary) becomes the v3.0-blocking redesign proposal. The mint stays blocked until either redesigned probes pass DG-2 with DG-9 satisfied, or the separately-embedded channel is ratified by dated amendment. Referenced from the DG-9 gate row.

### 4.4 The seven baseline arms (draft §10.3; frozen at this prereg)

**retrieve_by_cmd, no_history, copy_prev, within_traj** (exact ratified v2 definition — continuity), **within_traj_mut** (retrieved stale render patched with the overlay delta, re-encoded; patcher reuses the one tracker; mut-vs-plain is a reported diagnostic, not an assert), **sst** (⊥→zeros), **sst_composite** (SST-where-determined else within_traj_mut — expected to dominate; the aggregate-max fix). All as constitution §5 aggregate columns + solo ledger columns per verb. Fitness shape unchanged: pooled content top-1 margin = `content_top1(WM) − max(aggregate arms)` over the measured content cells, on inner-val; counterfactual foil injection per UD-4(b) is the ONE v3 retrieval-code change (version identity). SST/wtm predictions are rendered and encoded via the ROOT'S perception module (`reencode.load_perception_for_root(root)`, fail-closed on stamp-less v3 roots), with `render_canon.canon` applied to each predicted step-dict before the perception render, standardized with train stats — never via realenv render_obs. Render-parity gate: G-EMB (cos ≥0.9999 on exact-text determined steps). Tie rule: `_rank_stats` counts only strictly-closer foils, so an exact prediction keeps top-1 against identical foils; near-constant cells are removed as degeneracy, not tie-refusal.

### 4.5 Information-parity disclosure (draft §10.4)

The SST reads OBS_CAP=1600-char renders; the encoder's 256-token window covers ~1000 chars — the asymmetry is conservative (it strengthens the baseline) and is DISCLOSED here rather than claimed away. Companion floors are gated (G-⊥: ≥40% SST-undetermined eval steps per content verb AND ≥15% determined per content verb — one denominator, mint-scale assert; joint satisfiability with ≥500 steps/verb arithmetic-checked at pilot). Freeze-then-audit: SST rules + template tables are authored from train-pilot renders + docs only, content-hashed, frozen pre-mint; post-freeze edits = dated amendment + re-baseline. Calibration-on-v2 (not v1) is a dated deviation note (§3.1).

---

### 4.6 P3 CLASS-MEASUREMENT PILOT FINDINGS (2026-07-23) — calibrations to resolve at the class freeze (step 7)

The P3 pilot (160 seqs / 4512 steps, 4 shells, `benchmarks/class_measure.py` + e5, DRAFT `benchmarks/dockerfs3-classes.json`, honesty-reviewed) confirms the world is HONEST: 8 content cells (2244 steps), the ≥40% SST-undetermined floor MET for all 6 content verbs (0.928–1.000), the workspace echo-loop caught by axis-2′ (`cat|hit|created-obs` uniformly 1.000 → semi-echo), DG-9 mut-affected content **15.7% PASS** (≥10%). Four calibrations, each frozen by dated amendment at step 7 (ideally re-measured at mint scale):

1. **axis-2′ threshold → ~0.96** (§2 OPEN item 3, now pinned) — returns linked-grep-hits to content.
2. **Per-step echo purge in content cells (Finding 1, a v2 command-echo-class recurrence).** Cell-MEAN aggregation lets a per-step minority of command-echo readbacks into content cells; the leak is concentrated where an `echo>`-created workspace file is later `mv`/`rm`-touched, flipping `state_scope` created→mutated and BYPASSING the ws_observed split → the echo readback lands in `cat|hit|mutated` content. Pooled command-uncovered leak = **8/2244 = 0.36%** (margin impact ≲0.004). FIX at freeze: the harness's classes.json→pseudo-verb rewrite (§9.5) additionally masks any content-cell STEP whose axis-2′ (recomputed at eval, incl. through-mutation echo propagation) ≥ the frozen threshold to an excluded `<cell>-echo` pseudo-verb (generalizing grep-miss) — a per-step purge, not per-cell-mean. Zero command-echo in content, by construction.
3. **θ1 recalibration must stay ≥ ~0.46** — `cat|hit|native` axis-1 = 0.450, `tail|hit|native` = 0.424 sit nearest the echo/const gate (honest headroom, not a leak); confirm the recalibrated θ1 does not eject them and shrink content mass.
4. **Composed content = 0% at pilot scale (DG-9 composed ≥6% not yet measurable).** Every G3 family is coverage-starved (<30 pairs) at 40 seqs/image; their axes are content-shaped. Re-check at the mint's 600 seqs/image; the §4.3 fallback fork applies if still breaching.

## 5. Scope: v3.0 vs v3.1 vs v4 (draft §15)

**v3.0 (ships now)**: everything in draft §3–§13. Exec substrate; mutation + revisit world; kill-signal automaton + ps + after-jobs; G3 composition; 7-arm baseline max; fused-render pooled fitness + counterfactual foils per UD-4(b) (DECIDED); exit/delta data channels (target-only); EXIT/DELTA probes + ROLL@{2,4} as instruments; ablate arm; the full §3 gate battery.

**v3.1 (pre-registered forks, no new collection)**: promotion of exit or ROLL channels into fitness per §4.3; counterfactual top-1 as a fitness component; per-stage pipeline exits as meta; miss-prediction report slice as a candidate channel; depth-2 probe battery results informing a grammar extension proposal.

**v4 (separate charter)**: the PTY spike — jobs/fg/bg/%n, literal ctrl-keys, `wait` — inheriting the infra memo PLUS its verdict's mandatory fixes (bg-job closure semantics; Tier-1 resend; sentinel regex nonce + trailing newline; shadow fencing). Trigger condition: the v3 process-dynamics slice shows the model saturating the signal-delivered automaton. Also v4-or-later: `<<<`, variable expansion/globbing in payloads, multi-channel training targets if the v3.1 probe fork selects them.

**Retired framings stay retired**: no held-out-tool, no synthetic-ontology substrate, no training-signal-auxiliary re-propose (R10/R11 recorded priors; retry only under the §4.3 evidence fork).

---

## 6. Freeze order & pilot protocol (draft §16)

**Freeze order**: (1) substrate decision = exec (done, D-B1) → (2) P0 images/capability audit + DG-7 reference-constant measurement on dockerfs2 → (3) tracker module + render_canon implemented, unit-tested → (4) P1 mechanism pilot (determinism: DG-3a/b/c/d, DG-4b independent cross-check disagreement = 0, DG-10c, G-EM) → (5) P2 rate tuning (G-RATE, G-CF, DG-5, DG-8 mix) → (6) P3 class + honesty measurement (all four axes + axis-2′/DG-6 columns, DG-1, DG-2, DG-4a, DG-5, DG-7, DG-8b, G-EMB, G-COV; G-SEP measured here — both arms training-free — enforcement stays at re-baseline) → (7) SST freeze-then-audit; render_canon freeze; class table freeze BY AMENDMENT → (8) P4 post-fix verify (re-verify every frozen number) → (9) P5 = DG-9 aggregate mass check → (10) reviews A/B/C (adversarial, convergence-ruled) → (11) prereg GO → (12) the one mint (full + ablate arms, digest-gated, gates as code asserts) → (13) encode + publish (scan_publish extended for .pt inventory/exemptions) → (14) re-baseline + plangoals-v3 harvest → (15) README/insights record.

**Pilot sizes**: P0 = probe-only (no trajectories); P1/P2 = 24 seqs/image × 12 (~8K steps, ~3 min at the measured per-step model); P1's twin arms double it; P3 = 60 seqs/image × 12 (~20K steps — class measurement needs the coverage floors); P4 = P3-sized rerun; P5 = analysis over P3/P4 data (no new collection). Shadow-audit collections, where used, are separate arms outside P1/P3 — class-measurement and determinism pilots run shadow-OFF; shadow-ON never feeds class freezing.

**Pilot iteration cap**: UD-7 — >6 iterations or >3 calendar weeks without all gates green ⇒ NO-GO, shelve, record the negative.

---

## 7. Implementation checklist (v2→v3 harness/CLI diff list; draft §16 — each item lands with its §9 test)

- `evolve/bench_versions.py`: v3 entry — the (sig, mode, state_scope) cell table + fitness-role map {content, semi-echo, ack, echo/const, sim, noisy-excluded, report-only}, classes-file loader (`dockerfs3-classes.json`, verbatim summary assert), ablate-flag recognition (`"ablate": true` summary flag, never val.jsonl sniffing). ACTIVE_BENCH does NOT live here.
- `evolve/archive.py`: owns `ACTIVE_BENCH` (flips to `"v3"` only at the re-baseline commit); `_bench_of` recognizes dockerfs3 data roots.
- `evolve/cli.py`: `--bench` choices gain `"v3"`; argparse wiring for `--val-data` and `--stats-root`.
- `harness.py`: `_data_tensors` meta columns + slices + counterfactual foil rows + the cell pseudo-verb rewrite (`"sig|mode|scope"`, generalizing `"<verb>-miss"`; sig re-derived from cmd text via verbsig and ASSERTED, F8) + 3 new arms + the `_strip_target_only` seam (§4.2) + eval-path leakage assert incl. new tensors + `_base_for(..., stats=(mo, so))` signature change + base_cache keying extended with the arm set + classes_sha + root artifact sha + train-set descriptor (`full` or `sub<seqs-per-image>:<subsample-seed>`) + val/stats roots' artifact shas.
- `score_genome`: `--val-data`, `--stats-root`, `--subsample-seqs`, `--subsample-seed` — the last two wire the train-side seeded subsampling (full root → 450 seqs/image) for the ablate comparison arm and feed the train-set descriptor.
- `realenv/verbsig.py` (NEW): the ONE `composed_verb()`/sig labeler, imported by collector, harness, and class_measure.py; v1/v2 keep first-token verb_of bit-identically.
- `realenv/seq_worldmodel.py`: the ONE v3 retrieval-code change — `retrieval(forced_foils=...)` + `content_retrieval`'s subset-seam translation (row-subset, full→subset value remap, drop-outside-subset with `cf_dropped` ledger column and the G-CF realized-rate floor); `encode_split`/`_rank_stats` and everything else UNCHANGED; regressed bit-identical for v1/v2 by the archived-margin test.
- `cached_encode` fail-closed via a harness-owned wrapper (layering: the gate consults `bench_versions.resolve` + `cache_meta.json`, both evolve-side concepts, so realenv stays evolve-free): raises on a v3-policy root lacking the format block/perception stamp; raw-root scoring via `encode_split` stays a v1/v2-only path.
- `reencode.py`: render_canon interposition (re-folds shell_state over the raw jsonl per sequence) + meta-column emission from that same re-fold (`exit_cls` computation, delta-text recompute + F8 equality assert against `meta.delta_text`, `pre_obs_step`) + delta-text encode into `z_delta` + `cache_meta.json`/summary stamping + the perception stamp `{perception: {impl, model, content_sha}}` + the `load_perception_for_root(root)` resolver (fail-closed on stamp-less v3 roots).
- `mv_encode.py`: the SAME emission + stamping + fail-closed path (the third encode path), render_canon interposed on the multi-vector SEGMENT renders too; the mv root PROPAGATES the src root's perception stamp UNCHANGED (mv recipe recorded separately as `mv_recipe`).
- reencode.py/mv_encode.py: TRAIN-ONLY-root tolerance (the ablate raw root has no val.jsonl by F6; the split loop skips absent splits instead of raising — required to build dockerfs3-ablate-e5).
- `evolve/precompute_baselines.py` (NEW): deterministic, seed-free, cached once per root; reads the FULL val split (inner AND final); folds the SST per trajectory; canon's + renders predicted step-dicts via the root's perception; writes `sst-val.pt` + determined-mask, `wtm-val.pt`, aligned to `_data_tensors` step order; sha256s → summary.json + this prereg.
- `evolve/instruments.py` (NEW): EXIT-PROBE, DELTA-PROBE, ROLL@{2,4} — one immutable-eval-path module owns all report-only instruments; ROLL indexing pinned (real history through obs_t; imagined obs at t+1..t+k−1; scored at obs_{t+k}; prefix+horizon ≤32 asserted); probes 3 seeds; balanced accuracy computed over the classes PRESENT in the eval slice — absent classes are excluded from the mean and reported as a separate coverage column (never imputed, never scored as 0) — PINNED here per the draft's §8.3 obligation (verdict:targets M9).
- `docker_env.py`: prologue-injection seam (`DockerBox.run` accepts a collector-composed prologue fragment — fire-script + post-signal barrier — prepended inside the same `sh -c`, cd special-case branch integrated identically), per-step monotonic `dur_ms` capture, `--init` + `--label tj3-mint=<seed>` container-run flags, per-step `extra_timeout` plumbed into the exec timeout.
- `collect_docker.py`: `gen_sequence_v3`, per-seq seeding, fresh-container collect_image, container bootstrap (`mkdir -p /tmp/.tj` + `after` helper install + tini/keeper PID probe) with the bidirectional canonical-PID map applied at exec and before storage, store-time virtualization (uptime → `up <vt>` + fixed masks, mutated-path mtimes → `T+<vt>`, ps canonicalizer — all before storage; render_canon owns only the measured render-side masks), `MutGuard` (homed here; collection-side only, never eval-path), `--audit-shadow` flag, probe split, gates-as-asserts, two-arm mint, the `timing-<split>.jsonl` side-channel writer (dur_ms stripped from recorded steps; gitignored, excluded from replay byte-diffs).
- `benchmarks/scan_publish.py`: .pt inventory rows + the UD-8 exemption list (sst-val.pt / wtm-val.pt / ROLL anchors as regenerables) + ablate-root inventorying.
- `evolve/sanity.py` (+ calib_bench.py, path_battery.py, plan_eval.py): ALL direct `M.cached_encode` callers route through the harness-owned cache wrapper on v3 roots (a stamp-less v3 cache must raise there too).
- `realenv/plan_env.py`: run side resolves perception via `load_perception_for_root` + the cache gate on v3 roots; **plangoals-v3 harvest mode** (NEW input path: reads the mint jsonl + folds shell_state per trajectory; goals reference the world state AT THEIR HARVEST STEP, stratified by depth × first path component × state_scope, sha-pinned; goals referencing filesystem-mutated state carry their trajectory-prefix and replay through the self-contained replay unit — re-bootstrap + Tier-W re-seed regenerated from the per-seq seed and verified against the step-0 manifest hash FIRST, then the recorded commands; JOB-DEPENDENT goals EXCLUDED from the v3.0 harvest). Review-C precondition.
- New tracked artifacts: `benchmarks/dockerfs3-digests.json`, `benchmarks/dockerfs3-classes.json`, `benchmarks/class_measure.py` (frozen four-axis tool), `realenv/shell_state.py` (one tracker, two modes; sha-pinned, never a genome chunk), `realenv/render_canon.py` (frozen, pre-perception, version identity).
- Doc-sync in the same commits per draft §18 (CLAUDE.md module/data-root updates, evolve/CLAUDE.md fitness arms + CLI, README inventories, HF dataset card, bench-constitution amendments, evolve-insights stats).

---

## 8. Runbook deltas (`benchmarks/dockerfs3-runbook.md`; owner: prereg assembler)

Two-arm mint invocation; per-seq seeding; fresh-container lifecycle + orphan sweep (`docker ps -aq --filter label=tj3-mint=<seed> | xargs docker rm -f`); probe-container step; capability-audit step; timing side-channel handling; mint-host environment + uv.lock freeze recorded in summary; the budget table (draft §3.7) with measured 2-image pilot numbers (per-step model ~120ms/step pilot-verified before the runbook freezes; mint ≤3h under G-BUDGET: full ≈1.7h + ablate ≈0.85h); the UD-7 abort rule. Prose rules that gate become CODE asserts.

---

## 9. Tests (`tests/test_collect_v3.py`)

The fakeable exec DockerBox seam survives (no PTY seam needed); gate-at-entry tests (timeout/125/digest/mismatch); planted-fault tests per DG-10d (kill a container mid-trajectory ⇒ image aborts; stale cache ⇒ raise; planted scanner canary; planted DG-1 lexicon payload; planted DG-7 duplicate); MutGuard property tests; render_canon unit tests incl. adversarial marker strings; tracker rule tests R1–R9 on hand-built trajectories; visibility-parity test; render-inert-meta test; **archived-margin regression**: the r13 champion's recorded 0.4781 final-test and the v1 0.5848 reproduce bit-identically under the v3 harness (per-verb columns, new arms, meta reading, v1/v2 caches).

---

## 10. Review plan (three rounds, all pre-committed; convergence-ruled as in v2)

- **Review A (design/prereg)**: the amendment set (§3.1), the axis definitions, the UD decisions as ratified, the BNF ≡ G3 identity, the gate thresholds vs their planted-violation tests.
- **Review B (pilot evidence)**: SST fidelity table both directions (under- AND over-prediction); tracker-coverage per verb; blind-capture replay audit (SST actually fails on ws_observed=False); mismatch-counter triage log; DG-2 naturalism cost review (small-dir probing as a policy fingerprint); realized-rate tables; mask-list size review (>10 masked fields is itself a finding to surface); per-image skip/availability table.
- **Review C (pre-mint GO)**: DG-9 mass figures; the frozen class table; the headroom ledger (incumbent-vs-tracker per slice); plangoals-v3 harvest plan; the budget table against measured pilot ms/step; the UD-7 clock.

Convergence rule (v2 precedent): iterate until a verify-only round produces ZERO new blocker/serious findings with independent judge corroboration; the amendment log below is the findings register.

---

## 11. GO criteria (all required; GO is a dated amendment)

1. All 29 gates green at their binding `when`, each ratified by a recorded DG-10d planted-violation rejection.
2. Every §2 OPEN number frozen by dated amendment on pilot data (cpid scheme after DG-8b pricing; DG-7 constant measured on dockerfs2 first; axis-2′ threshold by the committed midpoint rule; uptime/LN-CONTRAST retention; G-SEP δ), plus the pilot-measured gate thresholds (DG-1, G-EM, G-CF floor, θ1 recalibration).
3. SST + render_canon frozen (sha-pinned) and the class table frozen by dated amendment (`dockerfs3-classes.json` sha recorded here), in the §6 freeze order — before the mint.
4. DG-9 satisfied at P5, OR the §4.3 pre-registered fallback fork ratified by dated amendment (the mint stays blocked otherwise).
5. Reviews A/B/C converged under the convergence rule.
6. UD-7 caps respected (≤6 pilot iterations, ≤3 calendar weeks); breach ⇒ NO-GO: v3 shelved, negative recorded in README.md, dockerfs2 remains the active bench.
7. The mint then proceeds per `benchmarks/dockerfs3-runbook.md` at the GO commit: one run, full + ablate arms, `--pin-digests` + `--expect-digests` required at collect() entry, gates as code asserts; any abort ⇒ delete the output dir wholesale and re-run the single one-command mint.

---



---

## Annex P0 (dated 2026-07-23) — capability audit + DG-7 reference measurement

**Image capability audit** (`benchmarks/p0/p0-images.json`; all 12 v2 images + 3 candidates, arm64 host):
- **archlinux: EXCLUDED** — no arm64 manifest, pull failed. UD-5's surviving train-only candidates: **debian:stable** (dash, no ps) and **opensuse/leap** (bash, procps). Final candidate adoption at the class-freeze amendment.
- **`ps` absent from 6/12 v2 images** — debian:stable-slim, python:3.12-slim, node:22-slim, fedora, rockylinux, httpd — including BOTH final-test images and inner-val fedora (mariadb has procps). Consequence under §9.3 coverage rules as-designed: ps-cells cannot reach final-test coverage and land report-only there; the automaton signal survives on the 6 ps-capable train images + mariadb (inner). **UD-9 DECIDED 2026-07-23 — Route B adopted (user sign-off)**: the unrecorded container bootstrap installs a sha-pinned STATIC BUSYBOX (the `after`-helper precedent) and `ps` is routed through it on ALL images (incl. those with native procps — one canonical ps output format everywhere). Consequences, binding on the implementation: (a) final-test ps coverage restored (the audit's 6-image gap is moot for coverage; the capability table stays recorded as a world fact); (b) DG-5's format-variance surface largely dissolves — its entropy/predictability gates REMAIN, its per-dialect canonicalizer shrinks to the one busybox format; (c) the busybox binary (per-arch) joins version identity next to the `after` helper: sha-pinned, excluded from mining pools, denylisted from mutation, known to the DG-3 scanner, expected echo/const under axis-1 if ever observed; (d) native `ps` binaries stay untouched on-image (world honesty: `command -v ps` may resolve either; the POLICY always invokes the vendored path `/usr/local/bin/tj3-ps` via the ps arm's template — the template, not PATH resolution, is frozen). Install-as-observable-action (Route C) deferred to v4, recorded in BACKLOG.
- **Substrate viability confirmed on all 14 pullable images**: fractional sleep, fifo, and the full mutation verb set present everywhere; shell dialects busybox-ash (4) / dash (7) / bash (3); fresh digests captured for the dockerfs3-digests.json re-pin (final pin at mint entry).

**DG-7 reference constant** (`benchmarks/p0/dg7_reference.json`, script `benchmarks/p0/dg7_measure.py`; measured on dockerfs2 BEFORE v3 work per §2, protocol per §11.2's union definition):
- Overall (all val obs vs all 115,073 train obs renders): exact **0.2951**, union (cos>0.995 or exact) **0.3707**.
- Content slice (v2 mode rule): exact **0.1955**, union **0.2889**. Inner 0.2530/0.3204; final 0.3370/0.4208.
- Per-verb (exact/union): uname .80/.80, cd .66/.68, grep .34/.37, find .31/.37, cat .25/.34, ls .17/.30, stat .15/.20, head .13/.20, tail .13/.16.
- Five ambiguities flagged for the freezing amendment (metric choice exact-vs-union; val-side slice matching; obs-renders-only; rate-not-distance transfer; e5 256-token truncation) — the constant freezes by dated amendment with the metric matched to the v3 LHS, per §2.

## Amendment log

Protocol: amendments are dated, appended below, and never rewrite the body above. Each pilot-frozen number lands as a dated amendment quoting the measured pilot evidence. Post-freeze edits to sha-pinned eval-path artifacts (shell_state.py, render_canon.py, dockerfs3-classes.json) require a dated amendment + re-baseline. Constitutional amendments (§3.1) land in bench-constitution.md in the same commit. The class freeze, the fallback-fork ratification (if triggered), and GO/NO-GO are all amendments here.

*(no amendments yet)*
