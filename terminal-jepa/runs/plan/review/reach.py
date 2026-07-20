"""Reachability + success-predicate audit for plan-probe stage2.
For each goal: walk ancestors from /, at each ancestor run the REAL `ls -la`,
parse with plan_env.parse_child_dirs, check the true next path component is a candidate.
Also: cd the exact goal path and check box.cwd == goal dir (the success predicate).
Saves candidate sets per ancestor for the M1 geometry analysis."""
import json, pathlib, sys
sys.path.insert(0, ".")
from realenv.plan_env import parse_child_dirs, depth_of
from realenv.docker_env import DockerBox

IMAGES = {"fedora:latest": "fedora_latest", "mariadb:latest": "mariadb_latest",
          "rockylinux:9": "rockylinux_9", "httpd:2.4": "httpd_2.4"}
out = {"images": {}}
for image, tag in IMAGES.items():
    goals = [json.loads(l) for l in open(f"data/plangoals-v1/goals-{tag}.jsonl")]
    box = DockerBox(image)
    ls_cache = {}
    def ls_children(cwd):
        if cwd not in ls_cache:
            box.run(f"cd {cwd}")
            r = box.run("ls -la")
            kids = parse_child_dirs(r["output"], box.cwd)
            box.run("cd /")
            ls_cache[cwd] = [k.split("/")[-1] for k in kids]
        return ls_cache[cwd]
    rows = []
    for g in goals:
        comps = [c for c in g["dir"].split("/") if c]
        cwd = "/"
        reachable = True
        blocked_at = None
        chain = []
        for i, c in enumerate(comps):
            kids = ls_children(cwd)
            chain.append({"cwd": cwd, "n_cands": len(kids), "true_next": c,
                          "visible": c in kids, "cands": kids})
            if c not in kids:
                reachable = False
                blocked_at = cwd
            cwd = (cwd.rstrip("/") + "/" + c) if cwd != "/" else "/" + c
        # success predicate check: cd exact path, compare box.cwd
        box.run("cd /")
        r = box.run(f"cd {g['dir']}")
        cwd_match = (box.cwd == g["dir"]) and r["exit"] == 0
        box.run("cd /")
        rows.append({"goal": g["dir"], "depth": g["depth"], "reachable": reachable,
                     "blocked_at": blocked_at, "cwd_match": cwd_match, "chain": chain})
    box.close()
    n = len(rows)
    reach = sum(r["reachable"] for r in rows)
    match = sum(r["cwd_match"] for r in rows)
    out["images"][image] = {"n": n, "reachable": reach, "cwd_match": match, "rows": rows}
    print(f"{image}: {reach}/{n} reachable via parse_child_dirs, {match}/{n} success-predicate ok", flush=True)
    for r in rows:
        if not r["reachable"] or not r["cwd_match"]:
            print("  PROBLEM:", r["goal"], "blocked_at", r["blocked_at"], "cwd_match", r["cwd_match"], flush=True)
pathlib.Path("/private/tmp/claude-501/-Users-fanzhu-PyCharmProjects-jepa/d198d7af-d892-4db5-b49e-72d82a05b137/scratchpad/review/reach.json").write_text(json.dumps(out))
print("done")
