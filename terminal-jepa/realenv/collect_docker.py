"""Collect real shell trajectories from Docker images (Phase R2, 2026-07-16 redesign).

Each trajectory is a SEQUENCE that explores a real Linux filesystem: identify the system
(uname + cat a config file), then navigate and inspect (cd / ls / cat) with option and
target variety over the image's real paths. The world model must predict later
observations from the accumulated history. Split by held-out IMAGE (unseen system types)
— the fair "does it generalize to new places/systems" test, NOT the unreasonable
"infer an unseen tool" test.

Tools (initial set, per user): uname, cat (system config files), ls, cd.

Usage:
  python3 -m realenv.collect_docker --out data/dockerfs --seqs-per-image 300 --seq-len 16
"""

import argparse
import concurrent.futures as cf
import json
import pathlib
import random

from realenv.docker_env import DockerBox, image_present, pull

TRAIN_IMAGES = ["alpine:latest", "ubuntu:latest", "debian:stable-slim", "python:3.12-slim"]
VAL_IMAGES = ["fedora:latest", "redis:alpine", "nginx:alpine"]  # held-out system types

CONFIG_FILES = ["/etc/os-release", "/etc/hostname", "/etc/issue", "/proc/version",
                "/etc/passwd", "/etc/group", "/etc/hosts", "/etc/resolv.conf",
                "/etc/shells", "/etc/profile"]
UNAME_OPTS = ["-a", "-s", "-m", "-r", "-n", "-o", "-v", "-sm", "-sr", ""]
# options common to busybox and GNU ls (kept safe across images)
LS_OPTS = ["", "-l", "-a", "-la", "-R", "-1", "-lh", "-lt", "-lS", "-ld", "-i", "-lr", "-ln"]


def gen_sequence(box, dirs, files, rng, length):
    steps = []

    def do(cmd):
        steps.append(box.run(cmd))

    do("uname " + rng.choice(UNAME_OPTS))
    do("cat " + rng.choice(CONFIG_FILES))
    for _ in range(max(0, length - 2)):
        act = rng.choices(["cd", "ls", "cat", "config"], weights=[0.32, 0.4, 0.2, 0.08])[0]
        if act == "cd":
            tgt = rng.choice(dirs + ["..", ".", "..", "/", box.cwd] if dirs else ["..", "/", "."])
            do(f"cd {tgt}")
        elif act == "ls":
            opt = rng.choice(LS_OPTS)
            tgt = (" " + rng.choice(dirs)) if (dirs and rng.random() < 0.45) else ""
            do(f"ls {opt}{tgt}".strip())
        elif act == "cat" and files:
            do("cat " + rng.choice(files))
        else:
            do("cat " + rng.choice(CONFIG_FILES))
    return steps


def collect_image(image, n_seqs, seq_len, seed):
    if not image_present(image) and not pull(image):
        return image, None, f"could not pull {image}"
    try:
        box = DockerBox(image)
        sysid = box.system_id()
        dirs, files = box.enumerate()
        if not dirs:
            dirs = ["/etc", "/var", "/usr", "/"]
        rng = random.Random(f"dockerfs:{seed}:{image}")
        seqs = []
        for i in range(n_seqs):
            box.cwd = "/"  # each sequence starts fresh at root
            ln = rng.randint(max(4, seq_len - 4), seq_len + 4)
            seqs.append({"image": image, "system_id": sysid,
                         "steps": [{"cmd": s["cmd"], "output": s["output"],
                                    "exit": s["exit"], "cwd": s["cwd"]}
                                   for s in gen_sequence(box, dirs, files, rng, ln)]})
        box.close()
        return image, seqs, f"{len(dirs)} dirs / {len(files)} files"
    except Exception as e:  # noqa: BLE001
        return image, None, f"error: {e}"


def collect(out_dir, train_imgs, val_imgs, n_seqs, seq_len, seed, workers):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def run_split(images, path, split):
        n_steps = 0
        with open(path, "w") as fh, cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(collect_image, im, n_seqs, seq_len, seed): im for im in images}
            for fut in cf.as_completed(futs):
                image, seqs, info = fut.result()
                if seqs is None:
                    print(f"  [{split}] SKIP {image}: {info}", flush=True)
                    continue
                for s in seqs:
                    fh.write(json.dumps(s) + "\n")
                    n_steps += len(s["steps"])
                print(f"  [{split}] {image}: {len(seqs)} seqs ({info})", flush=True)
        return n_steps

    tr_steps = run_split(train_imgs, out / "train.jsonl", "train")
    va_steps = run_split(val_imgs, out / "val.jsonl", "val")
    summary = {"seed": seed, "seqs_per_image": n_seqs, "seq_len": seq_len,
               "train_images": train_imgs, "val_images_heldout": val_imgs,
               "train_steps": tr_steps, "val_steps": va_steps}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dockerfs")
    ap.add_argument("--seqs-per-image", type=int, default=300)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--train-images", default=",".join(TRAIN_IMAGES))
    ap.add_argument("--val-images", default=",".join(VAL_IMAGES))
    args = ap.parse_args(argv)
    summary = collect(args.out, args.train_images.split(","), args.val_images.split(","),
                      args.seqs_per_image, args.seq_len, args.seed, args.workers)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
