"""Mandatory pre-publication content scan (bench-constitution §9): run on every root before
any HF upload; publication proceeds only on a clean report. Patterns cover credential/secret
material a public dataset must never carry; matches are reported with context, never fixed
silently.

  uv run python -m benchmarks.scan_publish --root data/dockerfs2
"""

import argparse
import json
import pathlib
import re
import sys

# NOTE: public CERTIFICATE blocks (CA trust stores like ca-certificates.crt) are
# deliberately NOT flagged — certificates are public documents (v1 published the same
# content); only PRIVATE KEY material blocks publication.
PATTERNS = [
    ("private-key-block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP |ENCRYPTED )?PRIVATE KEY")),
    ("crypt-hash", re.compile(r"\$(?:1|2[aby]|5|6|7|y|gy)\$[^:\s]{3,}\$[./A-Za-z0-9]{8,}")),
    ("crypt-rounds", re.compile(r"\$(?:5|6)\$rounds=\d+\$")),
    ("aws-akid", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("openai-key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("hf-token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("ssh-pubkey-material", re.compile(r"\bssh-(?:rsa|ed25519|dss) AAAA[A-Za-z0-9+/=]{40,}")),
    ("secret-assignment", re.compile(r"(?i)\b(?:password|passwd|secret|api[_-]?key|token)\s*[=:]\s*['\"][^'\"\s]{8,}['\"]")),
    ("host-username-path", re.compile(r"/Users/[a-z][a-z0-9_-]{2,}/")),
]


def scan_file(path, findings, max_ctx=120):
    for ln, line in enumerate(open(path, errors="replace"), 1):
        for name, rx in PATTERNS:
            m = rx.search(line)
            if m:
                a = max(0, m.start() - 40)
                findings.append({"file": str(path), "line": ln, "pattern": name,
                                 "context": line[a:a + max_ctx].strip()})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    root = pathlib.Path(args.root)
    if not root.is_dir():
        print(f"SCAN FAIL-CLOSED: root {root} does not exist", file=sys.stderr)
        sys.exit(2)
    findings = []
    files = sorted(root.rglob("*.jsonl")) + sorted(root.rglob("*.json"))
    if not any(f.name == "train.jsonl" for f in files):
        print(f"SCAN FAIL-CLOSED: no train.jsonl under {root} — scanned nothing publishable "
              f"({len(files)} files matched)", file=sys.stderr)
        sys.exit(2)
    for f in files:
        scan_file(f, findings)
    print(f"scanned {len(files)} files")
    report = {"root": str(root), "files_scanned": [str(f) for f in files],
              "n_findings": len(findings), "findings": findings[:200]}
    print(json.dumps({k: report[k] for k in ("root", "n_findings")}, indent=1))
    for x in findings[:20]:
        print(f"  [{x['pattern']}] {x['file']}:{x['line']}: {x['context'][:100]}")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(report, indent=1))
    sys.exit(1 if findings else 0)


if __name__ == "__main__":
    main()
