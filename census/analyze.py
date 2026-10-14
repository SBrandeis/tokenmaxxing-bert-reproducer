#!/usr/bin/env python3
"""Census job log → results/<job-id>/: models.json, chains.tsv (downloads per normalizer chain),
unicode.tsv (only chains with BertNormalizer or a Unicode normal form). Usage: analyze.py <job-id>
"""
import base64
import gzip
import json
import pathlib
import re
import subprocess
import sys
from collections import defaultdict

UNICODE = re.compile(r"\bBertNormalizer\b|\bNFK?[CD]\b")


def main(job):
    out = pathlib.Path(__file__).parent / "results" / job
    out.mkdir(parents=True, exist_ok=True)
    log = subprocess.run(["hf", "jobs", "logs", job], capture_output=True, text=True, check=True).stdout
    (out / "log.txt").write_text("\n".join(l for l in log.splitlines() if not l.startswith("RESULT")) + "\n")
    parts, n = {}, 0
    for l in log.splitlines():
        if l.startswith("RESULT\t"):
            _, i, n, chunk = l.split("\t", 3)
            parts[int(i)] = chunk
    if not parts or len(parts) != int(n):
        raise SystemExit(f"got {len(parts)} of {n} result chunks")
    rows = json.loads(gzip.decompress(base64.b64decode("".join(parts[i] for i in range(len(parts))))))
    rows.sort(key=lambda r: -r["downloads"])
    (out / "models.json").write_text(json.dumps(rows, indent=0) + "\n")
    groups = defaultdict(lambda: {"models": 0, "downloads": 0, "top": []})
    for r in rows:
        key = r["chain"] if r["status"] == "ok" else f"({r['status']})"
        g = groups[key]
        g["models"] += 1
        g["downloads"] += r["downloads"]
        if len(g["top"]) < 5:
            g["top"].append(r["id"])
    ordered = sorted(groups.items(), key=lambda kv: -kv[1]["downloads"])
    for name, keep in (("chains.tsv", lambda k: True), ("unicode.tsv", lambda k: bool(UNICODE.search(k)))):
        lines = ["chain\tmodels\tdownloads_30d\ttop_models"]
        lines += [f"{k}\t{g['models']}\t{g['downloads']}\t{', '.join(g['top'])}" for k, g in ordered if keep(k)]
        (out / name).write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} models; {sum(1 for k, _ in ordered if UNICODE.search(k))} Unicode/BERT chains")
    for k, g in [kv for kv in ordered if UNICODE.search(kv[0])][:15]:
        print(f"{g['downloads']:>13,}  {g['models']:>5}  {k}")


if __name__ == "__main__":
    main(sys.argv[1])
