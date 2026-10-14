#!/usr/bin/env python3
"""start_jobs.sh logs → tables: python3 analyze.py results/

Per results/<job-id>/log.txt: machine.json, throughput.tsv, sizes.tsv, ir.tsv, versions.tsv,
stacks.<model>.<corpus>.<arm>.folded; plus results/summary.md. Speedups are medians of per-round
ratios; round 0 (warm-up) is dropped.
"""
import base64
import gzip
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def parse(log):
    machine, in_machine = [], False
    tp, sz, ir, notes = [], [], [], []
    stacks, irbase, ver = {}, {}, []
    for line in log.read_text().splitlines():
        if line == "== machine.json begin":
            in_machine = True
        elif line == "== machine.json end":
            in_machine = False
        elif in_machine:
            machine.append(line)
        elif line.startswith("== "):
            notes.append(line[3:])
        else:
            f = line.split("\t")
            if f[0] == "tp" and len(f) == 9:
                tp.append(dict(model=f[1], corpus=f[2], arm=f[3], round=int(f[4]), mbs=float(f[5]), hash=f[6]))
            elif f[0] == "sz" and len(f) == 11:
                sz.append(dict(arm=f[1], round=int(f[2]), name=f[3], corpus=f[4], size=int(f[5]),
                               bytes=int(f[7]), ns=int(f[8]), hash=f[9]))
            elif f[0] == "ir" and len(f) == 6:
                ir.append(dict(model=f[1], corpus=f[2], arm=f[3], total=int(f[4] or 0), norm=int(f[5] or 0)))
            elif f[0] == "ver" and len(f) == 9:
                ver.append(dict(model=f[1], corpus=f[2], arm=f[3], round=int(f[4]), mbs=float(f[5]), hash=f[6]))
            elif f[0] == "stacks" and len(f) == 5 and f[4]:
                stacks[tuple(f[1:4])] = gzip.decompress(base64.b64decode(f[4])).decode()
            elif f[0] == "irbase" and len(f) == 5:
                irbase[(f[1], f[2])] = int(f[3] or 0)
    machine = json.loads("\n".join(machine)) if machine else {}
    for r in ir:
        r["load"] = irbase.get((r["model"], r["arm"]), 0)
    return machine, tp, sz, ir, notes, stacks, ver


def cpu_model(machine):
    for row in machine.get("lscpu", {}).get("lscpu", []):
        if row["field"] == "Model name:":
            return row["data"]
    return "?"


def throughput(tp):
    cells = defaultdict(lambda: defaultdict(dict))
    for r in tp:
        cells[(r["model"], r["corpus"])][r["round"]][r["arm"]] = r
    rows = []
    for (model, corpus), rounds in cells.items():
        timed = [v for k, v in rounds.items() if k > 0 and {"main", "branch"} <= v.keys()]
        if not timed:
            continue
        ratios = [v["branch"]["mbs"] / v["main"]["mbs"] for v in timed]
        hashes = {v[a]["hash"] for v in rounds.values() for a in v}
        rows.append(dict(model=model, corpus=corpus,
                         main=statistics.median(v["main"]["mbs"] for v in timed),
                         branch=statistics.median(v["branch"]["mbs"] for v in timed),
                         speedup=statistics.median(ratios), lo=min(ratios), hi=max(ratios),
                         same=len(hashes) == 1, n=len(timed)))
    return rows


VERSION_ORDER = ["v0.23.2", "main", "branch"]


def versions(ver):
    cells = defaultdict(lambda: defaultdict(dict))
    for r in ver:
        cells[(r["model"], r["corpus"])][r["round"]][r["arm"]] = r
    rows = []
    for (model, corpus), rounds in cells.items():
        timed = [v for k, v in rounds.items() if k > 0 and set(VERSION_ORDER) <= v.keys()]
        hashes = {x["hash"] for v in rounds.values() for x in v.values()}
        for arm in VERSION_ORDER:
            if not timed:
                continue
            ratios = [v[arm]["mbs"] / v[VERSION_ORDER[0]]["mbs"] for v in timed]
            rows.append(dict(model=model, corpus=corpus, arm=arm, mbs=statistics.median(v[arm]["mbs"] for v in timed),
                             x=statistics.median(ratios), lo=min(ratios), hi=max(ratios), same=len(hashes) == 1,
                             n=len(timed)))
    return rows


def sizes(sz):
    cells = defaultdict(lambda: defaultdict(list))
    hashes = defaultdict(dict)
    for r in sz:
        key = (r["name"], r["corpus"], r["size"])
        cells[key][r["arm"]].append(r["ns"])
        hashes[key][r["arm"]] = r["hash"]
    rows = []
    for key, arms in cells.items():
        if not {"main", "branch"} <= arms.keys():
            continue
        main, branch = statistics.median(arms["main"]), statistics.median(arms["branch"])
        aa = statistics.median(arms["branch2"]) / branch if arms.get("branch2") else float("nan")
        rows.append(dict(name=key[0], corpus=key[1], size=key[2], main=main, branch=branch,
                         speedup=main / branch, aa=aa, same=hashes[key]["main"] == hashes[key]["branch"]))
    return rows


def write_tsv(path, header, rows):
    if not rows:
        path.unlink(missing_ok=True)
        return
    path.write_text("\t".join(header) + "\n" + "".join("\t".join(map(str, r)) + "\n" for r in rows))


def main(root):
    summary = ["# Results", "", "Generated by `analyze.py` from `results/*/log.txt`. "
               "Speedup = branch MB/s ÷ main MB/s per round; median [min–max] over timed rounds.", ""]
    for log in sorted(Path(root).glob("*/log.txt")):
        out = log.parent
        machine, tp, sz, ir, notes, stacks, ver = parse(log)
        if machine:
            (out / "machine.json").write_text(json.dumps(machine, indent=2) + "\n")
        t, s = throughput(tp), sizes(sz)
        write_tsv(out / "throughput.tsv", ["model", "corpus", "main_mbs", "branch_mbs", "speedup", "ratio_min",
                                           "ratio_max", "same_tokens", "rounds"],
                  [(r["model"], r["corpus"], f"{r['main']:.2f}", f"{r['branch']:.2f}", f"{r['speedup']:.3f}",
                    f"{r['lo']:.3f}", f"{r['hi']:.3f}", r["same"], r["n"]) for r in t])
        write_tsv(out / "sizes.tsv", ["normalizer", "corpus", "size", "main_ns", "branch_ns", "speedup", "aa_ratio",
                                      "same_output"],
                  [(r["name"], r["corpus"], r["size"], r["main"], r["branch"], f"{r['speedup']:.3f}",
                    f"{r['aa']:.3f}", r["same"]) for r in s])
        write_tsv(out / "ir.tsv", ["model", "corpus", "arm", "instructions", "in_normalize_all", "load_only"],
                  [(r["model"], r["corpus"], r["arm"], r["total"], r["norm"], r["load"]) for r in ir])
        v = versions(ver)
        if v:
            write_tsv(out / "versions.tsv", ["model", "corpus", "arm", "mbs", "x_vs_first", "ratio_min", "ratio_max",
                                             "same_tokens", "rounds"],
                      [(r["model"], r["corpus"], r["arm"], f"{r['mbs']:.2f}", f"{r['x']:.3f}", f"{r['lo']:.3f}",
                        f"{r['hi']:.3f}", r["same"], r["n"]) for r in v])
            summary += ["", "| model | corpus | release | MB/s | × 0.23.2 [min–max] | same tokens |", "|---|---|---|---:|---|---|"]
            summary += [f"| {r['model']} | {r['corpus']} | {r['arm']} | {r['mbs']:.1f} | {r['x']:.2f}× [{r['lo']:.2f}–{r['hi']:.2f}] | "
                        f"{'yes' if r['same'] else '**NO**'} |" for r in v]
        for (m, c, arm), text in stacks.items():
            (out / f"stacks.{m}.{c}.{arm}.folded").write_text(text)

        summary += [f"## {out.name}", "",
                    f"- flavor `{machine.get('flavor', '?')}`, **{cpu_model(machine)}**, kernel {machine.get('kernel', '?')}, "
                    f"SIMD: {machine.get('simd_flags', '?').strip()}, pinned: {machine.get('pinned_to', '?')}"]
        summary += [f"- {n}" for n in notes if n.startswith(("commit", "toolchain", "steal", "done"))]
        for (m, c, arm), text in sorted(stacks.items()):
            head = next((l for l in text.splitlines() if l.startswith("#")), "# ?")
            n = sum(int(l.rsplit(" ", 1)[1]) for l in text.splitlines() if l and not l.startswith("#"))
            summary.append(f"- stacks {m} {c} {arm}: {n} samples, {head[2:]}")
        if t:
            summary += ["", "| model | corpus | main MB/s | branch MB/s | speedup [min–max] | same tokens |",
                        "|---|---|---:|---:|---|---|"]
            summary += [f"| {r['model']} | {r['corpus']} | {r['main']:.1f} | {r['branch']:.1f} | "
                        f"{r['speedup']:.2f}× [{r['lo']:.2f}–{r['hi']:.2f}] | {'yes' if r['same'] else '**NO**'} |"
                        for r in t]
        if s:
            # A loss counts only beyond the A/A control's noise.
            noise = statistics.median(abs(r["aa"] - 1) for r in s if r["aa"] == r["aa"])
            worst = max(abs(r["aa"] - 1) for r in s if r["aa"] == r["aa"])
            losses = sorted((r for r in s if r["speedup"] < 1 - max(0.03, 2 * noise)), key=lambda r: r["speedup"])
            summary += ["", f"Size sweep: {len(s)} cells, A/A control median |1−ratio| {noise:.1%}, worst {worst:.1%}. "
                        f"Output differs between main and branch in {sum(not r['same'] for r in s)} cells.",
                        "", "Losses beyond noise:", ""]
            summary += [f"- {r['name']} / {r['corpus']} / {r['size']} B: {r['speedup']:.2f}×" for r in losses] or ["- none"]
        if ir:
            by = {(r["model"], r["corpus"], r["arm"]): r for r in ir}
            summary += ["", "| model | corpus | instructions main → branch | normalize_all share main → branch |",
                        "|---|---|---|---|"]
            enc = lambda x: x["total"] - x["load"]
            for (m, c, a), r in by.items():
                if a == "main" and (m, c, "branch") in by:
                    b = by[(m, c, "branch")]
                    share = lambda x: f"{x['norm'] / enc(x):.0%}" if enc(x) > 0 else "?"
                    tag = "" if r["load"] else " (incl. load)"
                    summary += [f"| {m} | {c} | {enc(r) / 1e9:.2f} G → {enc(b) / 1e9:.2f} G "
                                f"({enc(r) / max(enc(b), 1):.2f}×){tag} | {share(r)} → {share(b)}{tag} |"]
        summary.append("")
    (Path(root) / "summary.md").write_text("\n".join(summary))
    print("\n".join(summary))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
