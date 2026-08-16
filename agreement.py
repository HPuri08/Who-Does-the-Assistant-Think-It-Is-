"""
Compute human-vs-LLM-judge agreement (percent agreement + Cohen's kappa)
from the filled-in human coding CSV.

Usage:
    python agreement.py --human data/human_sample.csv --coded data/coded.jsonl
"""
import argparse
import csv
import json
from collections import Counter


def kappa(pairs):
    """Cohen's kappa for a list of (rater1, rater2) label pairs."""
    n = len(pairs)
    if n == 0:
        return float("nan"), float("nan")
    po = sum(1 for a, b in pairs if a == b) / n
    c1, c2 = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    labels = set(c1) | set(c2)
    pe = sum((c1[l] / n) * (c2[l] / n) for l in labels)
    k = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return po, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True)
    ap.add_argument("--coded", default="data/coded.jsonl")
    args = ap.parse_args()

    judge = {}
    for line in open(args.coded):
        r = json.loads(line)
        if not r.get("judge") or "error" in r["judge"]:
            continue
        key = (r["model"], r["condition"], r["item_id"], r["response"][:80])
        judge[key] = r["judge"]

    id_pairs, st_pairs = [], []
    with open(args.human) as f:
        for row in csv.DictReader(f):
            key = (row["model"], row["condition"], row["item_id"],
                   row["response"][:80])
            j = judge.get(key)
            if not j:
                continue
            if row["study"] == "identity" and row["HUMAN_CODE"].strip():
                id_pairs.append((row["HUMAN_CODE"].strip().upper(),
                                 j.get("code", "")))
            if row["study"] == "preference" and row["HUMAN_STANCE"].strip():
                st_pairs.append((row["HUMAN_STANCE"].strip().upper(),
                                 j.get("stance", "")))

    for name, pairs in [("identity codes", id_pairs),
                        ("preference stances", st_pairs)]:
        po, k = kappa(pairs)
        print(f"{name}: n={len(pairs)}, agreement={po:.2f}, kappa={k:.2f}")


if __name__ == "__main__":
    main()
