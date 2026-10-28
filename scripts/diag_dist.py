"""Find rows where a searcher found a proof SHORTER than the dataset's 'dist' label."""
import json
import sys
from collections import Counter

run = sys.argv[1] if len(sys.argv) > 1 else "runs/E1-test_depth"
with open("data/isobench-v0/test_depth.jsonl", encoding="utf-8") as f:
    data = {r["id"]: r for r in map(json.loads, f)}
with open(f"{run}/results.jsonl", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

shorter = [r for r in rows if r["solved"] and r["dist"] is not None and r["proof_len"] < r["dist"]]
print("rows with proof_len < dist, per method:", dict(Counter(r["method"] for r in shorter)))
print("bfs-bidir rows with proof_len != dist:",
      sum(1 for r in rows if r["method"] == "bfs-bidir" and r["solved"] and r["dist"] is not None
          and r["proof_len"] != r["dist"]))
for r in [r for r in shorter if r["method"] == "bfs-fwd"][:3]:
    d = data[r["id"]]
    print("\n---", r["id"], d["theory"], "dist", r["dist"], "fwd proof_len", r["proof_len"])
    print("lhs:", d["lhs"])
    print("rhs:", d["rhs"])
    print("fwd proof:", json.dumps(r["proof"]))
    print("dataset proof:", json.dumps(d["proof"]))