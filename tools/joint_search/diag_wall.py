#!/usr/bin/env python3
"""diag_wall.py — ILP-objective vs emitted-price decomposition (wall era).

For a problem, prints the ILP's chosen cost components (y/z/h/r/w vars with
wall and flop costs) and the emitted program's per-op wall prices, so the
self-check delta is attributable. Committed diagnostic per LOOP.md.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import proto  # noqa: E402
import wallmodel  # noqa: E402


def main():
    prob_path, sizes_s = sys.argv[1], sys.argv[2]
    ext = proto.parse_sizes(sizes_s)
    problem = json.load(open(prob_path))
    proto.SYMS = problem["symmetries"] if "sym" in sys.argv[3:] else {}
    stem = os.path.basename(prob_path).replace(".json", "")
    base_path = os.path.join(ROOT, "benches", "candidates",
                             f"{stem}_gristmill.json")
    if os.path.exists(base_path):
        proto.FLOP_CAP = 1.05 * proto.price_candidate(
            json.load(open(base_path)), ext)
    pack = proto.solve_problem(problem, ext)
    prog = proto.emit_global(pack, ext)
    rates = proto.wall_rates()

    print(f"ILP objective: {pack['obj']:.6f} s")
    emitted = wallmodel.price_candidate_wall(prog, ext, rates)
    print(f"emitted wall (clean_assumed): {emitted:.6f} s   "
          f"delta {emitted - pack['obj']:+.6f} s")
    emitted_aw = wallmodel.price_candidate_wall(prog, ext, rates,
                                                layout="as_written")
    print(f"emitted wall (as_written):    {emitted_aw:.6f} s")

    print("\ntop emitted ops (clean_assumed):")
    rows = []
    for i, v in enumerate(prog["values"]):
        if "Einsum" in v:
            code = v["Einsum"]["code"]
            dom = {int(l): d for l, d in code["domains"].items()}
            p = wallmodel.einsum_wall(code["inputs"], code["output"],
                                      dom, ext, rates)
            rows.append((p, i, "E", len(code["inputs"]),
                         len(code["output"])))
        else:
            code = v["LinComb"]["code"]
            dom = {int(l): d for l, d in code["domains"].items()}
            p = wallmodel.lincomb_wall(code["inputs"], code["output"],
                                       dom, ext, rates)
            rows.append((p, i, "L", len(v["LinComb"]["args"]),
                         len(code["output"])))
    rows.sort(reverse=True)
    for p, i, k, na, orank in rows[:12]:
        print(f"  op{i:<4} {k} args={na} out_rank={orank}  {p*1e3:9.3f} ms")


if __name__ == "__main__":
    main()
