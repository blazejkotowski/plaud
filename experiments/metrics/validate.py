"""Score every reconstruction we have listening verdicts for with the aligned measures (aligned.py), and count how
many of the verdicts each measure (and each simple combination) reproduces.

Verdicts:
  datasize  the 21 pairs judged on the data-size bench (one 16 s passage per set; experiments/datasize/results_whole)
  styles    percussion, 2026-10-22: the range-loss takes are worse than the baseline (tails cut), compared as the
            mean over the 8 passages (experiments/styles/results/percussion)
Caches the scores in experiments/metrics/aligned_scores.json.

  python experiments/metrics/validate.py
"""

import glob
import itertools
import json
import os
from multiprocessing import Pool

import numpy as np

from aligned import scores

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DS = os.path.join(ROOT, "experiments/datasize/results_whole")
ST = os.path.join(ROOT, "experiments/styles/results/percussion")
CACHE = os.path.join(HERE, "aligned_scores.json")
MEASURES = ["env20", "env100", "hf20", "dyncorr", "attack", "tail_cut", "tail_long", "floor"]


def _job(r):
  return os.path.relpath(r, ROOT), scores(r.replace("_recon.wav", "_orig.wav"), r)


def datasize_pairs():
  k = lambda s, t: f"{s}_full_e160" if t == "full" else f"ds_{s}_{t}"
  H, C, D = "harbour", "percussion", "drone_sampled"
  p = [(k(H, "s128_e160"), k(H, w)) for w in ["full", "s16_e160", "s16_e20", "s128_e20"]]
  p += [(k(H, b), k(H, "full")) for b in ["s16_e160", "s16_e20", "s128_e20"]]
  p += [(k(C, "s16_e160"), k(C, "s128_e160")), (k(C, "s128_e160"), k(C, "full")), (k(C, "s16_e160"), k(C, "full"))]
  p += [(k(C, b), k(C, "full")) for b in ["s16_e20", "s128_e20"]]
  p += [(k(D, "s16_e20"), k(D, w)) for w in ["s128_e160", "s16_e160", "s128_e20", "full"]]
  p += [(k(D, "s128_e160"), k(D, w)) for w in ["s16_e160", "s128_e20", "full"]]
  p += [(k(D, b), k(D, "full")) for b in ["s16_e160", "s128_e20"]]
  return [(f"experiments/datasize/results_whole/{a}_recon.wav", f"experiments/datasize/results_whole/{b}_recon.wav") for a, b in p]


def main():
  cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
  todo = [r for r in sorted(glob.glob(os.path.join(DS, "*_recon.wav")) + glob.glob(os.path.join(ST, "*", "*", "p*_recon.wav")))
          if os.path.relpath(r, ROOT) not in cache]
  with Pool(16) as pool:
    for key, s in pool.imap_unordered(_job, todo):
      cache[key] = s
  json.dump(cache, open(CACHE, "w"), indent=1)

  # a take's value: datasize = its one passage; styles = mean over passages
  def take(t):
    if t.endswith(".wav"):
      return cache[t]
    ks = [k for k in cache if k.startswith(f"experiments/styles/results/percussion/{t}/")]
    return {m: float(np.mean([cache[k][m] for k in ks])) for m in MEASURES}
  nd = len(datasize_pairs())
  pairs = datasize_pairs() + [("base/e160", "range/e80"), ("base/e160", "range/e160")]
  vals = {t: take(t) for t in {x for p in pairs for x in p}}

  # z-score each measure within each group of takes compared together (a set, or the styles bench)
  group = lambda t: t.split("/")[-1].split("_full")[0].replace("ds_", "").rsplit("_s1", 1)[0] if t.endswith(".wav") else "styles"
  z = {t: {} for t in vals}
  for g in {group(t) for t in vals}:
    ts = [t for t in vals if group(t) == g]
    for m in MEASURES:
      v = np.array([vals[t][m] for t in ts])
      for t in ts:
        z[t][m] = (vals[t][m] - v.mean()) / (v.std() + 1e-9)

  def agree(f):
    return [f(a) < f(b) for a, b in pairs]
  rows = []
  for r in (1, 2, 3):
    for combo in itertools.combinations(MEASURES, r):
      ok = agree(lambda t: sum(z[t][m] for m in combo))
      rows.append((sum(ok), sum(ok[:nd]), sum(ok[nd:]), "+".join(combo)))
  rows.sort(reverse=True)
  print(f"{len(pairs)} verdicts ({nd} data-size pairs, 2 style pairs)")
  print("single measures:")
  for r in [r for r in rows if "+" not in r[3]]:
    print(f"  {r[3]:28s} {r[0]:2d}/{len(pairs)}  (datasize {r[1]}/{nd}, styles {r[2]}/2)")
  print("best combinations (z-sum, equal weights):")
  for r in rows[:15]:
    print(f"  {r[3]:28s} {r[0]:2d}/{len(pairs)}  (datasize {r[1]}/{nd}, styles {r[2]}/2)")
  print("\npercussion bench, mean over 8 passages:")
  takes = sorted({k.split("/")[4] + "/" + k.split("/")[5] for k in cache if k.startswith("experiments/styles/")})
  print(f"  {'take':16s}" + "".join(f"{m:>10s}" for m in MEASURES))
  for t in takes:
    v = take(t)
    print(f"  {t:16s}" + "".join(f"{v[m]:10.2f}" for m in MEASURES))


if __name__ == "__main__":
  main()
