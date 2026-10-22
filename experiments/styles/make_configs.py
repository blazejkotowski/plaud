"""Configs for the style experiments: configs/research_styles/<set>_<exp>.yaml, each a variant of the
set's full factory model (the *_nosilence config, whose trained synth is the E0 baseline):

  oracle    E1  one style embedding per 16 s passage (upper limit of conditioning)
  k12       E2  12 style clusters of 16 s passages (the version a style pad would use)
  lat4      E3  4 latents instead of 2, no styles (capacity within a passage)
  rangeadv  E4  dynamic-range loss + the adversarial loss, no styles (loss-side fix)
  range     E5  dynamic-range loss alone (separates the two halves of E4)
  adv       E6  adversarial loss alone (distribution matching, no range loss)
  env       E7  time-aligned band-envelope loss (EnvelopeLoss) next to the spectral loss
  flat250   E8  synth with 250 constant-envelope noise bands (flatten_noise), baseline loss
  flat1000  E9  synth with its 1000 bands made constant-envelope, baseline loss
  clean1000 E10 synth fixes only: int16 noise bands (int8's quantisation floor is a hiss) and linear amplitude ramps
                (the steps are a zipper hiss), baseline loss
  syn250    E11 250 bands, iteratively flattened (flat envelope, exactly band-limited), clean stopband, int16, ramps
  syn500    E12 the same with 500 bands (the best ideal-control ceiling on drone and harbour, on par on drums)
  syn500env E13 syn500 with the band-envelope loss (the candidate recipe: new synth + MRSTFT + EnvelopeLoss 0.1)

Checkpoints every 20 epochs. Style labels come from make_styles.py (experiments/styles/labels).

  python experiments/styles/make_configs.py --set percussion
"""

import argparse
import os
import re

FULL = {
    "harbour": "configs/factory3/04_harbour_nosilence.yaml",
    "percussion": "configs/factory2/02_percussion_nosilence.yaml",
    "drone_sampled": "configs/factory2/04_drone_sampled_nosilence.yaml",
}
LABELS = "/home/btadeusz/code/plaud/experiments/styles/labels"


def sub(pattern, repl, s):
  s2, k = re.subn(pattern, repl, s, count=1, flags=re.M)
  assert k == 1, pattern
  return s2


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--set", required=True, choices=list(FULL))
  args = ap.parse_args()
  src = FULL[args.set]
  base = open(src).read()
  base = sub(r"^  training_dir: \S+", "  training_dir: training/research", base)
  base = sub(r"^(trainer:\n)", r"\1  save_every_n_epochs: 20\n", base)
  os.makedirs("configs/research_styles", exist_ok=True)
  for exp, what in [("oracle", "one style embedding per 16 s passage"), ("k12", "12 style clusters of 16 s passages"),
                    ("lat4", "4 latents, no styles"), ("rangeadv", "dynamic-range loss + adversarial loss"),
                    ("range", "dynamic-range loss alone"), ("adv", "adversarial loss alone"),
                    ("env", "band-envelope loss"), ("flat250", "250 constant-envelope noise bands"),
                    ("flat1000", "1000 constant-envelope noise bands"),
                    ("clean1000", "int16 noise bands and linear amplitude ramps"),
                    ("syn250", "250 flattened, band-limited int16 noise bands with linear ramps"),
                    ("syn500", "500 flattened, band-limited int16 noise bands with linear ramps"),
                    ("syn500env", "500 flattened, band-limited int16 noise bands with linear ramps, band-envelope loss")]:
    c = sub(r"^  name: \S+", f"  name: st_{args.set}_{exp}", base)
    if exp in ("oracle", "k12"):
      c = sub(r"^(data:\n)", rf"\1  styles: {LABELS}/{args.set}_{exp}.npy\n", c)
      c = sub(r"^(model:\n)", r"\1  style_dim: 16\n", c)
    if exp == "lat4":
      c = sub(r"(- name: latents\n\s+dim: )2", r"\g<1>4", c)
    if exp in ("rangeadv", "range"):
      c = sub(r"^(losses:\n)", r"\1  - type: DynamicRangeLoss\n    weight: 0.1\n    params:\n      fs: 48000\n", c)
    if exp in ("env", "syn500env"):
      c = sub(r"^(losses:\n)", r"\1  - type: EnvelopeLoss\n    weight: 0.1\n    params:\n      fs: 48000\n", c)
    if exp == "clean1000":
      c = sub(r"^(\s+)n_filters: (\d+)\n", r"\1n_filters: \2\n\1noise_bits: 16\n\1ramp: true\n", c)
    if exp in ("syn250", "syn500", "syn500env"):
      c = sub(r"^(\s+)n_filters: \d+\n", rf"\1n_filters: {exp[3:6]}\n\1flatten_noise: true\n\1flatten_iters: 8\n"
              r"\1clean_stopband: true\n\1noise_bits: 16\n\1ramp: true\n", c)
    if exp.startswith("flat"):
      c = sub(r"^(\s+)n_filters: \d+\n", rf"\1n_filters: {exp[4:]}\n\1flatten_noise: true\n", c)
    if exp in ("rangeadv", "adv"):
      c = sub(r"^(adversarial:\n  )enabled: false", r"\1enabled: true", c)
      c = sub(r"g_start_epoch: \d+", "g_start_epoch: 40", c)
      c = sub(r"d_start_epoch: \d+", "d_start_epoch: 30", c)
    head = (f"# STYLE EXPERIMENT (2026-10-22): {src} with {what}.\n"
            f"# experiments/styles/make_configs.py. Synth stage only.\n#\n")
    out = f"configs/research_styles/{args.set}_{exp}.yaml"
    open(out, "w").write(head + c)
    print(out)


if __name__ == "__main__":
  main()
