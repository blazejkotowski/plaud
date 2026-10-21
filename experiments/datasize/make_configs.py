"""Configs for the data-size experiment: configs/research_datasize/<set>_s<seconds>.yaml.

Each is the full model's config (the factory *_nosilence one, whose trained synth is the full-size
point) with the subset as dataset, training_dir training/research, and the full set's epoch length
(trainer.samples_per_epoch = its number of 2 s chunks, sampled with replacement), so every size gets
the same 160 epochs of the same number of steps and the same epoch-based beta warm-up. A checkpoint
is kept every 20 epochs for the learning curves.

  python experiments/datasize/make_configs.py --sizes 16,128,960
"""

import argparse
import glob
import os
import re

import soundfile as sf

FULL = {
    "harbour": "configs/factory3/04_harbour_nosilence.yaml",
    "percussion": "configs/factory2/02_percussion_nosilence.yaml",
    "drone_sampled": "configs/factory2/04_drone_sampled_nosilence.yaml",
}
DATA = "/mnt/mariadata/datasets/research/datasize"


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--sizes", default="16,128,960")
  args = ap.parse_args()
  os.makedirs("configs/research_datasize", exist_ok=True)
  for name, src in FULL.items():
    s = open(src).read()
    full_path = re.search(r"^  dataset_path: (\S+)", s, re.M).group(1)
    full_s = sum(sf.info(f).duration for f in glob.glob(os.path.join(full_path, "**", "*.wav"), recursive=True))
    per_epoch = int(full_s * 48000) // 96000
    for size in (int(x) for x in args.sizes.split(",")):
      c = s
      c = re.sub(r"^  name: \S+", f"  name: ds_{name}_s{size}", c, count=1, flags=re.M)
      c = re.sub(r"^  training_dir: \S+", "  training_dir: training/research", c, count=1, flags=re.M)
      c = re.sub(r"^  dataset_path: \S+", f"  dataset_path: {DATA}/{name}/s{size}/audio", c, count=1, flags=re.M)
      c, k = re.subn(r"^(trainer:\n)", rf"\1  samples_per_epoch: {per_epoch}\n  save_every_n_epochs: 20\n", c, count=1, flags=re.M)
      assert k == 1, src
      head = (f"# DATA-SIZE EXPERIMENT (2026-10-21): {src} trained on its {size} s nested subset\n"
              f"# ({DATA}/{name}, experiments/datasize/make_subsets.py) with the full set's epoch length\n"
              f"# ({per_epoch} chunks = {full_s / 60:.1f} min, sampled with replacement). Synth stage only.\n#\n")
      out = f"configs/research_datasize/{name}_s{size}.yaml"
      open(out, "w").write(head + c)
      print(out, f"(epoch {per_epoch} chunks)")


if __name__ == "__main__":
  main()
