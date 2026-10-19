"""Status of a training batch as JSON: runs, validation curves, datasets and preview checks.

  python -m cli.batch_status configs/factory1 --previews /tmp/previews > status.json

{"runs": [...], "curves": [...], "datasets": [...], "previews": [...]}; the synth stage only (the
prior and export are reported by train_multiple's report).
"""

import argparse
import glob
import json
import os
import time

from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def scalars(run_dir: str) -> dict:
  out = {}
  for f in sorted(glob.glob(os.path.join(run_dir, "*", "*", "events*")), key=os.path.getmtime):
    ea = EventAccumulator(f, size_guidance={"scalars": 0})
    ea.Reload()
    for tag in ea.Tags()["scalars"]:
      out.setdefault(tag, []).extend(ea.Scalars(tag))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("configs_dir")
  ap.add_argument("--previews", default=None, help="folder of cli.preview_synth outputs, one subfolder per run")
  args = ap.parse_args()

  runs, curves, sets = [], [], {}
  for path in sorted(glob.glob(os.path.join(args.configs_dir, "*.yaml"))):
    cfg = OmegaConf.load(path)
    name = cfg.experiment.name
    data = os.path.dirname(cfg.data.dataset_path.rstrip("/"))
    silence = bool(cfg.get("silence", {}).get("enabled", False))
    run_dir = os.path.join(cfg.experiment.training_dir, "synth", name)
    s = scalars(run_dir) if os.path.isdir(run_dir) else {}
    epochs = {}
    for tag in ("val_loss", "val_silence_db"):
      for e in s.get(tag, []):
        epochs.setdefault(e.step, {})[tag] = e.value
    epoch_of = {e.step: int(e.value) for e in s.get("epoch", [])}
    for step in sorted(epochs):
      v = epochs[step]
      curves.append(dict(run=name, dataset=os.path.basename(data), silence="on" if silence else "off",
                         step=step, epoch=epoch_of.get(step, max([0] + [x for k, x in epoch_of.items() if k <= step])),
                         val_loss=round(v["val_loss"], 4) if "val_loss" in v else None,
                         val_silence_db=round(v["val_silence_db"], 1) if "val_silence_db" in v else None))
    vl = [e.value for e in s.get("val_loss", [])]
    sd = [e.value for e in s.get("val_silence_db", [])]
    ep = s.get("epoch", [])
    wall = [e.wall_time for e in s.get("recons_loss", [])]
    max_epochs = int(cfg.trainer.max_epochs)
    epoch = int(ep[-1].value) + 1 if ep else 0
    done = os.path.exists(os.path.join(run_dir, "last.ckpt")) and epoch >= max_epochs
    state = "done" if done else ("training" if wall and time.time() - wall[-1] < 600 else ("stopped" if wall else "queued"))
    runs.append(dict(run=name, dataset=os.path.basename(data), silence="on" if silence else "off", state=state,
                     epoch=min(epoch, max_epochs), max_epochs=max_epochs, steps=ep[-1].step if ep else 0,
                     best_val_loss=round(min(vl), 4) if vl else None, last_val_loss=round(vl[-1], 4) if vl else None,
                     silence_db=round(sd[-1], 1) if sd else None,
                     hours=round((wall[-1] - wall[0]) / 3600, 2) if len(wall) > 1 else 0.0))
    if data not in sets and os.path.exists(os.path.join(data, "manifest.jsonl")):
      rows = [json.loads(l) for l in open(os.path.join(data, "manifest.jsonl"))]
      minutes = 0.0
      import soundfile as sf
      for r in rows:
        minutes += sf.info(os.path.join(data, r["file"])).duration / 60
      licences = sorted({r["licence"] for r in rows})
      sets[data] = dict(dataset=os.path.basename(data), minutes=round(minutes, 1), files=len(rows),
                        sources=len({r["credit"] for r in rows}), licence="; ".join(licences),
                        ready=os.path.exists(os.path.join(data, "READY")))

  previews = []
  if args.previews and os.path.isdir(args.previews):
    for st in sorted(glob.glob(os.path.join(args.previews, "*", "stats.json"))):
      d = json.load(open(st))
      run = os.path.basename(os.path.dirname(st))
      for c in d["clips"]:
        previews.append(dict(id=f"{run}#{c['clip']}", run=run.rsplit("_e", 1)[0], epoch=d.get("epoch"), clip=c["clip"],
                             recon_db=c["recon_db"], orig_db=c["orig_db"], fade_tail_db=c["fade_tail_db"],
                             loudness0_db=c["loudness0_db"]))
  print(json.dumps(dict(at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), runs=runs, curves=curves,
                        datasets=list(sets.values()), previews=previews)))


if __name__ == "__main__":
  main()
