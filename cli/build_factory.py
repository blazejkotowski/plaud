"""Build the factory training sets from the raw sources (see cli/source_*.py).

Each set goes to <root>/<name>/audio/*.wav (48 kHz mono, level-normalised) with a manifest
(manifest.jsonl: file, source, licence, credit) and CREDITS.md next to it. Selection:

  jukebox_jazz     Citizen DJ National Jukebox: Jazz excerpts without a voice in the credits.
  wax_percussion   UCSB cylinders (pre-1923, no voice), spoken announcements trimmed, the most
                   percussive first (percussive energy share x onset rate).
  magnetosphere    Cassini RPWS wideband audio, capped per day so no single event dominates.
  drone            Everything sustained: Citizen DJ classical / popular / folk / FMA excerpts and
                   UCSB organ, choir, harmonium, strings..., the fewest attacks first.

  python -m cli.build_factory --root /mnt/mariadata/datasets/factory --set jukebox_jazz --minutes 75
"""

import argparse
import glob
import json
import os
import random
import re
import subprocess

import librosa
import numpy as np
import soundfile as sf

FS = 48000
VOICE = re.compile(r"\b(vocals?|vocalists?|tenor|baritone|soprano|contralto|mezzo-soprano|singers?|speakers?|"
                   r"recitation|reciter|monologue|talking|yodel(ing)?|whistl(ing|er)|duet|quartet|trio vocal|"
                   r"sings?|singing|sung|lyrics?|words by|dialogues?|vaudeville|humor|comic|sketch(es)?)\b", re.I)
CHOIR = re.compile(r"\b(choir|chorus|choral)\b", re.I)


def load_mono(path: str) -> np.ndarray:
  if path.endswith(".mp3"):
    raw = subprocess.run(["ffmpeg", "-v", "quiet", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(FS), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()
  x, sr = sf.read(path, dtype="float32", always_2d=True)
  x = x.mean(axis=1)
  return librosa.resample(x, orig_sr=sr, target_sr=FS) if sr != FS else x


def normalise(x: np.ndarray, rms_db: float = -20.0) -> np.ndarray:
  x = x - x.mean()
  rms = np.sqrt(np.mean(x ** 2)) + 1e-9
  x = x * (10 ** (rms_db / 20) / rms)
  peak = np.abs(x).max()
  return x / peak * 0.98 if peak > 0.98 else x


def onset_rate(x: np.ndarray) -> float:
  """Onsets per second."""
  on = librosa.onset.onset_detect(y=x, sr=FS, hop_length=512, units="time")
  return len(on) / max(1.0, len(x) / FS)


def percussive_share(x: np.ndarray) -> float:
  S = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
  H, P = librosa.decompose.hpss(S)
  return float((P ** 2).sum() / ((H ** 2).sum() + (P ** 2).sum() + 1e-12))


def citizen_items(pack: str) -> list:
  """Excerpts of a Citizen DJ pack (the files on disk; the CSV lists more) with their item's credits."""
  items = []
  for path in sorted(glob.glob(os.path.join(pack, "excerpts", "*.wav"))):
    item = os.path.basename(path).rsplit("_", 2)[0]          # <title>_<item id>
    att = os.path.join(pack, "attributions", item + ".txt")
    text = open(att).read() if os.path.exists(att) else ""
    line = lambda key: next((l[len(key):].strip() for l in text.splitlines() if l.startswith(key)), "")
    credit = next((l for l in text.splitlines() if l.startswith("Contains samples of")), item)
    items.append(dict(path=path, item=item, title=line("Title:"), contributors=line("Contributors:"), url=line("URL:"),
                      credit=credit + " (Citizen DJ, Library of Congress)",
                      licence="Public domain (Citizen DJ, Library of Congress)"))
  return items


def ucsb_items(raw: str, allow_choir: bool) -> list:
  items = []
  for line in open(os.path.join(raw, "items.jsonl")):
    it = json.loads(line)
    text = " ".join(it["performers"] + it["subjects"] + it["notes"] + [it["title"]])
    if VOICE.search(text) or (not allow_choir and CHOIR.search(text)):
      continue
    announced = any("announc" in n.lower() for n in it["notes"]) or int(it["year"]) <= 1908
    items.append(dict(path=os.path.join(raw, it["file"]), item=f"cyl{it['cylinder']}", title=it["title"],
                      url=it["page"], credit=it["credit"], licence=it["licence"], trim=10.0 if announced else 1.0))
  return items


def write_set(out: str, chosen: list, minutes: float):
  audio_dir = os.path.join(out, "audio")
  os.makedirs(audio_dir, exist_ok=True)
  total = 0.0
  with open(os.path.join(out, "manifest.jsonl"), "w") as man:
    for it in chosen:
      if total >= minutes * 60:
        break
      x = it.get("audio")
      if x is None:
        x = load_mono(it["path"])
        x = x[int(it.get("trim", 0.0) * FS):]
      if len(x) < 2 * FS:
        continue
      name = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.basename(it["path"]).rsplit(".", 1)[0]) + ".wav"
      sf.write(os.path.join(audio_dir, name), normalise(x).astype(np.float32), FS, subtype="PCM_24")
      total += len(x) / FS
      man.write(json.dumps(dict(file="audio/" + name, source=it.get("url", it["path"]), title=it.get("title", ""),
                                licence=it["licence"], credit=it["credit"],
                                score=it.get("score"))) + "\n")
  credits = sorted({json.loads(l)["credit"] for l in open(os.path.join(out, "manifest.jsonl"))})
  with open(os.path.join(out, "CREDITS.md"), "w") as f:
    f.write(f"# {os.path.basename(out)}\n\n{total / 60:.1f} minutes of audio. Sources and licences are listed per file in "
            "manifest.jsonl. Credits:\n\n" + "\n".join("- " + c for c in credits) + "\n")
  print(f"{os.path.basename(out)}: {total / 60:.1f} min from {len(credits)} sources")


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--root", required=True)
  ap.add_argument("--set", required=True, choices=["jukebox_jazz", "wax_percussion", "magnetosphere", "drone", "drone_ambient"])
  ap.add_argument("--minutes", type=float, default=75.0)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()
  random.seed(args.seed)
  raw = os.path.join(args.root, "_raw")
  cdj = os.path.join(raw, "citizendj")
  out = os.path.join(args.root, args.set)

  if args.set == "jukebox_jazz":
    items = [it for it in citizen_items(os.path.join(cdj, "loc.gov_national-jukebox-jazz"))
             if not VOICE.search(it["contributors"]) and not CHOIR.search(it["contributors"])]
    random.shuffle(items)
    write_set(out, items, args.minutes)

  elif args.set == "wax_percussion":
    items = ucsb_items(os.path.join(raw, "ucsb"), allow_choir=False)
    for it in items:
      x = load_mono(it["path"])[int(it["trim"] * FS):]
      it["audio"] = x
      it["score"] = round(percussive_share(x) * onset_rate(x), 4)
    items.sort(key=lambda it: -it["score"])
    write_set(out, items, args.minutes)

  elif args.set == "magnetosphere":
    items = []
    for line in open(os.path.join(raw, "cassini", "items.jsonl")):
      it = json.loads(line)
      path = os.path.join(raw, "cassini", it["file"])
      x = load_mono(path)
      seg = 90 * FS                                   # 90 s pieces, so no single long file dominates
      for k in range(0, len(x), seg):
        items.append(dict(path=f"{path[:-4]}_s{k // seg:02d}.wav", audio=x[k:k + seg], url=it["source"],
                          title=it["day"] + " " + it["band"], credit=it["credit"], licence=it["licence"],
                          key=(it["day"], it["band"])))
    # round-robin over (day, band) so every event and both bands are represented before any repeats
    by_day = {}
    for it in items:
      by_day.setdefault(it["key"], []).append(it)
    ordered = []
    while any(by_day.values()):
      for d in sorted(by_day):
        if by_day[d]:
          ordered.append(by_day[d].pop(0))
    write_set(out, ordered, args.minutes)

  elif args.set == "drone_ambient":
    build_drone_ambient(raw, out, args.minutes)

  elif args.set == "drone":
    items = []
    for pack in ["loc.gov_national-jukebox-classical", "loc.gov_national-jukebox-popular",
                 "loc.gov_national-jukebox-folk-songs", "loc.gov_fma"]:
      items += [it for it in citizen_items(os.path.join(cdj, pack)) if not VOICE.search(it["contributors"])]
    items += ucsb_items(os.path.join(raw, "ucsb_sustained"), allow_choir=True)
    scored = []
    for it in items:
      x = load_mono(it["path"])[int(it.get("trim", 0.0) * FS):]
      if len(x) < 5 * FS or np.sqrt(np.mean(x ** 2)) < 1e-3:
        continue
      it["audio"] = x
      it["score"] = round(onset_rate(x), 3)
      scored.append(it)
    scored.sort(key=lambda it: it["score"])          # fewest attacks first
    write_set(out, scored, args.minutes)


def build_drone_ambient(raw: str, out: str, minutes: float, per_author: int = 6):
  """CC0 OpenGameArt tracks, the most sustained first (fewest onsets per second), >= 45 s, a few per composer."""
  items = []
  for line in open(os.path.join(raw, "oga", "items.jsonl")):
    it = json.loads(line)
    for f in it["files"]:
      try:
        x = load_mono(os.path.join(raw, "oga", f))
      except Exception:
        continue
      if len(x) < 45 * FS or np.sqrt(np.mean(x ** 2)) < 1e-3:
        continue
      items.append(dict(path=os.path.join(raw, "oga", f), audio=x, title=it["title"], url=it["page"], author=it["author"],
                        credit=it["credit"], licence=it["licence"], score=round(onset_rate(x), 3)))
      break   # one file per entry (the others are usually loops or alternate mixes of the same piece)
  items.sort(key=lambda it: it["score"])
  per, chosen = {}, []
  for it in items:
    if per.get(it["author"], 0) < per_author:
      per[it["author"]] = per.get(it["author"], 0) + 1
      chosen.append(it)
  write_set(out, chosen, minutes)


if __name__ == "__main__":
  main()
