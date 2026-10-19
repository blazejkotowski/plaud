"""Collect public-domain cylinder recordings from the UCSB Cylinder Audio Archive.

Recordings made before 1923 are in the public domain ("can be freely downloaded and used for
whatever purpose", https://cylinders.library.ucsb.edu/licensing.php). Only items with a known
release year <= --max-year are kept; anything with a voice in its performers or subjects can be
dropped with --instrumental.

Writes <out>/items.jsonl (one line per digitised take: title, year, performers, subjects, page,
audio URL, credit) and the MP3s to <out>/mp3/.

  python -m cli.source_ucsb --out /mnt/mariadata/datasets/factory/_raw/ucsb \\
      --query keyword:xylophone --query subject:Bells --instrumental
"""

import argparse
import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://cylinders.library.ucsb.edu/"
UA = {"User-Agent": "Mozilla/5.0 (compatible; plaud-dataset-builder/1.0; github.com/blazejkotowski/plaud)"}

SOLO_VOICE = (r"vocals?|vocalists?|tenor|baritone|soprano|contralto|mezzo-soprano|singers?|speakers?|"
              r"recitation|reciter|monologue|monologues|songs|talking|comic dialogues?|sketch(es)?|"
              r"yodel(ing)?|whistl(ing|er)")
VOICE = re.compile(r"\b(" + SOLO_VOICE + r"|choir|chorus|quartet)\b", re.I)
SOLO = re.compile(r"\b(" + SOLO_VOICE + r")\b", re.I)
VOICE_BASS = re.compile(r"\bbass\b(?!\s*(drum|horn|viol|clarinet|saxophone))", re.I)


def fetch(url: str, delay: float) -> str:
  time.sleep(delay)
  with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
    return r.read().decode("utf-8", "replace")


def field(page: str, label: str) -> list:
  """The entries of a labelled row on a detail page."""
  m = re.search(r'data-label">\s*' + re.escape(label) + r'\s*:?\s*</div>(.*?)</div>\s*</div>', page, re.S)
  if not m:
    return []
  entries = re.findall(r'data-entry">(.*?)</div>', m.group(1) + "</div>", re.S)
  return [html.unescape(re.sub(r"<[^>]+>", "", e)).strip() for e in entries if e.strip()]


def search_ids(query_type: str, query: str, delay: float) -> list:
  ids, start = [], 1
  while True:
    url = BASE + "search.php?" + urllib.parse.urlencode({"query_type": query_type, "query": query, "nq": 1, "start": start})
    try:
      page = fetch(url, delay)
    except urllib.error.HTTPError:
      if start > 1:
        return ids   # past the last page
      raise
    found = re.findall(r"detail\.php\?query_type=mms_id&query=(\d+)", page)
    new = [i for i in dict.fromkeys(found) if i not in ids]
    if not new:
      return ids
    ids += new
    start += 25


def parse_item(mms: str, delay: float) -> list:
  url = BASE + "detail.php?" + urllib.parse.urlencode({"query_type": "mms_id", "query": mms})
  page = fetch(url, delay)
  title = re.search(r'<h1 id="uniform_title[^"]*">(.*?)</h1>', page, re.S)
  title = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", title.group(1))).split()) if title else ""
  years = field(page, "Release year")
  performers = field(page, "Performers")
  subjects = field(page, "Subjects")
  notes = field(page, "Notes")
  takes = re.findall(r'<h3>Cylinder (\d+)</h3>\s*<audio[^>]*src="([^"]+)"', page)
  return [dict(mms_id=mms, cylinder=int(n), title=title, year=years[0] if years else "", performers=performers,
               subjects=subjects, notes=notes, page=url, audio=src) for n, src in takes]


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--out", required=True)
  ap.add_argument("--query", action="append", required=True, help="type:text, type in keyword/subject/title/author")
  ap.add_argument("--max-year", type=int, default=1922)
  ap.add_argument("--instrumental", action="store_true", help="drop items with a voice in performers or subjects")
  ap.add_argument("--allow-choir", action="store_true", help="with --instrumental: keep choirs and vocal ensembles")
  ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
  args = ap.parse_args()

  os.makedirs(os.path.join(args.out, "mp3"), exist_ok=True)
  manifest = os.path.join(args.out, "items.jsonl")
  seen = set()
  if os.path.exists(manifest):
    with open(manifest) as f:
      seen = {json.loads(l)["mms_id"] for l in f if l.strip()}

  ids = []
  for q in args.query:
    qt, text = q.split(":", 1)
    found = search_ids(qt, text, args.delay)
    print(f"{q}: {len(found)} records", flush=True)
    ids += [i for i in found if i not in ids]

  kept = skipped_year = skipped_voice = 0
  with open(manifest, "a") as out:
    for mms in ids:
      if mms in seen:
        continue
      seen.add(mms)
      for take in parse_item(mms, args.delay):
        year = take["year"]
        if not re.fullmatch(r"\d{4}", year) or int(year) > args.max_year:
          skipped_year += 1
          continue
        text = " ".join(take["performers"] + take["subjects"] + take["notes"] + [take["title"]])
        voice = SOLO if args.allow_choir else VOICE
        if args.instrumental and (voice.search(text) or VOICE_BASS.search(" ".join(take["performers"]))):
          skipped_voice += 1
          continue
        path = os.path.join(args.out, "mp3", f"cyl{take['cylinder']}.mp3")
        if not os.path.exists(path):
          time.sleep(args.delay)
          urllib.request.urlretrieve(take["audio"], path)
        take["file"] = os.path.relpath(path, args.out)
        take["licence"] = "Public domain (US, recorded before 1923)"
        take["credit"] = f"\"{take['title']}\" ({year}), UCSB Cylinder Audio Archive, University of California, Santa Barbara Library"
        out.write(json.dumps(take) + "\n")
        out.flush()
        kept += 1
        print(f"  cyl{take['cylinder']} {year} {take['title'][:70]}", flush=True)
  print(f"kept {kept}, skipped {skipped_year} (year unknown or after {args.max_year}), {skipped_voice} (voice)")


if __name__ == "__main__":
  main()
