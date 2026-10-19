"""Collect CC0 music from OpenGameArt.org.

Searches the music section with the CC0 licence filter for each keyword, opens every result and keeps
it only if its own licence field links to CC0 (creativecommons.org/publicdomain/zero). Downloads its
audio files (ogg/mp3/wav/flac) to <out>/audio/ and writes <out>/items.jsonl (title, author, page,
files, licence, credit).

  python -m cli.source_opengameart --out /mnt/mariadata/datasets/factory/_raw/oga --keyword ambient --keyword drone
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

BASE = "https://opengameart.org"
UA = {"User-Agent": "Mozilla/5.0 (compatible; plaud-dataset-builder/1.0; github.com/blazejkotowski/plaud)"}
AUDIO = re.compile(r'href="(https://opengameart\.org/sites/default/files/[^"]+\.(?:ogg|mp3|wav|flac))"', re.I)


def fetch(url: str, delay: float, tries: int = 5) -> str:
  for k in range(tries):
    time.sleep(delay * (2 ** k))
    try:
      with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, ConnectionError, TimeoutError):
      if k == tries - 1:
        raise


def search(keyword: str, delay: float, max_pages: int) -> list:
  nodes = []
  for page in range(max_pages):
    q = urllib.parse.urlencode({"keys": keyword, "field_art_type_tid[0]": 12, "field_art_licenses_tid[0]": 4,
                                "sort_by": "count", "sort_order": "DESC", "page": page})
    found = re.findall(r'<span class="art-preview-title"><a href="(/content/[^"]+)"', fetch(f"{BASE}/art-search-advanced?{q}", delay))
    new = [n for n in dict.fromkeys(found) if n not in nodes]
    if not new:
      break
    nodes += new
  return nodes


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--out", required=True)
  ap.add_argument("--keyword", action="append", required=True)
  ap.add_argument("--max-pages", type=int, default=15)
  ap.add_argument("--delay", type=float, default=1.0)
  args = ap.parse_args()
  os.makedirs(os.path.join(args.out, "audio"), exist_ok=True)
  manifest = os.path.join(args.out, "items.jsonl")
  seen = set()
  if os.path.exists(manifest):
    seen = {json.loads(l)["page"] for l in open(manifest) if l.strip()}

  nodes = []
  for k in args.keyword:
    found = search(k, args.delay, args.max_pages)
    print(f"{k}: {len(found)} results", flush=True)
    nodes += [n for n in found if n not in nodes]

  kept = skipped = 0
  with open(manifest, "a") as out:
    for node in nodes:
      url = BASE + node
      if url in seen:
        continue
      page = fetch(url, args.delay)
      lic = re.search(r"field-name-field-art-licenses.*?</div></div></div>", page, re.S)
      if not lic or "publicdomain/zero" not in lic.group(0):
        skipped += 1
        continue
      title = re.search(r"<title>(.*?)\s*\|", page, re.S)
      title = html.unescape(title.group(1)).strip() if title else node
      author = re.search(r'field-name-author-submitter.*?<a href="/users/[^"]+">([^<]+)</a>', page, re.S)
      author = html.unescape(author.group(1)).strip() if author else "unknown"
      files = []
      for src in dict.fromkeys(AUDIO.findall(page)):
        name = urllib.parse.unquote(os.path.basename(src))
        path = os.path.join(args.out, "audio", f"{node.rsplit('/', 1)[-1]}__{name}")
        if not os.path.exists(path):
          time.sleep(args.delay)
          try:
            urllib.request.urlretrieve(urllib.parse.quote(src, safe=":/"), path)
          except Exception as e:
            print(f"  failed {src}: {e}", flush=True)
            continue
        files.append(os.path.relpath(path, args.out))
      if not files:
        continue
      item = dict(page=url, title=title, author=author, files=files, licence="CC0 1.0 (OpenGameArt)",
                  credit=f"\"{title}\" by {author}, OpenGameArt.org, CC0")
      out.write(json.dumps(item) + "\n")
      out.flush()
      kept += 1
      print(f"  {title[:60]} ({author}): {len(files)} files", flush=True)
  print(f"kept {kept}, skipped {skipped} (not CC0 on the page)")


if __name__ == "__main__":
  main()
