"""Collect CC0 field recordings from Freesound by theme.

  search    queries a theme's searches with the CC0 licence filter, drops sounds tagged as music, speech
            or game audio, ranks the rest (rating, downloads, length, sample rate) with at most a few
            sounds per recordist, and writes <out>/<theme>/candidates.jsonl
  previews  downloads the high-quality preview of the top N candidates (API key only) for auditioning
  download  downloads the originals of the chosen candidates (needs an OAuth2 token, see `token`)
  token     exchanges an authorisation code for an OAuth2 token (stored next to the credentials)

Credentials: ~/.freesound.json = {"client_id", "client_secret"}; the client secret doubles as the API key.

  python -m cli.source_freesound search --out /mnt/mariadata/datasets/factory/_raw/freesound --theme underground
"""

import argparse
import json
import os
import time
import urllib.parse
import urllib.request

API = "https://freesound.org/apiv2/"
CRED = os.path.expanduser("~/.freesound.json")
FIELDS = "id,name,username,duration,license,tags,description,samplerate,channels,avg_rating,num_ratings,num_downloads,geotag,previews,type,url"
EXCLUDE = {"music", "song", "singing", "voice", "speech", "talking", "talk", "vocal", "game", "videogame", "8bit", "synth",
           "synthesizer", "melody", "beat", "drum-loop", "announcement", "announcer", "narration", "podcast", "interview"}

THEMES = {
    "underground": ["metro station", "subway station", "underground station", "u-bahn", "tube station london", "metro train arriving",
                    "subway train interior", "tunnel train", "escalator", "underpass", "parking garage", "metro paris", "metro moscow",
                    "subway new york", "metro tokyo", "s-bahn tunnel", "service tunnel", "underground car park"],
    "harbour": ["harbour", "harbor", "port ambience", "docks", "ship horn", "foghorn", "container terminal", "port crane",
                "boat creaking", "rigging wind", "marina", "ferry", "buoy bell", "mooring ropes", "shipyard", "hamburg port",
                "rotterdam port", "fishing harbour"],
    "machine_halls": ["factory", "machine room", "printing press", "loom", "power station", "engine room", "server room",
                      "industrial machinery", "turbine", "generator hum", "workshop machines", "mill machinery", "steel works",
                      "boiler room", "ventilation plant", "assembly line"],
    "contact_structure": ["contact microphone", "contact mic", "piezo", "hydrophone", "fence contact", "bridge contact",
                          "cable vibration", "singing wire", "wind wire", "pipe resonance", "structure borne", "metal structure resonance",
                          "railing contact", "wire fence wind"],
}


def cred() -> dict:
  return json.load(open(CRED))


def get(path: str, params: dict, auth: str = "key", delay: float = 0.5) -> dict:
  c = cred()
  headers = {}
  if auth == "key":
    params = dict(params, token=c["client_secret"])
  else:
    headers["Authorization"] = f"Bearer {c['access_token']}"
  for k in range(5):
    time.sleep(delay * (2 ** k))
    try:
      req = urllib.request.Request(API + path + "?" + urllib.parse.urlencode(params), headers=headers)
      with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)
    except urllib.error.HTTPError as e:
      if e.code == 429 or e.code >= 500:
        continue
      raise
  raise RuntimeError(f"Freesound kept failing: {path}")


def score(s: dict) -> float:
  rating = (s.get("avg_rating") or 0.0) * min(1.0, (s.get("num_ratings") or 0) / 3.0)
  pop = min(1.0, (s.get("num_downloads") or 0) / 400.0)
  length = min(1.0, s["duration"] / 120.0)
  sr = 1.0 if (s.get("samplerate") or 0) >= 44100 else 0.5
  return round(2.0 * rating / 5.0 + pop + length + sr, 3)


def cmd_search(args):
  out_dir = os.path.join(args.out, args.theme)
  os.makedirs(out_dir, exist_ok=True)
  found = {}
  for q in THEMES[args.theme]:
    page = 1
    while page <= args.pages:
      d = get("search/text/", {"query": q, "filter": 'license:"Creative Commons 0" duration:[30 TO 1800]',
                               "fields": FIELDS, "page_size": 150, "page": page, "sort": "score"})
      for s in d["results"]:
        tags = {t.lower() for t in s.get("tags", [])}
        if tags & EXCLUDE:
          continue
        s["query"] = q
        found.setdefault(s["id"], s)
      if not d.get("next"):
        break
      page += 1
    print(f"{q}: {len(found)} so far", flush=True)
  ranked = sorted(found.values(), key=lambda s: -score(s))
  per_user, kept = {}, []
  for s in ranked:
    if per_user.get(s["username"], 0) >= args.per_user:
      continue
    per_user[s["username"]] = per_user.get(s["username"], 0) + 1
    s["score"] = score(s)
    kept.append(s)
  with open(os.path.join(out_dir, "candidates.jsonl"), "w") as f:
    for s in kept:
      f.write(json.dumps(s) + "\n")
  print(f"{args.theme}: {len(kept)} candidates from {len(per_user)} recordists, "
        f"{sum(s['duration'] for s in kept) / 60:.0f} min")


def cmd_previews(args):
  out_dir = os.path.join(args.out, args.theme)
  os.makedirs(os.path.join(out_dir, "previews"), exist_ok=True)
  rows = [json.loads(l) for l in open(os.path.join(out_dir, "candidates.jsonl"))][: args.top]
  for s in rows:
    path = os.path.join(out_dir, "previews", f"{s['id']}.ogg")
    if not os.path.exists(path):
      urllib.request.urlretrieve(s["previews"]["preview-hq-ogg"], path)
      time.sleep(0.5)
  print(f"{len(rows)} previews in {out_dir}/previews")


def cmd_token(args):
  c = cred()
  data = urllib.parse.urlencode({"client_id": c["client_id"], "client_secret": c["client_secret"],
                                 "grant_type": "authorization_code", "code": args.code}).encode()
  with urllib.request.urlopen(API + "oauth2/access_token/", data=data, timeout=60) as r:
    tok = json.load(r)
  c.update(access_token=tok["access_token"], refresh_token=tok["refresh_token"], token_time=time.time())
  old = os.umask(0o077)
  json.dump(c, open(CRED, "w"))
  os.umask(old)
  print("token stored; expires in", tok.get("expires_in"), "s")


def cmd_download(args):
  out_dir = os.path.join(args.out, args.theme)
  os.makedirs(os.path.join(out_dir, "originals"), exist_ok=True)
  ids = {int(x) for x in open(args.ids).read().split()}
  c = cred()
  with open(os.path.join(out_dir, "downloaded.jsonl"), "a") as man:
    for line in open(os.path.join(out_dir, "candidates.jsonl")):
      s = json.loads(line)
      if s["id"] not in ids:
        continue
      path = os.path.join(out_dir, "originals", f"{s['id']}.{s.get('type') or 'wav'}")
      if os.path.exists(path):
        continue
      req = urllib.request.Request(f"{API}sounds/{s['id']}/download/", headers={"Authorization": f"Bearer {c['access_token']}"})
      with urllib.request.urlopen(req, timeout=300) as r, open(path, "wb") as f:
        f.write(r.read())
      man.write(json.dumps(dict(id=s["id"], file=os.path.relpath(path, out_dir), name=s["name"], username=s["username"],
                                url=s["url"], licence="CC0 1.0 (Freesound)",
                                credit=f"\"{s['name']}\" by {s['username']}, freesound.org/s/{s['id']}, CC0")) + "\n")
      man.flush()
      time.sleep(2.1)   # 30 downloads a minute at most
  print("done")


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest="cmd", required=True)
  for name in ("search", "previews", "download"):
    p = sub.add_parser(name)
    p.add_argument("--out", required=True)
    p.add_argument("--theme", required=True, choices=list(THEMES))
    if name == "search":
      p.add_argument("--pages", type=int, default=2)
      p.add_argument("--per-user", type=int, default=6)
    if name == "previews":
      p.add_argument("--top", type=int, default=40)
    if name == "download":
      p.add_argument("--ids", required=True, help="file with the chosen sound ids")
  t = sub.add_parser("token")
  t.add_argument("--code", required=True)
  args = ap.parse_args()
  dict(search=cmd_search, previews=cmd_previews, download=cmd_download, token=cmd_token)[args.cmd](args)


if __name__ == "__main__":
  main()
