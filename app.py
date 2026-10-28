import atexit
import importlib
import math
import re
import json
import os
import shutil
import subprocess
import socket
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid
import webbrowser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import tqdm as _tqdm
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

# faster-whisper is several times faster on a normal CPU. Fall back to openai-whisper if it is not installed.
try:
    from faster_whisper import WhisperModel

    FAST = True
except Exception:
    FAST = False
    import whisper

app = FastAPI()
_models, JOBS = {}, {}
_lock = threading.Lock()
ACTIVE = {"job": None}


# ---- real progress: hook Whisper's internal progress bar -------------------
class _ProgressBar(_tqdm.tqdm):
    def update(self, n=1):
        self._c = getattr(self, "_c", 0) + n
        job = ACTIVE["job"]
        if job and self.total:
            job["pct"] = 10 + min(1.0, self._c / self.total) * job["span"]
        return super().update(n)


class _DownloadBar(_tqdm.tqdm):
    def update(self, n=1):
        self._c = getattr(self, "_c", 0) + n
        job = ACTIVE["job"]
        if job and self.total:
            job["pct"] = 2 + min(1.0, self._c / self.total) * 8
            job["stage"] = f"Downloading model: {self._c / 1048576:.0f} of {self.total / 1048576:.0f} MB"
        return super().update(n)


class _Shim:
    tqdm = _ProgressBar


if not FAST:
    try:
        _tm = importlib.import_module("whisper.transcribe")
        if hasattr(_tm, "tqdm"):
            _tm.tqdm = _Shim
    except Exception:
        pass
    if hasattr(whisper, "tqdm"):
        whisper.tqdm = _DownloadBar


def get_model(name):
    with _lock:
        if name not in _models:
            _models[name] = (
                WhisperModel(name, device="cpu", compute_type="int8") if FAST else whisper.load_model(name)
            )
        return _models[name]


def preload(name="base"):
    print(f"Loading the '{name}' model in the background...", flush=True)
    try:
        get_model(name)
        print(f"Model '{name}' is ready.", flush=True)
    except Exception as e:
        print(f"Could not preload model: {e}", flush=True)


def google_translate(text, tl):
    q = urllib.parse.urlencode({"client": "gtx", "sl": "auto", "tl": tl, "dt": "t", "q": text})
    req = urllib.request.Request(
        "https://translate.googleapis.com/translate_a/single?" + q,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=8) as r:
        data = json.loads(r.read().decode("utf-8"))
    return "".join(p[0] for p in data[0] if p and p[0])


NET = {"fails": 0}


def translate_all(texts, tl, errors, tick):
    def one(t):
        try:
            if not t.strip() or NET["fails"] >= 3:
                return t  # stop retrying when the internet or Google is not reachable
            try:
                out = google_translate(t, tl)
                NET["fails"] = 0
                return out
            except Exception as e:
                NET["fails"] += 1
                errors.add(f"{type(e).__name__}: {e}")
                return t
        finally:
            tick()

    with ThreadPoolExecutor(6) as ex:
        return list(ex.map(one, texts))


STOP = set(
    "a an the and or but if then so of to in on at by for with from as is are was were be been being it its "
    "this that these those i you he she we they me my your our their him her them do does did have has had "
    "not no yes can could will would should may might must about into over after before than too very just "
    "also there here what which who whom whose when where why how all any some each more most other such "
    "only own same up out off again further once between because while during tell give explain mean means".split()
)


def _stem(w):
    for suf in ("ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def _words(text):
    return re.findall(r"\w+", text.lower())


def build_sentences(segs, has_time):
    chunks = [" ".join(x["text"] for x in segs)] if has_time else [x["text"] for x in segs]
    out = []
    for c in chunks:
        out += [p.strip() for p in re.split(r"(?<=[.!?\u0964])\s+", c) if len(p.strip()) > 1]
    return out


def insights(sents):
    toks, surface = [], {}
    for sn in sents:
        t = []
        for w in _words(sn):
            if w in STOP or len(w) < 3:
                continue
            st = _stem(w)
            surface.setdefault(st, Counter())[w] += 1
            t.append(st)
        toks.append(t)
    freq = Counter(w for t in toks for w in t)
    if not freq:
        return {"keywords": [], "points": []}
    mx = max(freq.values())
    scores = [sum(freq[w] / mx for w in t) / (len(t) ** 0.6) if len(t) >= 6 else 0 for t in toks]
    k = min(8, max(3, len(sents) // 6))
    top = sorted(sorted(range(len(sents)), key=lambda i: -scores[i])[:k])
    keywords = [surface[w].most_common(1)[0][0] for w, _ in freq.most_common(12)]
    return {"keywords": keywords, "points": [sents[i] for i in top if scores[i] > 0]}


def run_job(jid, path, name, model, spoken, targets):
    job = JOBS[jid]
    ACTIVE["job"] = job
    try:
        has_time, detected = True, ""
        if name.lower().endswith(".txt"):
            job.update(stage="Reading text file", pct=10)
            raw = open(path, encoding="utf-8", errors="ignore").read()
            segs = [{"start": 0, "end": 0, "text": l.strip()} for l in raw.splitlines() if l.strip()]
            has_time = False
        else:
            span = 60 if targets else 88
            job.update(
                stage="Loading model (first run downloads it, see the terminal)" if FAST
                else "Loading model (the first run downloads it)",
                pct=2,
            )
            tA = time.time()
            m = get_model(model)
            tB = time.time()
            job.update(stage="Transcribing audio", pct=10, span=span)
            if FAST:
                it, info = m.transcribe(path, language=spoken or None, beam_size=1, temperature=0.0, condition_on_previous_text=False)
                detected, segs = info.language, []
                for sg in it:
                    segs.append({"start": sg.start, "end": sg.end, "text": sg.text.strip()})
                    if info.duration:
                        job["pct"] = 10 + min(1.0, sg.end / info.duration) * span
            else:
                res = m.transcribe(path, fp16=False, language=spoken or None, temperature=0.0, condition_on_previous_text=False)
                detected = res.get("language", "en")
                segs = [
                    {"start": x["start"], "end": x["end"], "text": x["text"].strip()}
                    for x in res["segments"]
                ]

        tC = time.time()
        texts = [s["text"] for s in segs]
        sents = build_sentences(segs, has_time)
        job["sents"] = sents
        NET["fails"] = 0
        errors, translations = set(), {}
        todo = [t for t in targets if t["code"].split("-")[0] != detected]
        total, done, lock = max(1, len(todo) * len(texts)), [0], threading.Lock()

        def tick():
            with lock:
                done[0] += 1
                job["pct"] = 70 + 28 * done[0] / total

        for t in targets:
            if t in todo:
                job.update(stage=f"Translating to {t['name']}", pct=max(job["pct"], 70))
                translations[t["name"]] = translate_all(texts, t["code"], errors, tick)
            else:
                translations[t["name"]] = texts

        tD = time.time()
        timing = {"load": round(tB - tA, 1), "transcribe": round(tC - tB, 1), "translate": round(tD - tC, 1)} if not name.lower().endswith(".txt") else None
        if timing:
            print(f"[timing] model {timing['load']}s | transcription {timing['transcribe']}s | translation {timing['translate']}s", flush=True)
        job["result"] = {
            "timing": timing,
            "detected": detected,
            "has_time": has_time,
            "insights": insights(sents),
            "segments": segs,
            "translations": translations,
            "warnings": sorted(errors),
        }
        job.update(stage="Done", pct=100, done=True)
    except Exception as e:
        job.update(error=f"{type(e).__name__}: {e}", done=True)
    finally:
        if name.lower().endswith(".txt"):
            try:
                os.remove(path)
            except OSError:
                pass
        else:
            job["media"] = path  # kept so that subtitles can be burned into a video later


def _drop_media():
    for old in JOBS.values():
        m = old.pop("media", None)
        if m:
            try:
                os.remove(m)
            except OSError:
                pass


atexit.register(_drop_media)
BURNS = {}


def srt_ts(sec):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def has_video(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=60,
        ).stdout
        return "video" in out
    except Exception:
        return path.lower().endswith((".mp4", ".mkv", ".mov", ".avi", ".webm"))


@app.post("/api/burn")
def burn(jid: str = Form(...), lang: str = Form("Original"), bi: str = Form("")):
    j = JOBS.get(jid)
    media = j.get("media") if j else None
    if not j or not j.get("result") or not media or not os.path.exists(media):
        raise HTTPException(404, "The original file is no longer available. Please process it again.")
    segs = j["result"]["segments"]
    tr = j["result"]["translations"].get(lang) if lang != "Original" else None
    work = tempfile.mkdtemp()
    with open(os.path.join(work, "sub.srt"), "w", encoding="utf-8") as f:
        for i, s in enumerate(segs):
            text = s["text"] if tr is None else (s["text"] + "\n" + tr[i] if bi else tr[i])
            f.write(f"{i + 1}\n{srt_ts(s['start'])} --> {srt_ts(s['end'])}\n{text}\n\n")
    vf = "subtitles=sub.srt:force_style='FontName=Nirmala UI,FontSize=22,Outline=2'"
    if has_video(media):
        cmd = ["ffmpeg", "-y", "-i", media, "-vf", vf, "-c:a", "aac", "-movflags", "+faststart", "subtitled.mp4"]
    else:
        cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=1280x720:r=25", "-i", media, "-vf", vf,
               "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "subtitled.mp4"]
    try:
        r = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=3600)
    except FileNotFoundError:
        raise HTTPException(500, "FFmpeg was not found. Install FFmpeg and restart the app.")
    if r.returncode != 0:
        raise HTTPException(500, "FFmpeg failed: " + " ".join(r.stderr.strip().splitlines()[-2:]))
    token = uuid.uuid4().hex[:10]
    BURNS[token] = os.path.join(work, "subtitled.mp4")
    return {"url": f"/api/file/{token}"}


@app.get("/api/file/{token}")
def get_file(token: str):
    path = BURNS.get(token)
    if not path or not os.path.exists(path):
        raise HTTPException(404, "File not found")
    return FileResponse(path, media_type="video/mp4", filename="subtitled.mp4")


@app.post("/api/start")
def start(
    file: UploadFile = File(...),
    model: str = Form("base"),
    spoken: str = Form(""),
    targets: str = Form("[]"),
):
    if model not in ("tiny", "base", "small"):
        model = "base"
    name = file.filename or "file"
    fd, path = tempfile.mkstemp(suffix=os.path.splitext(name)[1] or ".bin")
    os.close(fd)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    _drop_media()
    jid = uuid.uuid4().hex[:8]
    JOBS[jid] = {"stage": "Starting", "pct": 1, "span": 90, "t0": time.time(),
                 "done": False, "error": None, "result": None}
    threading.Thread(
        target=run_job, args=(jid, path, name, model, spoken, json.loads(targets)), daemon=True
    ).start()
    return {"id": jid}


@app.get("/api/status/{jid}")
def status(jid: str):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404, "Unknown job")
    return {
        "stage": j["stage"], "pct": round(j["pct"], 1), "elapsed": time.time() - j["t0"],
        "done": j["done"], "error": j["error"], "result": j["result"] if j["done"] else None,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


PAGE = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Whisper Studio</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap">
<style>
:root{--bg:#090b0a;--card:#121614;--card2:#0e110f;--line:rgba(213,232,195,.14);--txt:#f3f6ef;--mut:#8f9b89;--acc:#d5e8c3;--acc2:#a9cc8d;--on:#0b0f0a;--glow:rgba(213,232,195,.18)}
[data-theme=light]{--bg:#f7f9f3;--card:#fff;--card2:#f1f5ea;--line:rgba(20,40,15,.13);--txt:#101410;--mut:#5d6b57;--acc:#cfe3bb;--acc2:#6f9a52;--on:#0b0f0a;--glow:rgba(111,154,82,.2)}
*{box-sizing:border-box}html,body{margin:0}
body{background:var(--bg);color:var(--txt);font:400 16px/1.55 Inter,system-ui,sans-serif;overflow-x:hidden;transition:background .4s,color .4s}
.bg{position:fixed;inset:0;z-index:-1;overflow:hidden}
.bg i{position:absolute;width:44vw;height:44vw;border-radius:50%;filter:blur(100px);background:var(--acc);opacity:.13;animation:float 20s ease-in-out infinite}
.bg i:nth-child(1){top:-14%;left:-10%}.bg i:nth-child(2){bottom:-22%;right:-10%;animation-delay:-8s}
@keyframes float{50%{transform:translate(6vw,5vh) scale(1.12)}}
header{display:flex;justify-content:space-between;align-items:center;padding:18px 28px;gap:12px;flex-wrap:wrap}
.logo{display:flex;align-items:center;gap:10px;font-weight:700}
.logo span{width:34px;height:34px;border-radius:10px;background:var(--acc);display:grid;place-items:center}
.right{display:flex;gap:10px;align-items:center}
.steps{display:flex;gap:6px;font-size:13px}
.steps b{font-weight:500;color:var(--mut);padding:6px 14px;border-radius:99px;border:1px solid transparent;transition:.3s}
.steps b.on{color:var(--txt);background:var(--card);border-color:var(--line)}.steps b.done{color:var(--acc2)}
.tg{width:38px;height:38px;border-radius:50%;border:1px solid var(--line);background:var(--card);color:var(--txt);cursor:pointer;font-size:16px}
main{max-width:1080px;margin:0 auto;padding:0 22px 50px}
.view{display:none}.view.active{display:block;animation:up .6s cubic-bezier(.2,.8,.2,1)}
@keyframes up{from{opacity:0;transform:translateY(24px)}}
.hero{text-align:center;padding:7vh 0 2vh}
.pill{display:inline-block;font-size:13px;color:var(--acc2);border:1px solid var(--line);background:var(--card);padding:6px 14px;border-radius:99px;margin-bottom:22px}
h1{font-size:clamp(38px,7vw,74px);line-height:1.04;letter-spacing:-.035em;margin:0 0 18px;font-weight:800}
.grad{background:linear-gradient(100deg,var(--txt),var(--acc),var(--acc2),var(--txt));background-size:250% 100%;-webkit-background-clip:text;background-clip:text;color:transparent;animation:shine 7s linear infinite}
@keyframes shine{to{background-position:-250% 0}}
.sub{max-width:600px;margin:0 auto 30px;color:var(--mut);font-size:18px}
.eq{display:flex;justify-content:center;align-items:center;gap:4px;height:90px;margin:34px 0 8px}
.eq i{width:5px;height:100%;border-radius:4px;background:var(--acc);transform-origin:center;animation:eq 1.1s ease-in-out infinite}
@keyframes eq{0%,100%{transform:scaleY(.12)}50%{transform:scaleY(1)}}
.btn{font:inherit;font-weight:600;border:0;border-radius:12px;padding:14px 28px;color:var(--on);background:var(--acc);cursor:pointer;transition:transform .2s,box-shadow .2s;box-shadow:0 8px 30px -10px var(--glow)}
.btn:hover{transform:translateY(-2px);box-shadow:0 14px 36px -8px var(--acc2)}
.btn.ghost{background:var(--card);border:1px solid var(--line);box-shadow:none;color:var(--txt)}
.btn:disabled{opacity:.4;cursor:not-allowed;transform:none}
.feats{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-top:46px}
.feat{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px;animation:up .7s both}
.feat:nth-child(2){animation-delay:.12s}.feat:nth-child(3){animation-delay:.24s}
.feat h3{margin:0 0 6px;font-size:16px}.feat p{margin:0;color:var(--mut);font-size:14px}
h2{font-size:30px;letter-spacing:-.02em;margin:10px 0 4px}.lead{color:var(--mut);margin:0 0 22px}
.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px}
@media(max-width:820px){.grid{grid-template-columns:minmax(0,1fr)}}
.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:20px}
.lab{font-size:12px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);margin:0 0 10px}
.drop{border:1.5px dashed var(--line);border-radius:14px;padding:34px 14px;text-align:center;cursor:pointer;transition:.25s;color:var(--mut)}
.drop:hover,.drop.on{border-color:var(--acc2);background:var(--glow)}
.drop b{display:block;color:var(--txt);font-size:17px;overflow-wrap:anywhere}
.drop svg{margin-bottom:8px;animation:bob 2.4s ease-in-out infinite}@keyframes bob{50%{transform:translateY(-5px)}}
audio,video{width:100%;margin-top:12px;border-radius:12px;max-height:230px;background:#000}
pre.tp{margin:12px 0 0;padding:12px;border-radius:12px;background:var(--card2);color:var(--mut);font:13px/1.5 Inter,sans-serif;max-height:140px;overflow:auto;white-space:pre-wrap}
.seg{display:flex;border:1px solid var(--line);border-radius:12px;overflow:hidden;margin-bottom:18px}
.seg button{flex:1;font:inherit;font-size:14px;color:var(--mut);background:none;border:0;padding:10px;cursor:pointer;transition:.2s}
.seg button+button{border-left:1px solid var(--line)}.seg button.on{background:var(--acc);color:var(--on);font-weight:600}
select,input[type=search]{width:100%;font:inherit;color:var(--txt);background:var(--card2);border:1px solid var(--line);border-radius:12px;padding:11px 12px;margin-bottom:18px}
.chips{display:flex;flex-wrap:wrap;gap:8px;max-height:170px;overflow:auto;padding:2px}
.chip{font:inherit;font-size:13px;color:var(--mut);background:none;border:1px solid var(--line);border-radius:99px;padding:6px 13px;cursor:pointer;transition:.2s}
.chip:hover{border-color:var(--acc2);color:var(--txt)}.chip.on{background:var(--acc);border-color:var(--acc);color:var(--on);font-weight:600}
.row{display:flex;gap:12px;justify-content:space-between;align-items:center;margin-top:22px;flex-wrap:wrap}
.err{display:none;margin-top:14px;padding:12px 14px;border-radius:12px;background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.45);color:#fca5a5;font-size:14px;overflow-wrap:anywhere}
.warn{background:rgba(245,158,11,.1);border-color:rgba(245,158,11,.45);color:#e6b650;display:block;margin:0 0 14px}
.busy{position:fixed;inset:0;display:none;place-items:center;background:color-mix(in srgb,var(--bg) 92%,transparent);backdrop-filter:blur(10px);z-index:9;text-align:center;padding:20px}
.busy.on{display:grid;animation:up .3s}
.bcard{width:min(520px,100%)}.bcard .eq{height:64px;margin:0 0 10px}
.pct{font-size:68px;font-weight:800;letter-spacing:-.04em;line-height:1;font-variant-numeric:tabular-nums}
.track{height:10px;border-radius:99px;background:var(--card);border:1px solid var(--line);margin:18px 0 14px;overflow:hidden}
.track span{display:block;height:100%;width:0;border-radius:99px;background:linear-gradient(90deg,var(--acc2),var(--acc));transition:width .5s ease}
.bcard p{margin:4px 0;color:var(--mut)}#stage{color:var(--txt);font-weight:600}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:18px 0}
.stat{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px;animation:up .6s both}
.stat:nth-child(2){animation-delay:.08s}.stat:nth-child(3){animation-delay:.16s}.stat:nth-child(4){animation-delay:.24s}.stat:nth-child(5){animation-delay:.32s}
.stat b{display:block;font-size:24px;letter-spacing:-.02em}.stat span{color:var(--mut);font-size:13px}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}
.out{min-height:240px;max-height:420px;overflow:auto;white-space:pre-wrap;font-size:17px;line-height:1.75;animation:up .4s}
.out time{color:var(--acc2);font-variant-numeric:tabular-nums;font-size:13px;margin-right:10px}
mark{background:var(--acc);color:var(--on);border-radius:4px;padding:0 2px}
.tools{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px;align-items:center}
.tools label{color:var(--mut);font-size:14px;display:flex;gap:8px;align-items:center;margin-right:auto}
.small{padding:9px 16px;font-size:14px}.srch{margin:0 0 14px}
:focus-visible{outline:2px solid var(--acc2);outline-offset:2px}
.res{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(0,1fr);gap:16px;align-items:start}
@media(max-width:900px){.res{grid-template-columns:minmax(0,1fr)}}
.side{display:flex;flex-direction:column;gap:16px;position:sticky;top:16px}
@media(max-width:900px){.side{position:static}}
#pts{margin:0;padding-left:20px}#pts li{margin:0 0 10px;font-size:15px;cursor:pointer;transition:color .2s}#pts li:hover{color:var(--acc2)}
.askrow{display:flex;gap:8px}.askrow input{margin:0}
.ans{margin-top:12px;padding:12px 14px;border-radius:12px;background:var(--card2);border:1px solid var(--line);font-size:15px}
.ans p{margin:6px 0}.ans .lab{margin:10px 0 4px}.more{color:var(--mut);font-size:14px}
.hint{color:var(--mut);font-size:13px;margin:8px 0 0}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
.demo{font-style:normal;font-size:11px;font-weight:700;letter-spacing:.08em;background:var(--acc);color:var(--on);border-radius:6px;padding:2px 7px;margin-left:8px}
.note{margin:0 0 16px;padding:12px 14px;border-radius:12px;border:1px solid var(--line);background:var(--card);color:var(--mut);font-size:14px}
.screen video{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#000;display:none}
:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
html{scroll-padding-top:env(safe-area-inset-top,0px)}
.out{position:relative}
.ln{padding:6px 10px;border-radius:10px;cursor:pointer;transition:background .2s}.ln:hover{background:var(--card2)}
.ln.on{background:var(--acc);color:var(--on)}.ln.on time{color:var(--on)}.tr2{color:var(--mut);font-size:15px}.ln.on .tr2{color:var(--on)}
.screen{position:relative;aspect-ratio:16/6;min-height:130px;background:#000;border-radius:14px;display:grid;place-items:center;overflow:hidden}
.screen .eq{margin:0;height:62%}.screen .eq.p i{animation-play-state:paused}
.sub{position:absolute;left:0;right:0;bottom:0;text-align:center;color:#fff;font-size:clamp(14px,2.4vw,20px);padding:0 14px 10px;text-shadow:0 1px 3px #000}
.sub:not(:empty){background:linear-gradient(transparent,rgba(0,0,0,.65));padding-top:28px}
.ctl{display:flex;gap:10px;align-items:center;margin-top:12px;flex-wrap:wrap}.ctl input[type=range]{flex:1;min-width:120px;accent-color:var(--acc2);margin:0;padding:0}
.ctl select{width:auto;margin:0;padding:8px}#player{margin-bottom:16px}
.ftabs{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 14px}.pane{margin-bottom:16px}
.chip.bad{background:#ef4444;border-color:#ef4444;color:#fff}
.fc{min-height:150px;display:grid;place-items:center;text-align:center;padding:24px;border:1px solid var(--line);border-radius:16px;background:var(--card2);cursor:pointer}.fct{font-size:22px;font-weight:700}
table{width:100%;border-collapse:collapse;font-size:15px}th,td{text-align:left;padding:10px;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:500;font-size:12px;text-transform:uppercase;letter-spacing:.06em}
.bars{display:flex;align-items:flex-end;gap:6px;height:110px;margin:10px 0}.bars i{flex:1;border-radius:6px 6px 0 0;background:var(--acc2)}.bars i.w{background:#e6b650}
.cf{position:fixed;top:-12px;width:8px;height:14px;z-index:20;animation:fall 2.6s ease-in forwards}@keyframes fall{to{transform:translateY(105vh) rotate(540deg);opacity:.8}}
.ov{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);gap:18px}@media(max-width:820px){.ov{grid-template-columns:minmax(0,1fr)}}
.tl{position:relative;display:flex;height:16px;border-radius:99px;overflow:hidden;gap:3px;cursor:pointer;margin:6px 0 4px}.tl span{background:var(--acc2);opacity:.55;transition:opacity .2s}.tl span:nth-child(even){background:var(--acc)}.tl span:hover{opacity:1}
#ph{position:absolute;top:-3px;bottom:-3px;width:3px;background:var(--txt);border-radius:3px;left:0;pointer-events:none}
.chp{display:flex;gap:12px;padding:10px 12px;border-radius:12px;cursor:pointer;border:1px solid transparent;transition:.2s}.chp:hover{background:var(--card2)}.chp.on{border-color:var(--acc2);background:var(--glow)}
.chp b{display:block}.chp small{color:var(--mut)}.chp time{color:var(--acc2);font-variant-numeric:tabular-nums;font-size:13px;min-width:44px}
.cloud{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;justify-content:center;padding:6px}.wc{font:inherit;background:none;border:0;color:var(--acc2);cursor:pointer;font-weight:700;animation:up .6s both;transition:transform .2s,color .2s}.wc:nth-child(3n){color:var(--txt)}.wc:nth-child(3n+1){color:var(--acc)}.wc:hover{transform:scale(1.15)}
.donut{display:flex;gap:16px;align-items:center}.big{font-size:34px;font-weight:800;letter-spacing:-.03em;line-height:1}
.tld{padding:12px 14px;border-radius:12px;background:var(--card2);border:1px solid var(--line);margin:0 0 10px;font-size:15px}
</style></head><body>
<div class="bg"><i></i><i></i></div>
<header>
  <div class="logo"><span><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0b0f0a" stroke-width="2.6" stroke-linecap="round"><path d="M4 10v4M8 6v12M12 3v18M16 7v10M20 10v4"/></svg></span>Whisper Studio</div>
  <div class="right"><div class="steps"><b id="s1">1 Welcome</b><b id="s2">2 Upload</b><b id="s3">3 Results</b></div><button class="tg" id="tg" title="Switch dark / light">&#9788;</button></div>
</header>
<main>
<section id="v1" class="view active"><div class="hero">
  <div class="pill">Powered by OpenAI Whisper · runs on your computer</div>
  <h1>Speech to text,<br><span class="grad">in every language.</span></h1>
  <p class="sub">Drop in audio, video or a text file. Get a transcript and subtitles, then translate them into dozens of languages.</p>
  <button class="btn" onclick="go(2)">Get started &rarr;</button><div class="eq" id="eq1" aria-hidden="true"></div></div>
  <div class="feats">
    <div class="feat"><h3>Audio, video and text</h3><p>MP3, WAV, M4A, MP4, MKV, MOV. Or upload a .txt file to translate only.</p></div>
    <div class="feat"><h3>Live progress</h3><p>See the exact percentage, elapsed time and time remaining while it works.</p></div>
    <div class="feat"><h3>23 languages</h3><p>Hindi, Punjabi, Urdu, Tamil, Spanish, French, Japanese and more, with .srt subtitles.</p></div>
  </div></section>

<section id="v2" class="view">
  <h2>Upload and configure</h2><p class="lead">Choose a file, pick the accuracy, and select the languages you need.</p>
  <div class="grid">
    <div class="card"><p class="lab">Your file</p>
      <div class="drop" id="drop" tabindex="0" role="button">
        <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4m0 0L7 9m5-5 5 5M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/></svg>
        <b id="fname">Drop a file here</b><span id="fhint">audio, video or .txt · click to browse</span>
        <input type="file" id="file" accept="audio/*,video/*,.txt,.mkv,.mov,.flac,.m4a,.webm" hidden></div>
      <div id="preview"></div><div class="row" style="justify-content:flex-start;margin-top:12px"><button class="btn ghost small" id="rec">&#9679; Record from microphone</button><span class="hint" id="recmsg" style="margin:0"></span></div></div>
    <div class="card">
      <p class="lab">Model</p>
      <div class="seg" id="model"><button data-v="tiny">Fast</button><button data-v="base" class="on">Balanced</button><button data-v="small">Accurate</button></div>
      <p class="lab">Spoken language</p><select id="spoken"></select>
      <p class="lab">Translate into (optional)</p><div class="chips" id="chips"></div></div>
  </div>
  <div class="err" id="err"></div>
  <div class="row"><button class="btn ghost" onclick="go(1)">&larr; Back</button><button class="btn" id="run" disabled>Start</button></div>
</section>

<section id="v3" class="view">
  <h2>Your results</h2><p class="lead" id="meta"></p><div id="warn"></div><div class="stats" id="stats"></div>
  <div class="card" id="player">
    <div class="screen"><div class="eq p" id="eq4" aria-hidden="true"></div><video id="mv" playsinline></video><div class="sub" id="sub"></div></div>
    <div class="ctl"><button class="btn small" id="pp" aria-label="Play">&#9654;</button><input type="range" id="seek" min="0" max="100" value="0" step="0.1" aria-label="Seek">
      <span id="clk" class="hint" style="margin:0">0:00</span><select id="spd" aria-label="Speed"><option value="1">1x</option><option value="2">2x</option><option value="4">4x</option></select><select id="subl" aria-label="Subtitles"></select></div>
    <div class="tools"><button class="btn ghost small" id="mkvid">Create video with subtitles</button><span class="hint" id="vidmsg" style="margin:0"></span></div>
  </div>
  <div class="ftabs" id="ftabs"><button class="chip on" data-f="o">Overview</button><button class="chip" data-f="t">Transcript</button><button class="chip" data-f="s">Study mode</button><button class="chip" data-f="c">Speaking coach</button><button class="chip" data-f="m">Meeting notes</button></div>
  <div id="f_o" class="card pane" hidden></div><div id="f_t" hidden><div class="res"><div class="card">
    <input type="search" class="srch" id="q" placeholder="Search in the text...">
    <div class="tabs" id="tabs"></div><div class="out" id="out"></div>
    <div class="tools"><label id="tsl"><input type="checkbox" id="ts"> Show timestamps</label>
      <label id="bil" style="margin-right:0"><input type="checkbox" id="bi"> Bilingual</label><button class="btn ghost small" id="copy">Copy</button><button class="btn ghost small" id="listen">Listen</button><button class="btn ghost small" id="viewsrt">View .srt</button><button class="btn ghost small" id="dtxt">Download .txt</button>
      <button class="btn ghost small" id="dsrt">Download .srt</button><button class="btn small" id="dall">Download all</button></div>
  </div>
  <aside class="side">
    <div class="card"><p class="lab">Key topics</p><div class="chips" id="kw"></div><p class="hint">Click a topic to find it in the text.</p></div>
    <div class="card"><p class="lab">Key points</p><ol id="pts"></ol><p class="hint">Click a point to find it in the text.</p></div>
    
  </aside></div></div>
  <div id="f_s" class="card pane" hidden></div><div id="f_c" class="card pane" hidden></div><div id="f_m" class="card pane" hidden></div>
  <div class="row"><button class="btn ghost" onclick="go(2)">&larr; Change settings</button><button class="btn" onclick="reset()">New file</button></div>
</section>
</main>

<div class="busy" id="busy"><div class="bcard"><div class="eq" id="eq2"></div>
  <div class="pct" id="pct">0%</div><div class="track"><span id="fill"></span></div>
  <p id="stage">Starting</p><p><span id="el">Elapsed 00:00</span> &nbsp;·&nbsp; <span id="eta">Estimating time left...</span></p></div></div>

<script>
var LANGS=[["English","en","en"],["Hindi","hi","hi"],["Punjabi","pa","pa"],["Urdu","ur","ur"],["Bengali","bn","bn"],["Gujarati","gu","gu"],["Marathi","mr","mr"],["Tamil","ta","ta"],["Telugu","te","te"],["Kannada","kn","kn"],["Malayalam","ml","ml"],["Spanish","es","es"],["French","fr","fr"],["German","de","de"],["Portuguese","pt","pt"],["Italian","it","it"],["Russian","ru","ru"],["Arabic","ar","ar"],["Chinese","zh","zh-CN"],["Japanese","ja","ja"],["Korean","ko","ko"],["Turkish","tr","tr"],["Indonesian","id","id"]];
var $=function(s){return document.querySelector(s)},file=null,model="base",picked={},R=null,tab="Original",urlObj=null;
function go(n){document.querySelectorAll(".view").forEach(function(v){v.classList.remove("active")});$("#v"+n).classList.add("active");
  for(var i=1;i<=3;i++){$("#s"+i).className=i===n?"on":(i<n?"done":"")}window.scrollTo({top:0,behavior:"smooth"})}
function bars(id,n){var el=$(id);for(var i=0;i<n;i++){var b=document.createElement("i");b.style.animationDelay=(-Math.random()*1.1).toFixed(2)+"s";b.style.animationDuration=(.7+Math.random()*.8).toFixed(2)+"s";el.appendChild(b)}}
bars("#eq1",46);bars("#eq2",26);go(1);
$("#tg").onclick=function(){var d=document.documentElement;d.dataset.theme=d.dataset.theme==="dark"?"light":"dark"};

var sp=$("#spoken");sp.innerHTML='<option value="">Auto detect</option>'+LANGS.map(function(l){return'<option value="'+l[1]+'">'+l[0]+"</option>"}).join("");
LANGS.forEach(function(l){var c=document.createElement("button");c.className="chip";c.textContent=l[0];
  c.onclick=function(){if(picked[l[0]]){delete picked[l[0]];c.classList.remove("on")}else{picked[l[0]]=l[2];c.classList.add("on")}};$("#chips").appendChild(c)});
document.querySelectorAll("#model button").forEach(function(b){b.onclick=function(){document.querySelectorAll("#model button").forEach(function(x){x.classList.remove("on")});b.classList.add("on");model=b.dataset.v}});

function isTxt(f){return /\.txt$/i.test(f.name)}
function isVideo(f){return f.type.indexOf("video")===0||/\.(mp4|mkv|mov|webm|avi)$/i.test(f.name)}
function setFile(f){
  if(!f)return;file=f;$("#fname").textContent=f.name;$("#fhint").textContent=(f.size/1048576).toFixed(1)+" MB · click to change";
  if(urlObj)URL.revokeObjectURL(urlObj);var p=$("#preview");p.innerHTML="";
  if(isTxt(f)){var r=new FileReader();r.onload=function(){var pre=document.createElement("pre");pre.className="tp";pre.textContent=String(r.result).slice(0,600);p.appendChild(pre)};r.readAsText(f);$("#run").textContent="Translate text"}
  else{urlObj=URL.createObjectURL(f);p.innerHTML=isVideo(f)?'<video controls src="'+urlObj+'"></video>':'<audio controls src="'+urlObj+'"></audio>';$("#run").textContent="Transcribe & translate"}
  $("#run").disabled=false;$("#err").style.display="none"}
var drop=$("#drop"),inp=$("#file");
drop.onclick=function(){inp.click()};drop.onkeydown=function(e){if(e.key==="Enter")inp.click()};
inp.onchange=function(){setFile(inp.files[0])};
["dragover","dragenter"].forEach(function(t){drop.addEventListener(t,function(e){e.preventDefault();drop.classList.add("on")})});
["dragleave","drop"].forEach(function(t){drop.addEventListener(t,function(e){e.preventDefault();drop.classList.remove("on")})});
drop.addEventListener("drop",function(e){setFile(e.dataTransfer.files[0])});

function mmss(s){s=Math.max(0,Math.round(s));return("0"+Math.floor(s/60)).slice(-2)+":"+("0"+s%60).slice(-2)}
function fail(m){$("#busy").classList.remove("on");var x=$("#err");x.textContent="Something went wrong: "+m;x.style.display="block"}
$("#run").onclick=function(){
  var t=Object.keys(picked).map(function(n){return{name:n,code:picked[n]}}),fd=new FormData();
  fd.append("file",file);fd.append("model",model);fd.append("spoken",sp.value);fd.append("targets",JSON.stringify(t));
  $("#busy").classList.add("on");$("#err").style.display="none";$("#pct").textContent="0%";$("#fill").style.width="0";
  fetch("/api/start",{method:"POST",body:fd}).then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||"Server error");return j})})
  .then(function(j){poll(j.id)}).catch(function(e){fail(e.message)})};
function poll(id){
  fetch("/api/status/"+id).then(function(r){return r.json()}).then(function(j){
    if(j.error){fail(j.error);return}
    $("#pct").textContent=Math.round(j.pct)+"%";$("#fill").style.width=j.pct+"%";$("#stage").textContent=j.stage;
    $("#el").textContent="Elapsed "+mmss(j.elapsed);
    $("#eta").textContent=j.pct>=100?"Finished":(j.pct>8?"About "+mmss(j.elapsed*(100-j.pct)/j.pct)+" left":"Estimating time left...");
    if(j.done){R=j.result;R.id=id;R.names=Object.keys(R.translations);R.elapsed=j.elapsed;render();$("#busy").classList.remove("on");go(3);openF("o")}
    else setTimeout(function(){poll(id)},600)}).catch(function(e){fail(e.message)})}

function texts(n){return n==="Original"?R.segments.map(function(s){return s.text}):R.translations[n]}
function tm(s){return mmss(s)}
function srtT(s){var ms=Math.round(s*1000),h=Math.floor(ms/36e5);ms-=h*36e5;var m=Math.floor(ms/6e4);ms-=m*6e4;var x=Math.floor(ms/1e3);ms-=x*1e3;return[h,m,x].map(function(v){return("0"+v).slice(-2)}).join(":")+","+("00"+ms).slice(-3)}
function esc(s){return s.replace(/[&<>]/g,function(c){return{"&":"&amp;","<":"&lt;",">":"&gt;"}[c]})}
function hl(s){var e=esc(s),q=$("#q").value.trim();if(!q)return e;return e.replace(new RegExp(esc(q).replace(/[.*+?^${}()|[\]\\]/g,"\\$&"),"gi"),function(m){return"<mark>"+m+"</mark>"})}
function render(){
  var d=LANGS.filter(function(l){return l[1]===R.detected})[0],dn=R.has_time?(d?d[0]:R.detected.toUpperCase()):"Text file";
  var words=R.segments.map(function(s){return s.text}).join(" ").split(/\s+/).filter(Boolean).length;
  var dur=R.has_time&&R.segments.length?R.segments[R.segments.length-1].end:0;
  $("#meta").textContent="File: "+file.name+" · finished in "+mmss(R.elapsed)+(R.timing?" (model "+R.timing.load+"s, transcription "+R.timing.transcribe+"s, translation "+R.timing.translate+"s)":"");
  $("#stats").innerHTML=[[dn,"Language"],[words,"Words"],[R.has_time?mmss(dur):"-","Duration"],[R.has_time&&dur>5?Math.round(words/(dur/60)):"-","Words / minute"],[R.names.length,"Translations"]].map(function(a){return'<div class="stat"><b>'+a[0]+"</b><span>"+a[1]+"</span></div>"}).join("");
  $("#warn").innerHTML=R.warnings.length?'<div class="err warn">Some lines could not be translated. Check your internet connection.<br>Details: '+esc(R.warnings[0])+"</div>":"";
  $("#tsl").style.display=$("#dsrt").style.display=R.has_time?"":"none";$("#ts").checked=false;$("#q").value="";
  var tabs=$("#tabs");tabs.innerHTML="";tab="Original";
  ["Original"].concat(R.names).forEach(function(n){var b=document.createElement("button");b.className="chip"+(n===tab?" on":"");b.textContent=n;
    b.onclick=function(){tab=n;tabs.querySelectorAll(".chip").forEach(function(x){x.classList.toggle("on",x===b)});show()};tabs.appendChild(b)});
  show();renderInsights()}
function show(){var t=texts(tab),o=$("#out");o.style.animation="none";void o.offsetWidth;o.style.animation="";
  o.innerHTML=$("#ts").checked&&R.has_time?t.map(function(x,i){return"<div><time>"+tm(R.segments[i].start)+"</time>"+hl(x)+"</div>"}).join(""):hl(t.join(R.has_time?" ":"\n"))}
$("#ts").onchange=show;$("#q").oninput=show;
function escRe(s){return s.replace(/[.*+?^${}()|[\]\\]/g,"\\$&")}
function renderInsights(){
  var I=R.insights||{keywords:[],points:[]},kw=$("#kw");kw.innerHTML="";
  I.keywords.forEach(function(k){var b=document.createElement("button");b.className="chip";b.textContent=k;b.onclick=function(){$("#ts").checked=false;$("#q").value=k;show();var m=document.querySelector("#out mark");if(m)m.scrollIntoView({block:"center",behavior:"smooth"})};kw.appendChild(b)});
  $("#pts").innerHTML=I.points.map(function(p){return"<li>"+esc(p)+"</li>"}).join("")||'<p class="hint">Not enough text to pick key points.</p>';
  document.querySelectorAll("#pts li").forEach(function(li,i){li.onclick=function(){jump(I.points[i])}});
  
}
function jump(t){
  tab="Original";$("#tabs").querySelectorAll(".chip").forEach(function(x,i){x.classList.toggle("on",i===0)});
  $("#ts").checked=false;$("#q").value=t;show();var m=document.querySelector("#out mark");if(m)m.scrollIntoView({block:"center",behavior:"smooth"});
}

function save(n,b){var a=document.createElement("a");a.href=URL.createObjectURL(new Blob([b],{type:"text/plain;charset=utf-8"}));a.download=n;document.body.appendChild(a);a.click();a.remove()}
function base(){return file.name.replace(/\.[^.]+$/,"")}
function toTxt(n){return texts(n).join(R.has_time?" ":"\n")}
function toSrt(n){var t=texts(n),o=texts("Original"),bi=$("#bi").checked&&n!=="Original";return R.segments.map(function(s,i){return(i+1)+"\n"+srtT(s.start)+" --> "+srtT(s.end)+"\n"+(bi?o[i]+"\n"+t[i]:t[i])+"\n"}).join("\n")}
$("#copy").onclick=function(){var b=this;try{navigator.clipboard.writeText(toTxt(tab)).then(function(){b.textContent="Copied"})}catch(e){b.textContent="Select and copy"}setTimeout(function(){b.textContent="Copy"},1400)};
$("#dtxt").onclick=function(){save(base()+"_"+tab.toLowerCase()+".txt",toTxt(tab))};
$("#dsrt").onclick=function(){save(base()+"_"+tab.toLowerCase()+".srt",toSrt(tab))};
$("#dall").onclick=function(){var all=["Original"].concat(R.names),i=0;(function nx(){if(i>=all.length)return;var n=all[i++];save(base()+"_"+n.toLowerCase()+".txt",toTxt(n));if(R.has_time)save(base()+"_"+n.toLowerCase()+".srt",toSrt(n));setTimeout(nx,350)})()};
function reset(){file=null;R=null;inp.value="";$("#fname").textContent="Drop a file here";$("#fhint").textContent="audio, video or .txt · click to browse";$("#preview").innerHTML="";$("#run").disabled=true;$("#run").textContent="Start";go(2)}
if(window.matchMedia&&matchMedia("(prefers-color-scheme: light)").matches)document.documentElement.dataset.theme="light";
var SW=new Set("a an the and or but if then so of to in on at by for with from as is are was were be been it its this that these those i you he she we they do does did have has had not no can could will would should about into than very also there here what which who when where why how all any some each more most other only tell give explain mean means".split(" "));
function stm(w){if(/(ing|ed)$/.test(w)&&w.length-(w.endsWith("ing")?3:2)>=4)return w.replace(/(ing|ed)$/,"");if(w.endsWith("s")&&!w.endsWith("ss")&&w.length>3)return w.slice(0,-1);return w}
function sents(){return R.segments.map(function(s){return s.text}).join(" ").split(/(?<=[.!?])\s+/).filter(function(s){return s.length>1})}
var T=0,D=1,PL=false,SP=1,tick=null;
function subtxt(){var m=$("#subl").value,i=R.segments.findIndex(function(s){return T>=s.start&&T<s.end});if(i<0||m==="off"||(m!=="Original"&&m!=="both"&&!R.translations[m]))return"";var o=R.segments[i].text;return m==="Original"?o:m==="both"?o+"<br>"+esc(texts(R.names[0])[i]):esc(texts(m)[i])}
function upd(){$("#seek").value=T/D*100;$("#clk").textContent=mmss(T)+" / "+mmss(D);$("#sub").innerHTML=subtxt();var a=null;
  document.querySelectorAll("#out .ln").forEach(function(l,i){var on=T>=R.segments[i].start&&T<R.segments[i].end;l.classList.toggle("on",on);if(on)a=l});
  if(a&&PL){var o=$("#out");o.scrollTop=a.offsetTop-o.clientHeight/3}}
function setPlay(b){PL=b;$("#pp").innerHTML=b?"&#10074;&#10074;":"&#9654;";$("#eq4").classList.toggle("p",!b);clearInterval(tick);var mv=$("#mv");
  if(b){mv.playbackRate=SP;var pr=mv.play();if(pr&&pr.catch)pr.catch(function(){});tick=setInterval(function(){T=mv.currentTime;if(mv.ended||T>=D)setPlay(false);upd()},100)}else mv.pause()}
function seek(t){T=Math.max(0,Math.min(D,t));try{$("#mv").currentTime=T}catch(e){}upd()}
$("#pp").onclick=function(){if(T>=D)seek(0);setPlay(!PL)};$("#seek").oninput=function(){seek(this.value/100*D)};$("#spd").onchange=function(){SP=parseFloat(this.value);$("#mv").playbackRate=SP};$("#subl").onchange=upd;
$("#mkvid").onclick=function(){var b=this,m=$("#subl").value,fd=new FormData();if(b.disabled)return;fd.append("jid",R.id);fd.append("lang",(m==="off"||m==="Original")?"Original":(m==="both"?R.names[0]:m));if(m==="both")fd.append("bi","1");
  b.disabled=true;b.textContent="Rendering video...";$("#vidmsg").textContent="This can take a minute.";
  fetch("/api/burn",{method:"POST",body:fd}).then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.detail||"Error");return j})})
  .then(function(j){$("#vidmsg").innerHTML='<a href="'+j.url+'" download style="color:var(--acc2);font-weight:600">Download the video with subtitles</a>'})
  .catch(function(e){$("#vidmsg").textContent="Could not create the video: "+e.message}).then(function(){b.disabled=false;b.textContent="Create video with subtitles"})};
show=function(){var t=texts(tab),bil=$("#bi").checked&&R.names[0],ts=$("#ts").checked,og=texts("Original"),tn=tab==="Original"?R.names[0]:tab;
  $("#out").innerHTML=R.segments.map(function(s,i){var a=bil?hl(og[i])+'<div class="tr2">'+esc(texts(tn)[i])+"</div>":hl(t[i]);return'<div class="ln" data-i="'+i+'">'+(ts?"<time>"+tm(s.start)+"</time>":"")+a+"</div>"}).join("");upd()};
$("#ts").onchange=show;$("#q").oninput=show;$("#bi").onchange=show;
$("#out").onclick=function(e){var l=e.target.closest&&e.target.closest(".ln");if(l){seek(R.segments[+l.dataset.i].start);setPlay(true)}};
var _r=render;render=function(){
  if(!R.segments.length){R.segments=[{start:0,end:1,text:"(No speech detected)"}];R.names.forEach(function(n){R.translations[n]=["(No speech detected)"]})}
  if(!R.has_time)R.segments.forEach(function(s,i){s.start=i;s.end=i+1});
  D=R.segments[R.segments.length-1].end;T=0;setPlay(false);
  var mv=$("#mv"),hasM=!!file&&!isTxt(file)&&!!urlObj;$("#player").style.display=hasM?"":"none";mv.removeAttribute("src");
  if(hasM){mv.src=urlObj;mv.style.display=isVideo(file)?"block":"none"}else mv.style.display="none";
  document.querySelector('#ftabs [data-f="c"]').style.display=R.has_time?"":"none";
  $("#subl").innerHTML='<option value="off">Subtitles off</option><option value="Original" selected>Original</option>'+R.names.map(function(n){return"<option>"+n+"</option>"}).join("")+(R.names[0]?'<option value="both">Both</option>':"");
  $("#bi").checked=false;$("#bi").disabled=!R.names[0];$("#vidmsg").textContent="";_r()};
$("#listen").onclick=function(){var b=this;if(!window.speechSynthesis){b.textContent="Not supported";return}if(speechSynthesis.speaking){speechSynthesis.cancel();b.textContent="Listen";return}
  var u=new SpeechSynthesisUtterance(texts(tab).join(" "));u.lang=tab==="Original"?(R.detected||"en"):((LANGS.filter(function(l){return l[0]===tab})[0]||[])[2]||"en");u.rate=.95;u.onend=function(){b.textContent="Listen"};speechSynthesis.speak(u);b.textContent="Stop"};
$("#viewsrt").onclick=function(){var t=texts(tab),b=tab!=="Original",o=texts("Original");
  $("#out").innerHTML='<pre class="tp" style="max-height:none">'+esc(R.segments.map(function(g,i){return(i+1)+"\n"+srtT(g.start)+" --> "+srtT(g.end)+"\n"+(b?o[i]+"\n"+t[i]:t[i])+"\n"}).join("\n"))+"</pre>"};
function openF(f){["o","t","s","c","m"].forEach(function(k){$("#f_"+k).hidden=k!==f});document.querySelectorAll("#ftabs .chip").forEach(function(b){b.classList.toggle("on",b.dataset.f===f)});
  if(f==="o")overview();if(f==="s")study();if(f==="c")coach();if(f==="m")meet()}
document.querySelectorAll("#ftabs .chip").forEach(function(b){b.onclick=function(){openF(b.dataset.f)}});
function rx(k){return new RegExp("\\b"+k+"\\b","i")}
function study(){var P=$("#f_s"),ss=sents(),kw=R.insights.keywords,qs=[],sc=0,done=0;
  ss.forEach(function(s){if(qs.length>=6||s.length<40||s.length>170)return;var k=kw.find(function(w){return rx(w).test(s)});if(!k||qs.some(function(x){return x.k===k}))return;
    var o=[k].concat(kw.filter(function(w){return w!==k}).sort(function(){return Math.random()-.5}).slice(0,3)).sort(function(){return Math.random()-.5});qs.push({s:s.replace(rx(k),"_____"),k:k,o:o})});
  P.innerHTML='<div class="row" style="margin:0 0 12px"><div class="tabs" style="margin:0"><button class="chip on" id="sq">Quiz</button><button class="chip" id="sf">Flashcards</button></div><b id="score">Score: 0 / '+qs.length+"</b></div><div id=\"sbody\"></div>";
  function quiz(){$("#sbody").innerHTML=qs.map(function(q,i){return'<div class="ans" data-i="'+i+'"><p>'+(i+1)+". "+esc(q.s)+'</p><div class="chips">'+q.o.map(function(o){return'<button class="chip">'+esc(o)+"</button>"}).join("")+"</div></div>"}).join("")||'<p class="hint">Not enough text for a quiz.</p>';
    document.querySelectorAll("#sbody .ans").forEach(function(box){var q=qs[+box.dataset.i],got=false;box.querySelectorAll(".chip").forEach(function(b){b.onclick=function(){if(got)return;got=true;done++;var ok=b.textContent===q.k;if(ok)sc++;b.classList.add(ok?"on":"bad");
      if(!ok)box.querySelectorAll(".chip").forEach(function(x){if(x.textContent===q.k)x.classList.add("on")});$("#score").textContent="Score: "+sc+" / "+qs.length+(done===qs.length?(sc===qs.length?" · Perfect!":" · Finished"):"");if(done===qs.length&&sc===qs.length)confetti()}})})}
  function cards(){var i=0,fl=false,cs=kw.map(function(k){return{k:k,s:ss.find(function(x){return rx(k).test(x)})}}).filter(function(c){return c.s});
    function draw(){var c=cs[i];$("#sbody").innerHTML='<div class="fc" id="fc"><div><span class="hint">'+(fl?"Answer":"Term")+" · card "+(i+1)+" of "+cs.length+'</span><div class="fct">'+(fl?esc(cs[i].s):esc(cs[i].k))+'</div></div></div><div class="row"><button class="btn ghost small" id="pv">Previous</button><button class="btn small" id="nx">Next</button></div>';
      $("#fc").onclick=function(){fl=!fl;draw()};$("#pv").onclick=function(){i=(i+cs.length-1)%cs.length;fl=false;draw()};$("#nx").onclick=function(){i=(i+1)%cs.length;fl=false;draw()}}draw()}
  $("#sq").onclick=function(){this.classList.add("on");$("#sf").classList.remove("on");quiz()};$("#sf").onclick=function(){this.classList.add("on");$("#sq").classList.remove("on");cards()};quiz()}
function coach(){var P=$("#f_c"),S=R.segments,txt=S.map(function(s){return s.text}).join(" "),W=txt.split(/\s+/).filter(Boolean).length,wpm=Math.round(W/(D/60)),FR=/\b(um+|uh+|erm|like|you know|basically|actually)\b/gi,fl=txt.match(FR)||[],gaps=[],cnt={};
  for(var i=1;i<S.length;i++){var g=S[i].start-S[i-1].end;if(g>=1.5)gaps.push(g)}fl.forEach(function(f){f=f.toLowerCase();cnt[f]=(cnt[f]||0)+1});
  var pace=S.map(function(s){return Math.round(s.text.split(/\s+/).length/((s.end-s.start)/60))}),score=Math.max(0,Math.round(100-fl.length*3-gaps.length*4-Math.min(25,Math.abs(wpm-140)/2))),
  fs=Object.keys(cnt).map(function(k){return k+" x"+cnt[k]}).join(", "),tips=[];
  if(fl.length)tips.push("You used "+fl.length+" filler words ("+fs+"). Try a silent pause instead.");if(gaps.length)tips.push(gaps.length+" long pauses, the longest was "+Math.max.apply(null,gaps).toFixed(1)+" seconds. Plan your transitions.");
  tips.push(wpm<110?"Your pace is a little slow. Aim for 120 to 160 words per minute.":wpm>170?"Your pace is fast. Slow down so listeners can follow.":"Your pace is in the comfortable range.");
  P.innerHTML='<div class="stats"><div class="stat"><b>'+score+'</b><span>Speaking score</span></div><div class="stat"><b>'+wpm+'</b><span>Words per minute</span></div><div class="stat"><b>'+fl.length+'</b><span>Filler words</span></div><div class="stat"><b>'+gaps.length+'</b><span>Long pauses</span></div></div>'+
   '<p class="lab">Pace per sentence (words per minute)</p><div class="bars">'+pace.map(function(v){return'<i class="'+(v<110||v>170?"w":"")+'" style="height:'+Math.min(100,v/2)+'%" title="'+v+' wpm"></i>'}).join("")+'</div>'+
   '<p class="lab">Tips</p><ul>'+tips.map(function(t){return"<li>"+esc(t)+"</li>"}).join("")+'</ul><p class="lab">Your speech, fillers highlighted</p><p>'+esc(txt).replace(FR,"<mark>$&</mark>")+'</p><p class="hint">In the real app you can record from the microphone. Here a sample speech is analysed.</p>'}
function meet(){var P=$("#f_m"),S=R.segments,AC=/(\bI will\b|\bwe will\b|\bplease\b|\bshould we\b|\bneed to\b|\bmust\b|for homework|\bcan fix\b)/i,DEC=/(let us book|targeting|decided|agreed|to summarize|design fixes)/i,DUE=/\b((?:mon|tues|wednes|thurs|fri|satur|sun)day|tomorrow|this week|next (?:week|month|class))\b/i,ac=[],de=[];
  S.forEach(function(s){s.text.split(/(?<=[.!?])\s+/).forEach(function(x){var due=(x.match(DUE)||[])[1]||"-",who=/\b(I will|I'll)\b/i.test(x)?(s.sp||"Speaker"):"Team";if(AC.test(x))ac.push({a:x,w:who,d:due,t:s.start});else if(DEC.test(x))de.push({a:x,t:s.start})})});
  P.innerHTML='<p class="lab">Summary</p><p>'+S.length+' turns · '+ac.length+' action items · '+de.length+' decisions. Click a row to jump to that moment.</p><p class="lab">Decisions</p><ul>'+(de.map(function(x){return'<li class="jmp" data-t="'+x.t+'">'+esc(x.a)+"</li>"}).join("")||"<li>None found</li>")+'</ul><p class="lab">Action items</p><div style="overflow-x:auto"><table><tr><th>Action</th><th>Owner</th><th>Due</th></tr>'+
   (ac.map(function(x){return'<tr class="jmp" data-t="'+x.t+'" style="cursor:pointer"><td>'+esc(x.a)+"</td><td>"+esc(x.w)+"</td><td>"+esc(x.d)+"</td></tr>"}).join("")||'<tr><td colspan="3">None found</td></tr>')+"</table></div>";
  P.querySelectorAll(".jmp").forEach(function(e){e.onclick=function(){openF("t");seek(+e.dataset.t);setPlay(true);window.scrollTo({top:0,behavior:"smooth"})}})}
bars("#eq4",44);
function chunks(){var S=R.segments,k=S.length>=8?4:S.length>=4?2:1,sz=Math.ceil(S.length/k),o=[];for(var i=0;i<S.length;i+=sz)o.push(S.slice(i,i+sz));return o}
function topw(txt,n){var f={},sf={};(txt.toLowerCase().match(/\w+/g)||[]).forEach(function(w){if(SW.has(w)||w.length<4)return;var k=stm(w);f[k]=(f[k]||0)+1;if(!sf[k])sf[k]=w});
  return Object.keys(f).sort(function(a,b){return f[b]-f[a]}).slice(0,n).map(function(k){return sf[k].charAt(0).toUpperCase()+sf[k].slice(1)})}
function overview(){var P=$("#f_o"),S=R.segments,txt=S.map(function(s){return s.text}).join(" "),W=txt.split(/\s+/).filter(Boolean).length,sp=S.reduce(function(a,s){return a+s.end-s.start},0),pc=Math.round(sp/D*100),
  pts=R.insights.points.slice(0,3),rs=Math.round(pts.join(" ").split(/\s+/).length/230*60),cl=2*Math.PI*40,ch=chunks(),
  cnt=R.insights.keywords.map(function(k){return[k,(txt.match(new RegExp("\\b"+k,"gi"))||[]).length]}),mn=Math.min.apply(null,cnt.map(function(c){return c[1]})),mx=Math.max.apply(null,cnt.map(function(c){return c[1]}));
  P.innerHTML='<div class="ov"><div><p class="lab">In short</p>'+pts.map(function(x){return'<div class="tld">'+esc(x)+"</div>"}).join("")+(R.has_time?'<p class="hint">Listening takes '+mmss(D)+". Reading this summary takes about "+mmss(rs)+".</p>":"")+
   '<p class="lab" style="margin-top:18px">Chapters (click to jump)</p><div class="tl" id="tl">'+ch.map(function(c){return'<span style="width:'+((c[c.length-1].end-c[0].start)/D*100)+'%" title="'+esc(topw(c.map(function(s){return s.text}).join(" "),2).join(" & "))+'"></span>'}).join("")+'<div id="ph"></div></div>'+
   ch.map(function(c,i){var a=c[0].start,b=c[c.length-1].end;return'<div class="chp" data-a="'+a+'" data-b="'+b+'"><time>'+mmss(a)+"</time><div><b>"+esc(topw(c.map(function(s){return s.text}).join(" "),2).join(" & ")||"Chapter "+(i+1))+"</b><small>"+esc(c[0].text.slice(0,95))+"</small></div></div>"}).join("")+"</div>"+
   '<div><p class="lab">Talk time</p><div class="donut"><svg viewBox="0 0 100 100" width="120" height="120"><circle cx="50" cy="50" r="40" fill="none" stroke="var(--line)" stroke-width="12"/><circle cx="50" cy="50" r="40" fill="none" stroke="var(--acc2)" stroke-width="12" stroke-linecap="round" stroke-dasharray="'+(pc/100*cl)+" "+cl+'" transform="rotate(-90 50 50)"/><text x="50" y="56" text-anchor="middle" fill="currentColor" font-size="19" font-weight="700">'+pc+'%</text></svg>'+
   '<div><div class="big">'+W+'</div><span class="hint">words spoken</span><p class="hint">'+(100-pc)+'% silence</p></div></div><p class="lab" style="margin-top:18px">Topic cloud</p><div class="cloud">'+
   cnt.map(function(c,i){return'<button class="wc" style="font-size:'+(15+(c[1]-mn)/((mx-mn)||1)*22)+"px;animation-delay:"+i*.07+'s">'+esc(c[0])+"</button>"}).join("")+"</div></div></div>";
  if(!R.has_time){P.querySelectorAll(".donut,.chp time,#tl").forEach(function(x){x.style.display="none"})}P.querySelectorAll(".chp").forEach(function(e){e.onclick=function(){seek(+e.dataset.a);setPlay(true)}});
  $("#tl").onclick=function(e){var r=this.getBoundingClientRect();seek((e.clientX-r.left)/r.width*D)};
  P.querySelectorAll(".wc").forEach(function(b){b.onclick=function(){openF("t");$("#ts").checked=false;$("#q").value=b.textContent;show();var m=document.querySelector("#out mark");if(m)m.scrollIntoView({block:"center",behavior:"smooth"})}});chap()}
function chap(){var ph=$("#ph");if(ph)ph.style.left=(T/D*100)+"%";document.querySelectorAll(".chp").forEach(function(c){c.classList.toggle("on",T>=+c.dataset.a&&T<+c.dataset.b)})}
var _u=upd;upd=function(){_u();chap()};
function confetti(){for(var i=0;i<46;i++){var c=document.createElement("i");c.className="cf";c.style.cssText="left:"+Math.random()*100+"vw;background:"+["#d5e8c3","#a9cc8d","#f3f6ef","#e6b650"][i%4]+";animation-delay:"+Math.random()*.4+"s";document.body.appendChild(c);setTimeout(function(x){x.remove()},3300,c)}}
document.addEventListener("keydown",function(e){if(!$("#v3").classList.contains("active")||/INPUT|SELECT|TEXTAREA|BUTTON/.test(e.target.tagName))return;
  if(e.code==="Space"){e.preventDefault();$("#pp").click()}else if(e.key==="ArrowRight")seek(T+5);else if(e.key==="ArrowLeft")seek(T-5)});
$("#vidmsg").insertAdjacentHTML("afterend",'<span class="hint" style="margin:0 0 0 auto">Shortcuts: Space play/pause · Left and Right arrows seek 5 s</span>');
var rec=null,rch=[],rt=null;
$("#rec").onclick=function(){var b=this;if(rec&&rec.state==="recording"){rec.stop();return}
  if(!navigator.mediaDevices||!window.MediaRecorder){$("#recmsg").textContent="Recording is not supported in this browser.";return}
  navigator.mediaDevices.getUserMedia({audio:true}).then(function(st){rch=[];rec=new MediaRecorder(st);var t0=Date.now();
    rec.ondataavailable=function(e){if(e.data.size)rch.push(e.data)};
    rec.onstop=function(){clearInterval(rt);st.getTracks().forEach(function(t){t.stop()});b.innerHTML="&#9679; Record from microphone";var bl=new Blob(rch,{type:rec.mimeType||"audio/webm"});setFile(new File([bl],"recording.weba",{type:bl.type}));$("#recmsg").textContent="Recording ready ("+mmss((Date.now()-t0)/1000)+")"};
    rec.start();b.innerHTML="&#9632; Stop recording";rt=setInterval(function(){$("#recmsg").textContent="Recording "+mmss((Date.now()-t0)/1000)},500)}).catch(function(){$("#recmsg").textContent="Microphone permission was blocked."})};
</script></body></html>
"""


if __name__ == "__main__":
    port = 7860
    print("CPU threads:", os.cpu_count(), flush=True)
    print("Engine:", "faster-whisper (fast)" if FAST else "openai-whisper (slower). For more speed run: pip install faster-whisper", flush=True)
    threading.Thread(target=preload, daemon=True).start()
    threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    host = "0.0.0.0" if "--lan" in sys.argv else "127.0.0.1"
    if "--lan" in sys.argv:
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.connect(("8.8.8.8", 80))
            print(f"\nOther devices on the same Wi-Fi can open: http://{probe.getsockname()[0]}:{port}\n", flush=True)
            probe.close()
        except Exception:
            print("Could not detect your IP. Run ipconfig and use your IPv4 address.", flush=True)
    uvicorn.run(app, host=host, port=port, access_log=False)