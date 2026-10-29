"""SmartCam: a CCTV that understands what it sees.

Camera -> motion check -> YOLO detection -> tracking -> rules (visits, loitering, zones, crowds, unattended bags)
-> event log + snapshots -> optional Gemini description. The dashboard is a local web page at http://127.0.0.1:47920.
"""
import json
import logging
import logging.handlers
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import date, timedelta
from urllib.parse import urlparse

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_from_directory

import ai
from detector import Detector
from events import VISIT_CATEGORIES, EventLog, Rules, nice_duration, noun
from tracker import Tracker

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
SNAPS = os.path.join(DATA, "snapshots")
CONFIG = os.path.join(ROOT, "config.json")
MODEL = os.path.join(ROOT, "models", "yolo11n.onnx")
DEMO = os.path.join(ROOT, "samples", "vtest.avi")

STREAM_FPS = 15       # frames a second sent to the dashboard
IDLE_DETECT = 1.5     # seconds between detections when nothing moves (still catches parked cars, sleeping cats)
MAX_WIDTH = 1280      # bigger camera pictures are scaled down to this
WEAK = 0.25           # half-hidden detections down to this score may keep an existing object going

DEFAULTS = {
    "camera": "0",                 # "0" = built-in webcam, "1" = next one, "demo", rtsp://..., http://..., or a file
    "camera_name": "Camera 1",
    "port": 47920,
    "detect_fps": 4,
    "confidence": 0.4,
    "watch": ["person", "vehicle", "animal"],
    "loiter_seconds": 60,
    "crowd_size": 5,
    "unattended_seconds": 30,
    "zones": [],
    "ai_auto": True,
    "ai_gap_seconds": 45,
    "ai_daily_limit": 150,
    "keep_days": 14,
    "open_browser": True,
}
# what the dashboard may change: key -> (type, min, max)
EDITABLE = {"camera": (str, 1, 500), "camera_name": (str, 1, 40), "detect_fps": (float, 0.5, 15),
            "confidence": (float, 0.15, 0.9), "loiter_seconds": (int, 0, 3600), "crowd_size": (int, 0, 100),
            "unattended_seconds": (int, 0, 3600), "ai_auto": (bool, None, None),
            "ai_gap_seconds": (int, 10, 3600), "ai_daily_limit": (int, 0, 2000), "keep_days": (int, 1, 365)}

log = logging.getLogger("smartcam")


# ---------------- settings ----------------
def load_config():
    cfg = dict(DEFAULTS)
    try:
        with open(CONFIG, encoding="utf-8") as f:
            cfg.update(json.load(f))
    except FileNotFoundError:
        pass
    except ValueError as e:
        log.warning("config.json is broken (%s); using defaults", e)
    return cfg


def save_config(cfg):
    out = {k: v for k, v in cfg.items() if not k.startswith("_")}
    if "_saved_camera" in cfg:  # a --camera choice lasts for this run only
        out["camera"] = cfg["_saved_camera"]
    tmp = CONFIG + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, CONFIG)


def clean_zones(zones):
    out = []
    for z in zones if isinstance(zones, list) else []:
        name = str(z.get("name", "")).strip()[:40] if isinstance(z, dict) else ""
        pts = z.get("points") if isinstance(z, dict) else None
        if not name or not isinstance(pts, list) or not 3 <= len(pts) <= 30:
            continue
        try:
            pts = [[min(max(float(p[0]), 0.0), 1.0), min(max(float(p[1]), 0.0), 1.0)] for p in pts]
        except (TypeError, ValueError, IndexError):
            continue
        out.append({"name": name, "points": pts})
    return out


# ---------------- pictures ----------------
BGR = {"person": (132, 220, 61), "vehicle": (255, 163, 77), "animal": (71, 181, 255), "bag": (255, 140, 180),
       "object": (178, 164, 154), "alert": (94, 77, 255)}


def fit(frame, width):
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    return cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)


def jpeg(frame, quality=75):
    return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def _text(img, txt, org, scale, color, thick=1, bg=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(txt, font, scale, thick)
    x, y = org
    if bg is not None:
        cv2.rectangle(img, (x, y - th - base - 4), (x + tw + 8, y + 2), bg, -1)
        cv2.putText(img, txt, (x + 4, y - base + 1), font, scale, color, thick, cv2.LINE_AA)
    else:
        cv2.putText(img, txt, (x + 1, y + 1), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(img, txt, (x, y), font, scale, color, thick, cv2.LINE_AA)


def alerting(t):
    return bool(t.zones) or "loiter" in t.flags or "alone" in t.flags


def draw(frame, tracks, zones, now, name):
    img = frame.copy()
    h, w = img.shape[:2]
    s = w / 1280
    if zones:
        layer = img.copy()
        for z in zones:
            pts = (np.array(z["points"]) * [w, h]).astype(np.int32)
            cv2.fillPoly(layer, [pts], BGR["alert"])
        img = cv2.addWeighted(layer, 0.16, img, 0.84, 0)
        for z in zones:
            pts = (np.array(z["points"]) * [w, h]).astype(np.int32)
            cv2.polylines(img, [pts], True, BGR["alert"], max(1, round(2 * s)), cv2.LINE_AA)
            top = pts[pts[:, 1].argmin()]
            _text(img, z["name"], (int(top[0]), max(int(top[1]) - 6, 20)), 0.5 * s + 0.1, (255, 255, 255), 1,
                  bg=BGR["alert"])
    for t in sorted(tracks, key=lambda t: t.category != "object"):  # important boxes drawn on top
        # between detections, slide the box along with the object (at most half a second ahead)
        box = t.box if t.category == "object" else t.predicted(min(now, t.last_seen + 0.5))
        x1, y1, x2, y2 = map(int, box)
        if t.category == "object":
            cv2.rectangle(img, (x1, y1), (x2, y2), BGR["object"], 1, cv2.LINE_AA)
            _text(img, t.label, (x1 + 2, y1 + 14), 0.4 * s + 0.05, BGR["object"])
            continue
        col = BGR["alert"] if alerting(t) else BGR[t.category]
        if t.category in ("person", "vehicle") and len(t.trail) > 1:
            cv2.polylines(img, [np.array(t.trail, np.int32)], False, col, 1, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, max(2, round(2 * s)), cv2.LINE_AA)
        tag = "{} {}  {}".format(noun(t.label), t.id, nice_duration(t.dwell(now)))
        _text(img, tag, (x1, max(y1, 22)), 0.45 * s + 0.08, (20, 20, 20), 1, bg=col)
    stamp = "{}   {}".format(name.upper(), time.strftime("%Y-%m-%d  %H:%M:%S", time.localtime(now)))
    _text(img, stamp, (round(12 * s), round(34 * s) + 6), 0.55 * s + 0.12, (255, 255, 255), 1, bg=(40, 22, 12))
    return img


def placeholder(title, detail):
    img = np.zeros((720, 1280, 3), np.uint8)
    img[:] = (40, 22, 12)
    _text(img, title, (60, 330), 1.6, (255, 255, 255), 3)
    _text(img, detail[:90], (62, 390), 0.7, (200, 190, 170), 1)
    _text(img, time.strftime("%Y-%m-%d  %H:%M:%S"), (62, 430), 0.6, (150, 140, 120), 1)
    return img


class Motion:
    """Cheap change detector on a tiny grey copy of each frame; decides when YOLO needs to look closely."""

    def __init__(self):
        self.bg = None
        self.level = 0.0

    def update(self, frame):
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
        g = g.astype(np.float32)
        if self.bg is None or self.bg.shape != g.shape:
            self.bg = g
            return False
        changed = float((cv2.absdiff(g, self.bg) > 18).mean())
        cv2.accumulateWeighted(g, self.bg, 0.05)
        self.level = 0.7 * self.level + 0.3 * changed
        return changed > 0.003


# ---------------- camera ----------------
class Camera(threading.Thread):
    """Reads a webcam, IP camera (RTSP/HTTP) or video file, keeping only the newest frame."""

    def __init__(self, source):
        super().__init__(daemon=True)
        self.source = str(source).strip()
        self.status, self.message = "connecting", "Connecting to the camera..."
        self.frame, self.seq = None, 0
        self.cond = threading.Condition()
        self.stopped = threading.Event()

    def _open(self):
        src = self.source
        if src.isdigit():
            cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW)  # DirectShow opens laptop webcams quickly
            if not cap.isOpened():
                cap = cv2.VideoCapture(int(src))
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap, False
        path = DEMO if src.lower() == "demo" else src
        if os.path.isfile(path):
            return cv2.VideoCapture(path), True
        if "://" not in path:
            self.message = "File not found: " + path
            return None, False
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")  # fewer grey smears on Wi-Fi
        return cv2.VideoCapture(path, cv2.CAP_FFMPEG), False

    def run(self):
        backoff = 1.0
        while not self.stopped.is_set():
            self.status = "connecting"
            cap, is_file = self._open()
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                    self.message = ("Camera {} not found, or another app (Teams, Zoom, Camera) is using it."
                                    .format(self.source) if self.source.isdigit()
                                    else "Can't connect to " + self.source)
                self.status = "offline"
                self.stopped.wait(backoff)
                backoff = min(backoff * 2, 15)
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) if is_file else 0
            gap = 1 / fps if 1 <= fps <= 120 else 1 / 25
            next_t, fails = time.time(), 0
            while not self.stopped.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    if is_file and fails == 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # a video file loops forever
                        fails = 1
                        continue
                    fails += 1
                    if fails > 30:
                        break
                    time.sleep(0.05)
                    continue
                fails, backoff = 0, 1.0
                self.status, self.message = "live", ""
                with self.cond:
                    self.frame = fit(frame, MAX_WIDTH)
                    self.seq += 1
                    self.cond.notify_all()
                if is_file:  # play at the file's own speed
                    next_t += gap
                    lag = next_t - time.time()
                    if lag > 0:
                        time.sleep(lag)
                    elif lag < -1:
                        next_t = time.time()
            cap.release()
            if not self.stopped.is_set():
                self.status, self.message = "offline", "Lost the camera picture. Reconnecting..."
                self.stopped.wait(backoff)

    def latest(self, after_seq, timeout=1.0):
        with self.cond:
            if self.seq <= after_seq:
                self.cond.wait(timeout)
            return (self.frame, self.seq) if self.seq > after_seq else (None, after_seq)

    def stop(self):
        self.stopped.set()


# ---------------- the brain ----------------
class Engine(threading.Thread):
    def __init__(self, cfg, events):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.log = events
        self.detector = Detector(MODEL)
        self.tracker = Tracker()
        self.rules = Rules(cfg)
        self.motion = Motion()
        self.camera = None
        self.cam_lock = threading.Lock()
        self.jcond = threading.Condition()
        self.jpeg, self.jseq = jpeg(placeholder("STARTING", "Loading...")), 1
        self.lock = threading.Lock()  # guards the tracker and everything below that both loops touch
        self.raw = self.annotated = None
        self.visible = []
        self.moving = False
        self.pending = None           # newest (frame, time) waiting for the detector
        self.pcond = threading.Condition()
        self.fps = 0.0
        self.det_ms = 0.0
        self.narration = deque(maxlen=12)
        self.ai_busy = False
        self.ai_error = ""
        self.ai_last = 0.0
        self.ai_count = [date.today().isoformat(), 0]

    # --- camera ---
    def set_camera(self, source):
        with self.cam_lock:
            if self.camera:
                self.camera.stop()
            with self.lock:
                self.close_visits()
                self.tracker.clear()
                self.visible = []
            self.motion = Motion()
            self.camera = Camera(source)
            self.camera.start()
            log.info("camera source: %s", source)

    def close_visits(self):
        """Call with self.lock held."""
        for t in self.tracker.tracks:
            if t.event_id:
                self.log.update(t.event_id, ended=t.last_seen)
                t.event_id = None

    # --- live view loop: every frame gets the latest boxes drawn on it and goes to the dashboard ---
    def run(self):
        self.set_camera(self.cfg["camera"])
        threading.Thread(target=self._detect_loop, daemon=True).start()
        cam, seq, last_ph, shown = None, 0, 0.0, deque(maxlen=30)
        while True:
            if cam is not self.camera:
                cam, seq = self.camera, 0
            frame, seq = cam.latest(seq, timeout=0.5)
            now = time.time()
            if frame is None:
                if cam.status != "live" and now - last_ph > 1:
                    last_ph = now
                    with self.lock:
                        self.visible = []
                    title = "CONNECTING" if cam.status == "connecting" else "NO SIGNAL"
                    self._publish(jpeg(placeholder(title, cam.message or cam.source)))
                continue
            if shown and now - shown[-1] < 1 / STREAM_FPS:
                continue  # camera is faster than we need; skip this frame
            self.moving = self.motion.update(frame)
            with self.pcond:
                self.pending = (frame, now)
                self.pcond.notify()
            with self.lock:
                annotated = draw(frame, self.visible, self.cfg["zones"], now, self.cfg["camera_name"])
                self.raw, self.annotated = frame, annotated
            self._publish(jpeg(annotated))
            shown.append(time.time())
            if len(shown) > 1:
                self.fps = (len(shown) - 1) / max(shown[-1] - shown[0], 1e-6)

    # --- detection loop: YOLO a few times a second (less when nothing moves), then tracking and rules ---
    def _detect_loop(self):
        last = 0.0
        while True:
            interval = 1 / self.cfg["detect_fps"]
            with self.lock:
                busy = self.moving or any(t.category in VISIT_CATEGORIES for t in self.visible)
            gap = interval if busy else IDLE_DETECT
            if time.time() - last < gap:
                time.sleep(0.05)  # check again soon: movement may start any moment
                continue
            with self.pcond:
                while self.pending is None:
                    self.pcond.wait(1.0)
                frame, now = self.pending
                self.pending = None
            last = time.time()
            try:
                self._detect(frame, now, gap)
            except Exception:  # one bad frame must never stop the watching
                log.exception("detection failed")

    def _detect(self, frame, now, gap):
        cfg = self.cfg
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        dets = self.detector(frame, min(WEAK, cfg["confidence"]))
        ms = (time.perf_counter() - t0) * 1000
        self.det_ms = 0.8 * self.det_ms + 0.2 * ms if self.det_ms else ms
        with self.lock:
            newly, gone = self.tracker.update(dets, now, cfg["confidence"])
            visible = self.tracker.visible(now, 2.5 * gap)
            events, ended = self.rules.check(now, newly, gone, visible, (w, h))
            self.visible = visible
            snap = draw(frame, visible, cfg["zones"], now, cfg["camera_name"]) if events else None
        for t in ended:
            self.log.update(t.event_id, ended=t.last_seen)
        for e in events:
            self._record(e, frame, snap, now)

    def _publish(self, data):
        with self.jcond:
            self.jpeg, self.jseq = data, self.jseq + 1
            self.jcond.notify_all()

    def next_jpeg(self, after, timeout=2.0):
        with self.jcond:
            if self.jseq <= after:
                self.jcond.wait(timeout)
            return self.jpeg, self.jseq

    # --- events ---
    def save_snapshot(self, img, now, kind, tid):
        day = time.strftime("%Y-%m-%d", time.localtime(now))
        os.makedirs(os.path.join(SNAPS, day), exist_ok=True)
        rel = "{}/{}_{:03d}_{}_{}.jpg".format(day, time.strftime("%H%M%S", time.localtime(now)),
                                              int(now * 1000) % 1000, kind, tid or 0)
        with open(os.path.join(SNAPS, rel), "wb") as f:
            f.write(jpeg(img, 85))
        return rel

    def _record(self, e, frame, annotated, now):
        t = e["track"]
        cat, label = (t.category, t.label) if t else (e["category"], e["label"])
        snap = self.save_snapshot(annotated, now, e["kind"], t.id if t else 0)
        eid = self.log.add(now, e["kind"], e["level"], cat, label, t.id if t else None, e["message"], snap)
        if t and e["kind"] == "visit":
            t.event_id = eid
        log.info("event: %s", e["message"])
        if self.cfg["ai_auto"]:
            self._auto_describe(eid, frame, now, alert=e["level"] == "alert")

    # --- AI ---
    def context(self):
        with self.lock:
            tracks = list(self.visible)
        if not tracks:
            return "nothing"
        now = time.time()
        parts = []
        for t in sorted(tracks, key=lambda t: (t.category == "object", t.label)):
            extra = " in view {}".format(nice_duration(t.dwell(now))) if t.category != "object" else ""
            zone = " inside zone '{}'".format("', '".join(t.zones)) if t.zones else ""
            parts.append("{} #{}{}{}".format(t.label, t.id, extra, zone))
        return "; ".join(parts)

    def _auto_describe(self, eid, frame, now, alert):
        today = date.today().isoformat()
        if self.ai_count[0] != today:
            self.ai_count = [today, 0]
        gap = self.cfg["ai_gap_seconds"] / (3 if alert else 1)  # alerts may cut the queue a little
        if self.ai_busy or now - self.ai_last < gap or self.ai_count[1] >= self.cfg["ai_daily_limit"]:
            return
        if not ai.api_key():
            self.ai_error = "No Gemini key, so no AI descriptions (set GEMINI_API_KEY)."
            return
        self.ai_busy, self.ai_last = True, now
        self.ai_count[1] += 1
        threading.Thread(target=self._describe, args=(eid, frame, "auto"), daemon=True).start()

    def _describe(self, eid, frame, source):
        try:
            text = ai.describe(jpeg(fit(frame, 768), 80), self.cfg["camera_name"], self.context())
            self.ai_error = ""
            self.narration.appendleft({"ts": time.time(), "text": text, "source": source})
            if eid:
                self.log.update(eid, ai_text=text)
            return text
        except ai.AIError as e:
            self.ai_error = str(e)
            log.warning("AI: %s", e)
            raise
        finally:
            self.ai_busy = False

    def describe_now(self):
        with self.lock:
            frame = self.raw
        if frame is None:
            raise ai.AIError("No camera picture yet.")
        self.ai_busy = True
        return self._describe(None, frame, "manual")

    def ask(self, question):
        with self.lock:
            frame = self.raw
        if frame is None:
            raise ai.AIError("No camera picture yet.")
        return ai.ask(jpeg(fit(frame, 768), 80), self.cfg["camera_name"], question, self.context())

    def summary(self):
        day = date.today().isoformat()
        rows = list(reversed(self.log.day(day, 300)))
        if not rows:
            return "Nothing has been logged today yet."
        hm = lambda ts: time.strftime("%H:%M", time.localtime(ts))
        lines = ["(24-hour clock. The log runs from {} to {} with {} entries.)".format(
            hm(rows[0]["ts"]), hm(rows[-1]["ts"]), len(rows))]
        for r in rows:
            line = time.strftime("%H:%M:%S", time.localtime(r["ts"])) + " " + r["message"]
            if r["ended"]:
                line += " (stayed {})".format(nice_duration(r["ended"] - r["ts"]))
            if r["ai_text"]:
                line += " | seen: " + r["ai_text"]
            lines.append(line)
        return ai.summarise(self.cfg["camera_name"], day, lines)

    # --- for the dashboard ---
    def state(self):
        now = time.time()
        with self.lock:
            tracks = list(self.visible)
        cam = self.camera
        return {
            "camera": {"status": cam.status if cam else "connecting", "message": cam.message if cam else "",
                       "source": self.cfg["camera"], "name": self.cfg["camera_name"]},
            "fps": round(self.fps, 1), "detect_ms": round(self.det_ms), "motion": round(min(self.motion.level * 800,
                                                                                            100)),
            "objects": [{"id": t.id, "label": t.label, "category": t.category, "conf": round(t.conf, 2),
                         "dwell": round(t.dwell(now)), "alert": alerting(t), "zones": list(t.zones)}
                        for t in sorted(tracks, key=lambda t: (t.category == "object", t.id))],
            "ai": {"auto": self.cfg["ai_auto"], "busy": self.ai_busy, "error": self.ai_error,
                   "key": bool(ai.api_key()), "today": self.ai_count[1] if self.ai_count[0] == date.today().isoformat()
                   else 0, "limit": self.cfg["ai_daily_limit"], "narration": list(self.narration)},
            "rev": self.log.rev,
        }


def tidy(events, keep_days):
    """Deletes snapshots and log rows older than keep_days."""
    cutoff = (date.today() - timedelta(days=keep_days)).isoformat()
    events.forget_before(cutoff)
    if os.path.isdir(SNAPS):
        for d in os.listdir(SNAPS):
            if len(d) == 10 and d < cutoff:
                shutil.rmtree(os.path.join(SNAPS, d), ignore_errors=True)


# ---------------- web server ----------------
def make_app(cfg, engine, events):
    app = Flask(__name__, static_folder=None)
    allowed = {"127.0.0.1:{}".format(cfg["port"]), "localhost:{}".format(cfg["port"])}

    @app.before_request
    def local_only():
        # the camera feed is private: refuse other hostnames (DNS rebinding) and other websites' requests
        if request.host not in allowed:
            abort(403)
        origin = request.headers.get("Origin")
        if request.method != "GET" and origin and urlparse(origin).netloc not in allowed:
            abort(403)

    @app.after_request
    def caching(resp):
        # snapshots never change once saved; everything else is live
        resp.headers["Cache-Control"] = ("private, max-age=604800, immutable" if request.path.startswith("/snap/")
                                         else "no-store")
        return resp

    @app.get("/")
    def index():
        return send_from_directory(os.path.join(ROOT, "web"), "index.html")

    @app.get("/stream")
    def stream():
        def gen():
            seq = 0
            while True:
                data, seq = engine.next_jpeg(seq)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() +
                       b"\r\n\r\n" + data + b"\r\n")
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/frame.jpg")
    def frame():
        return Response(engine.next_jpeg(0)[0], mimetype="image/jpeg")

    @app.get("/snap/<path:rel>")
    def snap(rel):
        return send_from_directory(SNAPS, rel)

    @app.get("/api/state")
    def state():
        out = engine.state()
        since = request.args.get("rev", type=int)
        if since is not None:
            out["changes"] = events.changes(since)
            if out["changes"]:
                out["rev"] = out["changes"][-1]["rev"]
        return jsonify(out)

    @app.get("/api/events")
    def day_events():
        day = request.args.get("day") or date.today().isoformat()
        return jsonify({"day": day, "events": events.day(day), "rev": events.rev, "stats": events.stats(day)})

    @app.get("/api/stats")
    def stats():
        return jsonify(events.stats(request.args.get("day") or date.today().isoformat()))

    def ai_reply(fn, *args):
        try:
            return jsonify({"ok": True, "text": fn(*args)})
        except ai.AIError as e:
            return jsonify({"ok": False, "text": str(e)})

    @app.post("/api/describe")
    def describe():
        return ai_reply(engine.describe_now)

    @app.post("/api/ask")
    def ask():
        q = str((request.get_json(silent=True) or {}).get("question", "")).strip()[:300]
        if not q:
            return jsonify({"ok": False, "text": "Type a question first."})
        return ai_reply(engine.ask, q)

    @app.post("/api/summary")
    def summary():
        return ai_reply(engine.summary)

    @app.post("/api/snapshot")
    def snapshot():
        with engine.lock:
            img = engine.annotated
        if img is None:
            return jsonify({"ok": False})
        now = time.time()
        rel = engine.save_snapshot(img, now, "manual", 0)
        events.add(now, "manual", "info", "object", "snapshot", None, "Snapshot taken", rel)
        return jsonify({"ok": True, "snapshot": rel})

    def public_config():
        return jsonify({**{k: v for k, v in cfg.items() if not k.startswith("_")},
                        "demo_available": os.path.isfile(DEMO)})

    @app.get("/api/config")
    def get_config():
        return public_config()

    @app.post("/api/config")
    def set_config():
        body = request.get_json(silent=True) or {}
        old_camera = cfg["camera"]
        for k, v in body.items():
            if k == "zones":
                cfg["zones"] = clean_zones(v)
            elif k == "watch" and isinstance(v, list):
                cfg["watch"] = [c for c in v if c in VISIT_CATEGORIES]
            elif k in EDITABLE:
                kind, lo, hi = EDITABLE[k]
                try:
                    if kind is bool:
                        v = bool(v)
                    elif kind is str:
                        v = str(v).strip()
                        if not lo <= len(v) <= hi:
                            continue
                    else:
                        v = min(max(kind(v), lo), hi)
                except (TypeError, ValueError):
                    continue
                cfg[k] = v
        if cfg["camera"] != old_camera:
            cfg.pop("_saved_camera", None)  # picked in Settings, so it sticks
        save_config(cfg)
        if cfg["camera"] != old_camera:
            engine.set_camera(cfg["camera"])
        return public_config()

    @app.post("/api/quit")
    def quit_app():
        def bye():
            time.sleep(0.3)
            with engine.lock:
                engine.close_visits()
            if engine.camera:
                engine.camera.stop()
            os._exit(0)
        threading.Thread(target=bye, daemon=True).start()
        return jsonify({"ok": True})

    return app


def open_dashboard(url):
    """Opens the dashboard in its own app-style window (Chrome or Edge), else the default browser."""
    local = os.environ.get("LOCALAPPDATA", "")
    for exe in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                os.path.join(local, r"Google\Chrome\Application\chrome.exe"),
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"):
        if os.path.isfile(exe):
            subprocess.Popen([exe, "--app=" + url, "--window-size=1500,920"])
            return
    import webbrowser
    webbrowser.open(url)


def already_running(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main():
    os.makedirs(SNAPS, exist_ok=True)
    if sys.stdout is None:  # started with pythonw (no console window)
        sys.stdout = sys.stderr = open(os.path.join(DATA, "console.log"), "a", encoding="utf-8")
    handler = logging.handlers.RotatingFileHandler(os.path.join(DATA, "smartcam.log"), maxBytes=1_000_000,
                                                   backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler, logging.StreamHandler()])
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    cfg = load_config()
    if not os.path.isfile(CONFIG):
        save_config(cfg)
    if "--camera" in sys.argv:  # just for this run, e.g. --camera demo
        cfg["_saved_camera"] = cfg["camera"]
        cfg["camera"] = sys.argv[sys.argv.index("--camera") + 1]
    no_browser = "--no-browser" in sys.argv or not cfg["open_browser"]
    url = "http://127.0.0.1:{}".format(cfg["port"])
    if already_running(cfg["port"]):
        if not no_browser:
            open_dashboard(url)
        return

    events = EventLog(os.path.join(DATA, "events.db"))
    events.close_open()
    tidy(events, cfg["keep_days"])

    def tidy_forever():
        while True:
            time.sleep(6 * 3600)
            tidy(events, cfg["keep_days"])

    engine = Engine(cfg, events)
    engine.start()
    threading.Thread(target=tidy_forever, daemon=True).start()
    if not no_browser:
        threading.Timer(1.0, open_dashboard, args=(url,)).start()
    log.info("SmartCam running at %s", url)
    make_app(cfg, engine, events).run(host="127.0.0.1", port=cfg["port"], threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
