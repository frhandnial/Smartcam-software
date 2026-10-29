"""The 'what is happening' rules (visits, loitering, zones, crowds, unattended bags) and the event log they write to."""
import sqlite3
import threading
import time

import cv2
import numpy as np

NOUN = {"person": "Person", "tv": "TV", "cell phone": "Phone"}
VISIT_CATEGORIES = ("person", "vehicle", "animal")


def noun(label):
    return NOUN.get(label, label.capitalize())


def nice_duration(sec):
    sec = int(round(sec))
    if sec < 60:
        return "{} s".format(sec)
    if sec < 3600:
        return "{} min {} s".format(sec // 60, sec % 60) if sec % 60 else "{} min".format(sec // 60)
    return "{} h {} min".format(sec // 3600, sec % 3600 // 60)


class EventLog:
    """SQLite file in data/: one row per visit or alert, with its snapshot and the AI's description."""

    COLS = ("id", "ts", "day", "kind", "level", "category", "label", "track_id", "message", "snapshot",
            "ended", "ai_text", "rev")

    def __init__(self, path):
        self._lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY, ts REAL, day TEXT, kind TEXT, level TEXT, category TEXT, label TEXT,
            track_id INTEGER, message TEXT, snapshot TEXT, ended REAL, ai_text TEXT, rev INTEGER)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS events_day ON events(day)")
        self.db.execute("CREATE INDEX IF NOT EXISTS events_rev ON events(rev)")
        self.db.commit()
        # every insert or change gets the next revision number, so the dashboard can ask "what changed since N?"
        self._rev = self.db.execute("SELECT COALESCE(MAX(rev), 0) FROM events").fetchone()[0]

    def add(self, ts, kind, level, category, label, track_id, message, snapshot=None):
        with self._lock:
            self._rev += 1
            cur = self.db.execute(
                "INSERT INTO events (ts, day, kind, level, category, label, track_id, message, snapshot, rev) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ts, time.strftime("%Y-%m-%d", time.localtime(ts)), kind, level, category, label, track_id,
                 message, snapshot, self._rev))
            self.db.commit()
            return cur.lastrowid

    def update(self, event_id, **fields):
        fields = {k: v for k, v in fields.items() if k in self.COLS and k not in ("id", "rev")}
        if not fields:
            return
        with self._lock:
            self._rev += 1
            fields["rev"] = self._rev
            self.db.execute("UPDATE events SET {} WHERE id=?".format(", ".join(k + "=?" for k in fields)),
                            (*fields.values(), event_id))
            self.db.commit()

    def _rows(self, sql, args):
        with self._lock:
            return [dict(zip(self.COLS, r)) for r in self.db.execute(sql, args).fetchall()]

    def get(self, event_id):
        rows = self._rows("SELECT * FROM events WHERE id=?", (event_id,))
        return rows[0] if rows else None

    def close_open(self):
        """Visits still open from the last run (the app was closed while someone was in view)."""
        with self._lock:
            self._rev += 1
            self.db.execute("UPDATE events SET ended=ts, rev=? WHERE kind='visit' AND ended IS NULL", (self._rev,))
            self.db.commit()

    def changes(self, since_rev, limit=200):
        return self._rows("SELECT * FROM events WHERE rev>? ORDER BY rev LIMIT ?", (since_rev, limit))

    def day(self, day, limit=500):
        return self._rows("SELECT * FROM events WHERE day=? ORDER BY id DESC LIMIT ?", (day, limit))

    @property
    def rev(self):
        return self._rev

    def stats(self, day):
        with self._lock:
            rows = self.db.execute(
                "SELECT category, level, strftime('%H', ts, 'unixepoch', 'localtime'), COUNT(*) FROM events "
                "WHERE day=? GROUP BY category, level, 3", (day,)).fetchall()
        hourly = {c: [0] * 24 for c in VISIT_CATEGORIES}
        totals = {"person": 0, "vehicle": 0, "animal": 0, "alerts": 0}
        for cat, level, hour, n in rows:
            if level == "alert":
                totals["alerts"] += n
            elif cat in hourly:
                hourly[cat][int(hour)] += n
                totals[cat] += n
        return {"totals": totals, "hourly": hourly}

    def snapshots_before(self, day):
        with self._lock:
            return [r[0] for r in self.db.execute(
                "SELECT snapshot FROM events WHERE day<? AND snapshot IS NOT NULL", (day,)).fetchall()]

    def forget_before(self, day):
        with self._lock:
            self.db.execute("DELETE FROM events WHERE day<?", (day,))
            self.db.commit()


class Rules:
    """Looks at the tracked objects after every detection and decides what is worth logging."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.crowd_until = 0.0
        self.crowd_was = 0

    def check(self, now, newly, gone, visible, size):
        """Returns (new events, finished visits). Events are dicts the engine snapshots and logs."""
        cfg, w, h = self.cfg, size[0], size[1]
        events = []

        for t in newly:
            if t.category in VISIT_CATEGORIES and t.category in cfg["watch"]:
                events.append(dict(kind="visit", level="info", track=t,
                                   message="{} appeared (#{})".format(noun(t.label), t.id)))

        people = [t for t in visible if t.category == "person"]

        loiter = cfg["loiter_seconds"]
        for t in people:
            if loiter and "loiter" not in t.flags and t.dwell(now) >= loiter:
                t.flags.add("loiter")
                events.append(dict(kind="loiter", level="alert", track=t,
                                   message="Person #{} has been here {} (loitering)".format(
                                       t.id, nice_duration(t.dwell(now)))))

        zones = [z for z in cfg["zones"] if len(z.get("points", [])) >= 3]
        for t in visible:
            if t.category not in VISIT_CATEGORIES or t.category not in cfg["watch"]:
                continue
            fx, fy = t.foot[0] / w, t.foot[1] / h
            for z in zones:
                poly = np.array(z["points"], np.float32)
                inside = cv2.pointPolygonTest(poly, (fx, fy), False) >= 0
                was = z["name"] in t.zones
                if inside and not was:
                    t.zones[z["name"]] = now
                    if now - t.zone_alerted.get(z["name"], -1e9) > 10:  # walking along the edge fires once
                        t.zone_alerted[z["name"]] = now
                        events.append(dict(kind="zone", level="alert", track=t,
                                           message="{} #{} entered zone '{}'".format(noun(t.label), t.id,
                                                                                    z["name"])))
                elif not inside and was:
                    del t.zones[z["name"]]

        crowd = cfg["crowd_size"]
        if crowd and len(people) >= crowd > self.crowd_was and now > self.crowd_until:
            self.crowd_until = now + 120
            events.append(dict(kind="crowd", level="alert", track=None, category="person", label="person",
                               message="Crowd: {} people in view".format(len(people))))
        self.crowd_was = len(people)

        alone = cfg["unattended_seconds"]
        for t in visible:
            if t.category != "bag" or not alone or "alone" in t.flags or t.dwell(now) < alone or t.moved() > 0.5:
                continue
            near = any(abs(p.centre[0] - t.centre[0]) + abs(p.centre[1] - t.centre[1]) < (p.box[3] - p.box[1])
                       for p in people)
            if not near:
                t.flags.add("alone")
                events.append(dict(kind="unattended", level="alert", track=t,
                                   message="Unattended {} (#{}), still for {}".format(
                                       t.label, t.id, nice_duration(t.dwell(now)))))

        ended = [t for t in gone if t.event_id]
        return events, ended
