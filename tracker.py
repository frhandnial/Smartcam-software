"""Keeps the same ID on each object from one detection to the next, so 'person #7' stays #7 while in view.

Detection runs only a few times a second, so three tricks keep IDs steady:
  - each object's speed is remembered and boxes are matched where the object should be *now*;
  - weak detections (half hidden behind someone) may continue an object but never start a new one;
  - an object missing for a moment is searched for in a circle that grows the longer it has been gone.
"""
import itertools
import math
from collections import deque

from detector import COCO, category

# false alarms (a rock seen as a dog, a shadow as a bag) score low, so these need a surer first sighting
START_AT = {"animal": 0.5, "bag": 0.5}


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _affinity(a, b, lost):
    """How likely two boxes are the same object: overlap, else nearby centres of similar size."""
    iou = _iou(a, b)
    if iou > 0.05:
        return iou
    aw, ah, bw, bh = a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]
    if not (0.5 < (bw * bh) / max(aw * ah, 1.0) < 2.0):
        return 0.0
    dist = math.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)
    reach = max(aw, ah, bw, bh) * (1 + min(lost, 2.0))
    return 0.3 * (1 - dist / reach) if dist < reach else 0.0


class Track:
    def __init__(self, tid, det, now):
        self.id = tid
        self.box = list(det[:4])
        self.conf = det[4]
        self.votes = {det[5]: 1}
        self.first_seen = self.last_seen = now
        self.vel = (0.0, 0.0)  # centre speed, pixels per second
        self.hits = 1
        self.confirmed = False
        self.trail = deque([self.centre], maxlen=40)
        self.zones = {}         # zone name -> time it walked in
        self.zone_alerted = {}  # zone name -> last alert time
        self.flags = set()      # alerts already raised for this object
        self.event_id = None    # its row in the event log

    def predicted(self, now):
        dt = min(max(now - self.last_seen, 0.0), 1.5)
        dx, dy = self.vel[0] * dt, self.vel[1] * dt
        b = self.box
        return [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]

    def hit(self, det, now):
        dt = now - self.last_seen
        before = self.centre
        guess = self.predicted(now)
        k = 0.7  # lean towards the new box, but smooth out jitter
        self.box = [k * n + (1 - k) * g for n, g in zip(det[:4], guess)]
        if dt > 0:
            vx, vy = (self.centre[0] - before[0]) / dt, (self.centre[1] - before[1]) / dt
            self.vel = (0.6 * vx + 0.4 * self.vel[0], 0.6 * vy + 0.4 * self.vel[1])
        self.conf = det[4]
        self.votes[det[5]] = self.votes.get(det[5], 0) + 1
        self.last_seen = now
        self.hits += 1
        self.trail.append(self.centre)

    @property
    def label(self):
        return COCO[max(self.votes, key=self.votes.get)]  # car/truck flicker settles on the most common answer

    @property
    def category(self):
        return category(self.label)

    @property
    def centre(self):
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)

    @property
    def foot(self):
        return ((self.box[0] + self.box[2]) / 2, self.box[3])

    def dwell(self, now):
        return now - self.first_seen

    def moved(self):
        """How far it has wandered recently, in box heights."""
        if len(self.trail) < 2:
            return 0.0
        xs, ys = [p[0] for p in self.trail], [p[1] for p in self.trail]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys)) / max(self.box[3] - self.box[1], 1.0)


class Tracker:
    def __init__(self, max_age=4.0, min_hits=2):
        self.max_age = max_age    # seconds an object can go unseen (behind a pillar, missed frame) and keep its ID
        self.min_hits = min_hits  # sightings before it counts, so one-frame mistakes never become events
        self.tracks = []
        self._ids = itertools.count(1)

    def _match(self, tracks, dets, now, floor):
        pairs = []
        for ti, t in enumerate(tracks):
            guess, lost = t.predicted(now), now - t.last_seen
            for di, d in enumerate(dets):
                label = COCO[d[5]]
                if category(label) == t.category and (t.category != "object" or label == t.label):
                    a = _affinity(guess, d[:4], lost)
                    if a > floor:
                        pairs.append((a, ti, di))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti not in used_t and di not in used_d:
                used_t.add(ti)
                used_d.add(di)
                tracks[ti].hit(dets[di], now)
        return [t for i, t in enumerate(tracks) if i not in used_t], [d for i, d in enumerate(dets) if i not in used_d]

    def update(self, dets, now, conf=0.4):
        """dets may include weak ones below conf. Returns (newly confirmed tracks, confirmed tracks that have gone)."""
        strong = [d for d in dets if d[4] >= conf]
        weak = [d for d in dets if d[4] < conf]
        left, fresh = self._match(self.tracks, strong, now, 0.1)
        self._match([t for t in left if now - t.last_seen < 1.0], weak, now, 0.2)
        for d in fresh:
            if d[4] >= START_AT.get(category(COCO[d[5]]), 0):
                self.tracks.append(Track(next(self._ids), d, now))

        gone = [t for t in self.tracks if now - t.last_seen > self.max_age and t.confirmed]
        # a maybe-object seen once and then never again is dropped quickly so it can't steal a real one's box
        self.tracks = [t for t in self.tracks
                       if now - t.last_seen <= (self.max_age if t.confirmed else 1.0)]
        newly = [t for t in self.tracks if not t.confirmed and t.hits >= self.min_hits]
        for t in newly:
            t.confirmed = True
        return newly, gone

    def visible(self, now, fresh):
        """Confirmed objects seen within the last `fresh` seconds."""
        return [t for t in self.tracks if t.confirmed and now - t.last_seen <= fresh]

    def clear(self):
        self.tracks = []
