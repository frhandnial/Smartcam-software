"""Draws smartcam.ico (navy rounded square with the green eye from the dashboard).

Run once: venv\\Scripts\\python make_icon.py
"""
import os
import struct

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BIG = 1024                # drawn large, then shrunk, for smooth edges
BG = (40, 22, 13)         # BGR of the dashboard's #0d1628
EDGE = (92, 58, 42)       # #2a3a5c
GREEN = (132, 220, 61)    # #3ddc84


def draw(small=False):
    """small: a simpler eye (outline + solid pupil) that stays readable at 16-24 px."""
    img = np.zeros((BIG, BIG, 4), np.uint8)
    pad, r = 32, 200
    shape = np.zeros((BIG, BIG), np.uint8)
    cv2.rectangle(shape, (pad + r, pad), (BIG - pad - r, BIG - pad), 255, -1)
    cv2.rectangle(shape, (pad, pad + r), (BIG - pad, BIG - pad - r), 255, -1)
    for cx in (pad + r, BIG - pad - r):
        for cy in (pad + r, BIG - pad - r):
            cv2.circle(shape, (cx, cy), r, 255, -1, cv2.LINE_AA)
    inner = cv2.erode(shape, np.ones((3, 3), np.uint8), iterations=12)
    img[shape > 0] = EDGE + (255,)
    img[inner > 0] = BG + (255,)
    img[..., 3] = shape

    # almond eye outline, iris ring and pupil
    c, a, b = BIG // 2, 330, 215
    t = np.linspace(-1, 1, 200)
    top = np.stack([c + a * t, c - b * (1 - t ** 2)], 1)
    bottom = np.stack([c + a * t[::-1], c + b * (1 - t[::-1] ** 2)], 1)
    outline = np.concatenate([top, bottom]).astype(np.int32)
    if small:
        cv2.polylines(img, [outline], True, GREEN + (255,), 90, cv2.LINE_AA)
        cv2.circle(img, (c, c), 120, GREEN + (255,), -1, cv2.LINE_AA)
    else:
        cv2.polylines(img, [outline], True, GREEN + (255,), 58, cv2.LINE_AA)
        cv2.circle(img, (c, c), 118, GREEN + (255,), 52, cv2.LINE_AA)
        cv2.circle(img, (c, c), 52, GREEN + (255,), -1, cv2.LINE_AA)
    img[..., 3] = shape  # keep the rounded-square transparency after the green strokes
    return img


def main():
    full, simple = draw(), draw(small=True)
    sizes = [256, 48, 32, 24, 16]
    images = [cv2.imencode(".png", cv2.resize(simple if s <= 24 else full, (s, s),
                                              interpolation=cv2.INTER_AREA))[1].tobytes() for s in sizes]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries = b""
    for s, data in zip(sizes, images):
        entries += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    with open(os.path.join(HERE, "smartcam.ico"), "wb") as f:
        f.write(header + entries + b"".join(images))


if __name__ == "__main__":
    main()
