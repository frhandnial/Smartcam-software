"""Finds objects in a frame with YOLO11n (80 everyday object types) on ONNX Runtime, CPU only."""
import os

import cv2
import numpy as np
import onnxruntime as ort

COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"]

# what the rules care about; everything else is just "object"
CATEGORY = {"person": "person",
            **dict.fromkeys(["bicycle", "car", "motorcycle", "bus", "train", "truck", "boat", "airplane"], "vehicle"),
            **dict.fromkeys(["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
                            "animal"),
            **dict.fromkeys(["backpack", "handbag", "suitcase"], "bag")}


def category(label):
    return CATEGORY.get(label, "object")


class Detector:
    def __init__(self, model_path, size=640, threads=None):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or min(4, os.cpu_count() or 4)  # leave the rest of the laptop free
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")  # sleep between frames, don't spin
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.size = size

    def __call__(self, frame, conf=0.4, iou=0.5):
        """Returns [(x1, y1, x2, y2, score, class_id), ...] in frame pixels."""
        h, w = frame.shape[:2]
        s = self.size
        r = min(s / w, s / h)
        nw, nh = round(w * r), round(h * r)
        px, py = (s - nw) // 2, (s - nh) // 2
        canvas = np.full((s, s, 3), 114, np.uint8)  # letterbox: keep the shape, pad with grey
        canvas[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, swapRB=True)

        out = self.session.run(None, {self.input: blob})[0][0].T  # 8400 candidates x (cx, cy, w, h, 80 scores)
        scores = out[:, 4:]
        cls = scores.argmax(1)
        best = scores[np.arange(len(cls)), cls]
        keep = best >= conf
        if not keep.any():
            return []
        out, cls, best = out[keep], cls[keep], best[keep]

        boxes = out[:, :4].copy()
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        shifted = boxes.copy()
        shifted[:, :2] += cls[:, None] * 4096  # one NMS pass that never merges different classes
        idx = cv2.dnn.NMSBoxes(shifted.tolist(), best.tolist(), conf, iou)

        dets = []
        for i in np.array(idx, dtype=int).flatten():
            x, y, bw, bh = boxes[i]
            x1 = min(max((x - px) / r, 0), w - 1)
            y1 = min(max((y - py) / r, 0), h - 1)
            x2 = min(max((x + bw - px) / r, 0), w - 1)
            y2 = min(max((y + bh - py) / r, 0), h - 1)
            if x2 - x1 > 2 and y2 - y1 > 2:
                dets.append((float(x1), float(y1), float(x2), float(y2), float(best[i]), int(cls[i])))
        return dets
