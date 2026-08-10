"""
SCRFD face detector wrapper (ONNX) — used to locate the face in an arbitrary
portrait so Wav2Lip gets a real face crop (not a center-crop guess).

Model: scrfd_2.5g_bnkps.onnx (ships with instant-high/wav2lip-onnx-HQ).
Input : input.1 [1,3,H,W] float, RGB, /255
Outputs (3 strides 8/16/32): scores, bboxes [x1,y1,x2,y2], landmarks [5*2].
"""
from __future__ import annotations

import os
import numpy as np
import cv2
import onnxruntime as ort


class SCRFDetector:
    STRIDES = (8, 16, 32)
    SCORE_OUT = ["446", "466", "486"]
    BBOX_OUT = ["449", "469", "489"]
    KPS_OUT = ["452", "472", "492"]

    def __init__(self, onnx_path: str, providers=("CPUExecutionProvider",)):
        self.sess = ort.InferenceSession(onnx_path, providers=list(providers))
        self.in_name = self.sess.get_inputs()[0].name

    def _anchors(self, feat_h, feat_w, stride):
        shift_x = np.arange(feat_w) * stride + stride // 2
        shift_y = np.arange(feat_h) * stride + stride // 2
        xs, ys = np.meshgrid(shift_x, shift_y)
        return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)

    def detect(self, img_bgr, score_thresh=0.5, nms_thresh=0.4):
        h, w = img_bgr.shape[:2]
        # resize to multiple of 32 keeping aspect
        scale = min(640 / w, 640 / h)
        nw, nh = int(w * scale), int(h * scale)
        blob = cv2.resize(img_bgr, (nw, nh))
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]
        outs = self.sess.run(None, {self.in_name: blob})
        out_map = {o.name: outs[i] for i, o in enumerate(self.sess.get_outputs())}

        boxes, scores, kps_all = [], [], []
        for stride, so, bo, ko in zip(self.STRIDES, self.SCORE_OUT, self.BBOX_OUT, self.KPS_OUT):
            s = out_map[so].reshape(-1)
            b = out_map[bo].reshape(-1, 4)
            k = out_map[ko].reshape(-1, 5, 2)
            n = b.shape[0]
            # anchors: a (nw/stride) x (nh/stride) grid centered on stride//2
            fh_ = int(np.ceil(nh / stride)); fw_ = int(np.ceil(nw / stride))
            # the model's actual grid may differ slightly; build exactly n anchors
            anc = self._anchors(fh_, fw_, stride)
            if anc.shape[0] > n:
                anc = anc[:n]
            elif anc.shape[0] < n:
                # pad by repeating last row if model produced more (shouldn't)
                extra = np.repeat(anc[-1:], n - anc.shape[0], axis=0)
                anc = np.concatenate([anc, extra], axis=0)
            x1 = anc[:, 0] - b[:, 0]; y1 = anc[:, 1] - b[:, 1]
            x2 = anc[:, 0] + b[:, 2]; y2 = anc[:, 1] + b[:, 3]
            kps = anc[:, None, :] + k  # k already in input-pixel deltas from anchor
            mask = s > score_thresh
            if not mask.any():
                continue
            boxes.append(np.stack([x1, y1, x2, y2], axis=1)[mask] / scale)
            scores.append(s[mask])
            kps_all.append(kps[mask] / scale)

        if not boxes:
            return None
        boxes = np.concatenate(boxes); scores = np.concatenate(scores)
        kps_all = np.concatenate(kps_all)
        # NMS (cv2 5.x NMSBoxes wants python lists of ints/floats)
        idx = cv2.dnn.NMSBoxes(
            boxes.astype(int).tolist(), scores.astype(float).tolist(),
            score_thresh, nms_thresh)
        if idx is None or len(idx) == 0:
            return None
        i = int(idx[0])
        x1, y1, x2, y2 = boxes[i].astype(int)
        return {"box": (int(x1), int(y1), int(x2), int(y2)),
                "score": float(scores[i]),
                "landmarks": kps_all[i]}
