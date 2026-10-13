#!/usr/bin/env python3
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BC7_WEIGHTS = (0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64)


def ramp(v, n):
    return (v * 255) // max(n - 1, 1)


def pattern(content, w, h):
    y, x = np.mgrid[0:h, 0:w].astype(np.int64)
    img = np.zeros((h, w, 4), np.int64)
    if content in ("base", "albedo"):
        checker = (x // 8 + y // 8) & 1
        img[..., 0] = ramp(x, w)
        img[..., 1] = ramp(y, h)
        img[..., 2] = 64 + 128 * checker if content == "albedo" else 192 - 128 * checker
        img[..., 3] = 255
    elif content == "orm":
        img[..., 0] = 255 - ((x - w // 2) ** 2 + (y - h // 2) ** 2) * 128 // ((w // 2) ** 2 + (h // 2) ** 2)
        img[..., 1] = ramp(x, w)
        img[..., 2] = np.where(x >= w // 2, 255, 0)
        img[..., 3] = 96 + (y * 64) // max(h - 1, 1)
    elif content == "emit":
        inside = (np.abs(2 * x - (w - 1)) < w // 2) & (np.abs(2 * y - (h - 1)) < h // 2)
        img[..., 0:3] = np.where(inside[..., None], np.array([255, 160, 32]), 0)
        img[..., 3] = 255
    elif content == "nrm":
        # +Y up: rows above the centre tilt toward +G.
        img[..., 0] = np.clip(128 + (x - w // 2) * 96 // max(w // 2, 1), 0, 255)
        img[..., 1] = np.clip(128 - (y - h // 2) * 96 // max(h // 2, 1), 0, 255)
        img[..., 3] = 255
    elif content == "delight":
        img[..., 0] = 112 + (x + y) * 32 // max(w + h - 2, 1)
    elif content == "grassmask":
        img[..., 0] = np.where(x // 4 % 2 == 0, 255, 0)
    elif content == "mat":
        nrm = pattern("nrm", w, h)
        img[..., 0], img[..., 1], img[..., 2], img[..., 3] = 32, nrm[..., 1], 127, nrm[..., 0]
    elif content != "zero":
        raise ValueError(f"unknown content {content}")
    return img


def half(img):
    return (img[0::2, 0::2] + img[1::2, 0::2] + img[0::2, 1::2] + img[1::2, 1::2] + 2) // 4


def level_image(entry):
    w, h = entry["width"], entry["height"]
    img = pattern(entry["content"], entry.get("chainWidth", w), entry.get("chainHeight", h))
    for _ in range(entry.get("chainLevel", 0)):
        img = half(img)
    lh, lw = img.shape[:2]
    if (lw, lh) != (w, h):
        # Pack chains clamp at native size: lower levels are upscaled copies, not duplicates.
        img = np.repeat(np.repeat(img, h // lh, axis=0), w // lw, axis=1)
    return img


def blocks(img):
    h, w = img.shape[:2]
    img = np.pad(img, ((0, -h % 4), (0, -w % 4), (0, 0)), mode="edge")
    for by in range(0, img.shape[0], 4):
        for bx in range(0, img.shape[1], 4):
            yield [tuple(int(c) for c in img[by + i // 4, bx + i % 4]) for i in range(16)]


def bc4_block(values):
    lo, hi = min(values), max(values)
    if lo == hi:
        return bytes((hi, lo)) + bytes(6)
    bits = 0
    for i, v in enumerate(values):
        k = (14 * (v - lo) + (hi - lo)) // (2 * (hi - lo))
        bits |= (0 if k == 7 else 1 if k == 0 else 8 - k) << (3 * i)
    return bytes((hi, lo)) + bits.to_bytes(6, "little")


def bc7_block(pixels):
    e0 = [min(p[c] for p in pixels) & 0xFE for c in range(4)]
    e1 = [max(p[c] for p in pixels) | 1 for c in range(4)]
    d = [b - a for a, b in zip(e0, e1)]
    den = sum(v * v for v in d)
    idx = []
    for p in pixels:
        num = sum((p[c] - e0[c]) * d[c] for c in range(4))
        idx.append(min(range(16), key=lambda i: (abs(64 * num - BC7_WEIGHTS[i] * den), i)))
    if idx[0] >= 8:
        e0, e1 = e1, e0
        idx = [15 - i for i in idx]
    fields = [(1 << 6, 7)]
    for c in range(4):
        fields += [(e0[c] >> 1, 7), (e1[c] >> 1, 7)]
    fields += [(e0[0] & 1, 1), (e1[0] & 1, 1), (idx[0], 3)] + [(i, 4) for i in idx[1:]]
    bits = pos = 0
    for value, width in fields:
        bits |= value << pos
        pos += width
    return bits.to_bytes(16, "little")


def encode(img, code):
    out = bytearray()
    for px in blocks(img):
        if code == 80:
            out += bc4_block([p[0] for p in px])
        elif code == 83:
            out += bc4_block([p[0] for p in px]) + bc4_block([p[1] for p in px])
        elif code in (98, 99):
            out += bc7_block(px)
        elif code == 95:
            out += bytes(16)
        else:
            raise ValueError(f"unsupported format {code}")
    return bytes(out)


def header(entry, level0_size):
    style, w, h, mips = entry["header"], entry["width"], entry["height"], entry["mips"]
    if style == "gz2":
        assert mips == 1
        flags, pf_flags, caps, mip_count = 0x00081007, 0x44, 0, 0
    else:
        flags = 0x00081007 | (0x00020000 if mips > 1 else 0)
        pf_flags, mip_count = 0x4, mips
        caps = 0x1000 | (0x00400008 if mips > 1 else 0)
    fourcc = 0x32495441 if style == "ati2" else 0x30315844
    out = struct.pack("<4s7I44x8I5I", b"DDS ", 124, flags, h, w, level0_size, 0, mip_count,
                      32, pf_flags, fourcc, 0, 0, 0, 0, 0, caps, 0, 0, 0, 0)
    if style != "ati2":
        out += struct.pack("<5I", entry["dxgi"], 3, 0, 1, 0)
    return out


def build(entry):
    code = entry.get("dxgi", 83)
    img = level_image(entry)
    levels = []
    for i in range(entry["mips"]):
        if i:
            img = half(img)
        levels.append(encode(img, code))
    head = header(entry, len(levels[0]))
    payload = b"".join(levels)
    defect = entry.get("defect")
    if defect == "truncate":
        payload = payload[: len(payload) // 2]
    elif defect == "array2":
        head = head[:0x8C] + struct.pack("<I", 2) + head[0x90:]
    elif defect is not None:
        raise ValueError(f"unknown defect {defect}")
    return head + payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    index = json.loads((ROOT / "index.json").read_text(encoding="utf-8"))
    out_dir = ROOT / "dds"
    files = {entry["file"]: build(entry) for entry in index["dds"]}
    if args.check:
        stale = [n for n, data in files.items() if not (out_dir / n).is_file() or (out_dir / n).read_bytes() != data]
        extra = sorted(p.name for p in out_dir.iterdir() if p.name not in files)
        for name in stale + extra:
            print(name)
        return 1 if stale or extra else 0
    out_dir.mkdir(exist_ok=True)
    for name, data in files.items():
        (out_dir / name).write_bytes(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
