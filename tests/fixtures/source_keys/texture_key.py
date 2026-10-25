#!/usr/bin/env python3
import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import NamedTuple, Optional

ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "source_keys.json"

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
WILDCARD_TEXTURE_HASH = 0xFFFFFFFFFFFFFFFF
WILDCARD_TLUT_HASH = 0xFFFFFFFFFFFFFFFE

I4, I8, IA4, IA8, RGB565, RGB5A3, RGBA8 = 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6
C4, C8, C14X2, CMPR = 0x8, 0x9, 0xA, 0xE
Z8 = 0x11
R8_PC, RG8_PC, RGBA8_PC, BC1_PC = 0x41, 0x43, 0x46, 0x4E
PALETTE_FORMATS = (C4, C8, C14X2)
FORMAT_NAMES = {
    I4: "I4", I8: "I8", IA4: "IA4", IA8: "IA8", RGB565: "RGB565", RGB5A3: "RGB5A3", RGBA8: "RGBA8",
    C4: "C4", C8: "C8", C14X2: "C14X2", CMPR: "CMPR", Z8: "Z8",
    R8_PC: "R8_PC", RG8_PC: "RG8_PC", RGBA8_PC: "RGBA8_PC", BC1_PC: "BC1_PC",
}

P1 = 0x9E3779B185EBCA87
P2 = 0xC2B2AE3D27D4EB4F
P3 = 0x165667B19E3779F9
P4 = 0x85EBCA77C2B2AE63
P5 = 0x27D4EB2F165667C5


class SourceKey(NamedTuple):
    textureHash: int
    tlutHash: int
    width: int
    height: int
    format: int
    hasTlut: bool


def _rotl(value, bits):
    return ((value << bits) | (value >> (64 - bits))) & MASK64


def _round(acc, lane):
    return (_rotl((acc + lane * P2) & MASK64, 31) * P1) & MASK64


def _merge(acc, value):
    return (((acc ^ _round(0, value)) * P1) + P4) & MASK64


def xxh64(data, seed=0):
    size = len(data)
    pos = 0
    if size >= 32:
        v1 = (seed + P1 + P2) & MASK64
        v2 = (seed + P2) & MASK64
        v3 = seed & MASK64
        v4 = (seed - P1) & MASK64
        while pos + 32 <= size:
            a, b, c, d = struct.unpack_from("<4Q", data, pos)
            v1, v2, v3, v4 = _round(v1, a), _round(v2, b), _round(v3, c), _round(v4, d)
            pos += 32
        acc = (_rotl(v1, 1) + _rotl(v2, 7) + _rotl(v3, 12) + _rotl(v4, 18)) & MASK64
        for value in (v1, v2, v3, v4):
            acc = _merge(acc, value)
    else:
        acc = (seed + P5) & MASK64
    acc = (acc + size) & MASK64
    while pos + 8 <= size:
        acc ^= _round(0, struct.unpack_from("<Q", data, pos)[0])
        acc = (_rotl(acc, 27) * P1 + P4) & MASK64
        pos += 8
    if pos + 4 <= size:
        acc ^= (struct.unpack_from("<I", data, pos)[0] * P1) & MASK64
        acc = (_rotl(acc, 23) * P2 + P3) & MASK64
        pos += 4
    while pos < size:
        acc ^= (data[pos] * P5) & MASK64
        acc = (_rotl(acc, 11) * P1) & MASK64
        pos += 1
    acc ^= acc >> 33
    acc = (acc * P2) & MASK64
    acc ^= acc >> 29
    acc = (acc * P3) & MASK64
    acc ^= acc >> 32
    return acc


assert xxh64(b"") == 0xEF46DB3751D8E999
assert xxh64(b"a") == 0xD24EC4F1A98C6E5B
assert xxh64(b"abc") == 0x44BC2CF5AD770999


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)


def splitmix64_bytes(seed, size):
    rng = SplitMix64(seed)
    out = bytearray()
    while len(out) < size:
        out += rng.next().to_bytes(8, "little")
    return bytes(out[:size])


def palette_bytes(fmt, seed, size, lo, hi):
    count = {C4: size * 2, C8: size, C14X2: size // 2}[fmt]
    rng = SplitMix64(seed)
    indices = []
    for i in range(count):
        r = rng.next()
        index = lo if i == 0 else hi if i == count - 1 else lo + r % (hi - lo + 1)
        indices.append((index, r >> 62))
    out = bytearray()
    if fmt == C4:
        for i in range(0, count, 2):
            out.append((indices[i][0] << 4) | indices[i + 1][0])
    elif fmt == C8:
        out.extend(index for index, _ in indices)
    else:
        for index, noise in indices:
            out += ((noise << 14) | index).to_bytes(2, "big")
    return bytes(out)


def _div_ceil(value, divisor):
    return ((value + divisor - 1) & MASK32) // divisor


def texture_source_size(fmt, width, height, mip_count=1):
    if width == 0 or height == 0 or mip_count == 0:
        return 0
    total = 0
    for mip in range(mip_count):
        w = max(width >> mip, 1)
        h = max(height >> mip, 1)
        if fmt == R8_PC:
            total += w * h
        elif fmt == RG8_PC:
            total += w * h * 2
        elif fmt == RGBA8_PC:
            total += w * h * 4
        elif fmt == BC1_PC:
            total += (_div_ceil(w, 4) * _div_ceil(h, 4) * 8) & MASK32
        elif fmt in (I4, C4, CMPR):
            total += (_div_ceil(w, 8) * _div_ceil(h, 8) * 32) & MASK32
        elif fmt in (I8, IA4, C8):
            total += (_div_ceil(w, 8) * _div_ceil(h, 4) * 32) & MASK32
        elif fmt in (IA8, RGB565, RGB5A3, C14X2):
            total += (_div_ceil(w, 4) * _div_ceil(h, 4) * 32) & MASK32
        elif fmt == RGBA8:
            total += (_div_ceil(w, 4) * _div_ceil(h, 4) * 64) & MASK32
        else:
            return 0
    return total


def hash_source_base(fmt, width, height, texture):
    base_bytes = texture_source_size(fmt, width, height, 1)
    if base_bytes == 0 or len(texture) < base_bytes:
        return None
    base = texture[:base_bytes]
    if fmt == C4:
        indices = [b & 0xF for b in base] + [b >> 4 for b in base]
    elif fmt == C8:
        indices = list(base)
    elif fmt == C14X2:
        indices = [v & 0x3FFF for v in struct.unpack(f">{base_bytes // 2}H", base[: base_bytes & ~1])]
    else:
        indices = []
    if not indices:
        return xxh64(base), MASK32, 0
    return xxh64(base), min(indices), max(indices)


def hash_referenced_tlut(min_index, max_index, tlut):
    if min_index == MASK32:
        return None
    offset = min_index * 2
    size = ((max_index + 1 - min_index) & MASK32) * 2
    if offset + size > len(tlut):
        return None
    return xxh64(tlut[offset : offset + size])


def compute_source_key(fmt, width, height, texture, tlut=b"") -> Optional[SourceKey]:
    if width > 0xFFFF or height > 0xFFFF:
        return None
    base = hash_source_base(fmt, width, height, texture)
    if base is None:
        return None
    has_tlut = fmt in PALETTE_FORMATS
    if has_tlut and not tlut:
        return None
    tlut_hash = 0
    if has_tlut:
        tlut_hash = hash_referenced_tlut(base[1], base[2], tlut) or 0
    return SourceKey(base[0], tlut_hash, width, height, fmt, has_tlut)


def _ascii_lower(text):
    return "".join(chr(ord(ch) + 32) if "A" <= ch <= "Z" else ch for ch in text)


def _parse_hex(text):
    if not text:
        return None
    value = 0
    for ch in text:
        if "0" <= ch <= "9":
            digit = ord(ch) - ord("0")
        elif "a" <= ch <= "f":
            digit = ord(ch) - ord("a") + 10
        elif "A" <= ch <= "F":
            digit = ord(ch) - ord("A") + 10
        else:
            return None
        value = ((value << 4) | digit) & MASK64
    return value


def _parse_u32(text):
    if not text:
        return None
    value = 0
    for ch in text:
        if not "0" <= ch <= "9":
            return None
        value = value * 10 + ord(ch) - ord("0")
        if value > MASK32:
            return None
    return value


def _parse_dimensions(text):
    sep = text.find("x")
    if sep < 0:
        return None
    width = _parse_u32(text[:sep])
    height = _parse_u32(text[sep + 1 :])
    if width is None or height is None:
        return None
    return width, height


def parse_replacement_filename(filename) -> Optional[SourceKey]:
    dot = filename.rfind(".")
    if dot < 0 or _ascii_lower(filename[dot:]) not in (".dds", ".png"):
        return None
    stem = filename[:dot]
    if not stem.startswith("tex1_"):
        return None

    parts = []
    offset = 0
    consumed_all = False
    while offset <= len(stem) and len(parts) < 6:
        sep = stem.find("_", offset)
        if sep < 0:
            parts.append(stem[offset:])
            consumed_all = True
            break
        parts.append(stem[offset:sep])
        offset = sep + 1
    if not consumed_all or not 4 <= len(parts) <= 6 or parts[0] != "tex1":
        return None

    dimensions = _parse_dimensions(parts[1])
    if dimensions is None:
        return None

    index = 3 if parts[2] == "m" else 2
    remaining = len(parts) - index
    if remaining not in (2, 3):
        return None

    if parts[index] == "$":
        texture_hash = WILDCARD_TEXTURE_HASH
    else:
        texture_hash = _parse_hex(parts[index])
        if texture_hash is None:
            return None

    format_part = parts[-1]
    if format_part == "arb":
        format_part = parts[-2]
        remaining -= 1
    fmt = _parse_u32(format_part)
    if fmt is None:
        return None

    tlut_hash = 0
    has_tlut = remaining == 3
    if has_tlut:
        if parts[index + 1] == "$":
            tlut_hash = WILDCARD_TLUT_HASH
        else:
            tlut_hash = _parse_hex(parts[index + 1])
            if tlut_hash is None:
                return None
    return SourceKey(texture_hash, tlut_hash, dimensions[0], dimensions[1], fmt, has_tlut)


def format_key(key: SourceKey):
    out = f"tex1_{key.width}x{key.height}_"
    out += "$" if key.textureHash == WILDCARD_TEXTURE_HASH else f"{key.textureHash:016x}"
    if key.hasTlut:
        out += "_" + ("$" if key.tlutHash == WILDCARD_TLUT_HASH else f"{key.tlutHash:016x}")
    return out + f"_{key.format}"


def is_sidecar_mip(stem):
    digits = len(stem)
    while digits > 0 and "0" <= stem[digits - 1] <= "9":
        digits -= 1
    if digits == len(stem) or digits < 4:
        return False
    return stem[digits - 4 : digits] == "_mip"


GENERATOR_RULES = [
    "splitmix64 next(): state += 0x9e3779b97f4a7c15; z = state; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9; "
    "z = (z ^ (z >> 27)) * 0x94d049bb133111eb; return z ^ (z >> 31), all mod 2^64, state starts at the seed",
    "texture {seed, size}: next() values as 8 little-endian bytes each, truncated to size",
    "texture {seed, size, indices: [lo, hi]}: one next() r per index (C4 2 per byte, C8 1, C14X2 1 per 2 bytes); "
    "index = lo + r % (hi - lo + 1), except the first is lo and the last is hi; C4 packs the earlier index in the "
    "high nibble, C14X2 stores (r >> 62) << 14 | index big-endian",
    "tlut {seed, entries}: bytes as texture {seed, size: 2 * entries}; null passes an empty TLUT",
    "key null: compute_source_key returns nullopt; hashes are 16-digit lowercase hex",
    "baseSize: texture_source_size(format, width, height, 1)",
]


def _case(name, fmt, width, height, seed, size, indices=None, tlut=None):
    return {"name": name, "format": fmt, "width": width, "height": height, "seed": seed, "size": size,
            "indices": indices, "tlut": tlut}


def case_specs():
    specs = []
    seed = 1
    for fmt in (I4, I8, IA4, IA8, RGB565, RGB5A3, RGBA8, CMPR, R8_PC, RG8_PC, RGBA8_PC, BC1_PC):
        for width, height in ((8, 8), (13, 7)):
            name = f"{FORMAT_NAMES[fmt]} {width}x{height}"
            specs.append(_case(name, fmt, width, height, seed, texture_source_size(fmt, width, height)))
            seed += 1
    for fmt in (I4, RGBA8, CMPR, BC1_PC):
        specs.append(_case(f"{FORMAT_NAMES[fmt]} 1x1", fmt, 1, 1, seed, texture_source_size(fmt, 1, 1)))
        seed += 1
    specs.append(_case("CMPR 102x120", CMPR, 102, 120, 101, texture_source_size(CMPR, 102, 120)))
    specs += [
        _case("C4 indices 3..9, 16 entries", C4, 8, 8, 201, 32, [3, 9], {"seed": 202, "entries": 16}),
        _case("C4 indices 3..9 overrun 8 entries", C4, 8, 8, 201, 32, [3, 9], {"seed": 202, "entries": 8}),
        _case("C8 full index range", C8, 16, 16, 203, 256, [0, 255], {"seed": 204, "entries": 256}),
        _case("C8 indices 200..255", C8, 16, 16, 205, 256, [200, 255], {"seed": 206, "entries": 256}),
        _case("C8 indices 4..40 overrun 16 entries", C8, 8, 8, 207, 64, [4, 40], {"seed": 208, "entries": 16}),
        _case("C14X2 indices 1000..1100 noisy top bits", C14X2, 16, 8, 209, 256, [1000, 1100],
              {"seed": 210, "entries": 2048}),
        _case("C8 without TLUT", C8, 8, 8, 211, 64, [0, 255]),
        _case("CMPR 16x16 exact", CMPR, 16, 16, 301, 128),
        _case("CMPR 16x16 plus 64 bytes", CMPR, 16, 16, 301, 192),
        _case("RGBA8_PC 65535x1", RGBA8_PC, 65535, 1, 302, 65535 * 4),
        _case("RGBA8_PC 65536x1", RGBA8_PC, 65536, 1, 303, 65536 * 4),
        _case("I8 1x65536", I8, 1, 65536, 304, texture_source_size(I8, 1, 65536)),
        _case("format 7", 7, 8, 8, 305, 64),
        _case("Z8", Z8, 8, 8, 306, 64),
        _case("RGB565 8x8 short by 1", RGB565, 8, 8, 307, 127),
        _case("I8 width 0", I8, 0, 8, 308, 32),
    ]
    return specs


NAME_SPECS = [
    "tex1_128x128_0017919ec19004f3_14.dds",
    "tex1_128x128_m_005c13647e8d4744_14.dds",
    "tex1_256x256_289bd6f7a88282d2_81df6185a52bd4da_9.dds",
    "tex1_102x120_f3773035018b6280_459566e922a89796_9.dds",
    "tex1_112x144_a1c955530b5dc510_2.dds",
    "tex1_16x16_$_fedcba9876543210_8.dds",
    "tex1_16x16_0123456789abcdef_$_9.dds",
    "tex1_8x8_m_$_$_10.dds",
    "tex1_256x256_0123456789abcdef_14_arb.dds",
    "tex1_128x128_0017919EC19004F3_14.PNG",
    "tex1_8x8_1f_6.dds",
    "tex1_8x8_10123456789abcdef0_6.dds",
    "tex1_8x8_14_arb.dds",
    "tex1_8x8_0123456789abcdef_1_4.dds",
    "tex1_4294967295x1_0123456789abcdef_4294967295.dds",
    "tex1_0x8_0123456789abcdef_14.dds",
    "tex1_+8x8_0123456789abcdef_14.dds",
    "tex1_8x8_0x1f_14.dds",
    "tex1_8x8_0123456789abcdef_ 14.dds",
    "tex1_\uff18x8_0123456789abcdef_14.dds",
    "tex1_8x8_0123456789abcdef_4294967296.dds",
    "tex1_8x8__14.dds",
    "tex1_8x8_0123456789abcdef_14_.dds",
    "tex1_128x128_0017919ec19004f3_14_mip1.dds",
    "tex1_128x128_0017919ec19004f3_14.mat.dds",
    "tex1_128x128_0017919ec19004f3_14.jpg",
    "tex1_128x128_0017919ec19004f3_14",
    "tex1_8x8_m_14.dds",
    "tex1_8x8_m_0123456789abcdef_0123456789abcdef_9_arb.dds",
    "tex2_8x8_0123456789abcdef_14.dds",
]


def texture_for(spec):
    if spec["indices"] is not None:
        return palette_bytes(spec["format"], spec["seed"], spec["size"], *spec["indices"])
    return splitmix64_bytes(spec["seed"], spec["size"])


def tlut_for(spec):
    tlut = spec["tlut"]
    return b"" if tlut is None else splitmix64_bytes(tlut["seed"], tlut["entries"] * 2)


def _hex(value):
    return f"{value:016x}"


def build_fixture():
    cases = []
    for spec in case_specs():
        key = compute_source_key(spec["format"], spec["width"], spec["height"], texture_for(spec), tlut_for(spec))
        texture = {"seed": spec["seed"], "size": spec["size"]}
        if spec["indices"] is not None:
            texture["indices"] = spec["indices"]
        cases.append({
            "name": spec["name"],
            "format": spec["format"],
            "width": spec["width"],
            "height": spec["height"],
            "baseSize": texture_source_size(spec["format"], spec["width"], spec["height"]),
            "texture": texture,
            "tlut": spec["tlut"],
            "key": None if key is None else {
                "textureHash": _hex(key.textureHash),
                "tlutHash": _hex(key.tlutHash),
                "hasTlut": key.hasTlut,
            },
        })
    names = []
    for name in NAME_SPECS:
        key = parse_replacement_filename(name)
        entry = {"name": name, "key": None}
        if key is not None:
            entry["key"] = {
                "textureHash": _hex(key.textureHash),
                "tlutHash": _hex(key.tlutHash),
                "width": key.width,
                "height": key.height,
                "format": key.format,
                "hasTlut": key.hasTlut,
            }
            entry["canonical"] = format_key(key)
        names.append(entry)
    return {
        "version": 1,
        "generator": {"tool": "texture_key.py fixture --write", "rules": GENERATOR_RULES},
        "cases": cases,
        "names": names,
    }


def run_fixture(write):
    fixture = build_fixture()
    if write:
        FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {FIXTURE.name}: {len(fixture['cases'])} cases, {len(fixture['names'])} names")
        return 0
    try:
        stored = json.loads(FIXTURE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"{FIXTURE.name}: {error}", file=sys.stderr)
        return 1
    if stored != fixture:
        print(f"{FIXTURE.name} is stale; run fixture --write", file=sys.stderr)
        return 1
    print(f"{FIXTURE.name} ok: {len(fixture['cases'])} cases, {len(fixture['names'])} names")
    return 0


def run_scan(root):
    files = undotted = parsed = tlut = 0
    keys = set()
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            if not os.path.isfile(os.path.join(dirpath, name)):
                continue
            stem, ext = os.path.splitext(name)
            if _ascii_lower(ext) not in (".dds", ".png") or is_sidecar_mip(stem):
                continue
            files += 1
            undotted += "." not in stem
            key = parse_replacement_filename(name)
            if key is None:
                continue
            parsed += 1
            tlut += key.hasTlut
            keys.add(key)
    print(f"files {files} ({undotted} undotted stems), parsed {parsed}, unique {len(keys)}, TLUT {tlut}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("fixture")
    mode = fixture.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    scan = commands.add_parser("scan")
    scan.add_argument("dir", type=Path)
    args = parser.parse_args()
    if args.command == "fixture":
        return run_fixture(args.write)
    return run_scan(args.dir)


if __name__ == "__main__":
    sys.exit(main())
