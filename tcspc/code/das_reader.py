#!/usr/bin/env python3
"""
das_reader.py -- Read HORIBA DataStation/EzTime/DAS6 .das TCSPC files.

A .das file is an OLE2 compound document (the same container as legacy .doc/.xls).
This module contains a minimal, dependency-free OLE2 reader plus a decoder for the
HORIBA "DAS Data Set" layout, so it runs on stock Python (only the standard library).

Verified layout (DeltaFlex / DataStation, "version 1.0" DAS Data Set):
  /Information/TCal          float32  ns per channel  (NOTE: often 0.0 in batch files)
  /Information/Name          str      file/measurement name
  /Information/Measurement   str      e.g. "General TCSPC", "Sequential TCSPC"
  Single-decay file:
    /Decays/1/Data           8-byte header (int32 n_curves, int32 n_channels) + n float32 counts
    /Decays/1/Weights        n float32 fit weights
    /Prompts/1/Data          IRF, same layout
  Batch/sequential file:
    /Decays/<k>              curve k (same 8-byte header + float32 counts)
    /Prompts/<k>             IRF k
    /Fits/<i>/<j>/{In,Out,Name,...}  fit parameter blocks (float32 vectors)

Strings are stored as: int32 length-in-characters, then UTF-16LE.

Usage:
    from das_reader import read_das
    d = read_das("file.das")
    # d["tcal_ns"], d["decays"] -> list of dicts {name, counts(list), time_ns(list|None)}

CLI:
    python3 das_reader.py file.das [more.das ...]      # prints summary + writes <name>.csv
"""
import struct, sys, os

# ----------------------------- minimal OLE2 / CFB reader -----------------------------
_ENDOFCHAIN = 0xFFFFFFFE
_FREESECT   = 0xFFFFFFFF

class _CFB:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.d = f.read()
        d = self.d
        if d[:8] != b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
            raise ValueError("Not an OLE2 compound file: %s" % path)
        self.ss   = 1 << struct.unpack("<H", d[0x1e:0x20])[0]   # sector size
        self.mss  = 1 << struct.unpack("<H", d[0x20:0x22])[0]   # mini sector size
        self.first_dir     = struct.unpack("<I", d[0x30:0x34])[0]
        self.mini_cutoff   = struct.unpack("<I", d[0x38:0x3c])[0]
        self.first_minifat = struct.unpack("<I", d[0x3c:0x40])[0]
        self.first_difat   = struct.unpack("<I", d[0x44:0x48])[0]
        self.num_difat     = struct.unpack("<I", d[0x48:0x4c])[0]
        difat = list(struct.unpack("<109I", d[0x4c:0x4c+436]))
        sec = self.first_difat
        for _ in range(self.num_difat):
            off = (sec+1)*self.ss
            vals = struct.unpack("<%dI" % (self.ss//4), d[off:off+self.ss])
            difat += list(vals[:-1]); sec = vals[-1]
            if sec in (_ENDOFCHAIN, _FREESECT):
                break
        self.difat = [x for x in difat if x != _FREESECT]
        fat = []
        for fs in self.difat:
            off = (fs+1)*self.ss
            fat += list(struct.unpack("<%dI" % (self.ss//4), d[off:off+self.ss]))
        self.fat = fat
        dir_bytes = self._chain(self.first_dir)
        self.ents, self.meta = [], []
        for i in range(0, len(dir_bytes), 128):
            e = dir_bytes[i:i+128]
            if len(e) < 128: break
            nlen = struct.unpack("<H", e[0x40:0x42])[0]
            name = e[0:max(nlen-2, 0)].decode("utf-16-le", "replace") if nlen >= 2 else ""
            self.ents.append(dict(
                name=name, type=e[0x42],
                start=struct.unpack("<I", e[0x74:0x78])[0],
                size=struct.unpack("<Q", e[0x78:0x80])[0]))
            self.meta.append((struct.unpack("<I", e[0x44:0x48])[0],
                              struct.unpack("<I", e[0x48:0x4c])[0],
                              struct.unpack("<I", e[0x4c:0x50])[0]))
        root = next(i for i, e in enumerate(self.ents) if e["type"] == 5)
        self.mini_stream = self._chain(self.ents[root]["start"])[:self.ents[root]["size"]]
        mf = self._chain(self.first_minifat) if self.first_minifat not in (_ENDOFCHAIN, _FREESECT) else b""
        self.minifat = list(struct.unpack("<%dI" % (len(mf)//4), mf)) if mf else []
        # build path -> entry
        self.paths = {}
        _, _, rchild = self.meta[root]
        self._walk(rchild, "")

    def _chain(self, start):
        out, sec, seen = [], start, set()
        while sec not in (_ENDOFCHAIN, _FREESECT) and sec < len(self.fat) and sec not in seen:
            seen.add(sec); off = (sec+1)*self.ss
            out.append(self.d[off:off+self.ss]); sec = self.fat[sec]
        return b"".join(out)

    def _mini_chain(self, start, size):
        out, sec, seen = [], start, set()
        while sec not in (_ENDOFCHAIN, _FREESECT) and sec < len(self.minifat) and sec not in seen:
            seen.add(sec); off = sec*self.mss
            out.append(self.mini_stream[off:off+self.mss]); sec = self.minifat[sec]
        return b"".join(out)[:size]

    def _walk(self, idx, prefix):
        if idx == 0xFFFFFFFF or idx >= len(self.ents):
            return
        left, right, child = self.meta[idx]
        self._walk(left, prefix)
        p = prefix + "/" + self.ents[idx]["name"]
        self.paths[p] = self.ents[idx]
        if child != 0xFFFFFFFF:
            self._walk(child, p)
        self._walk(right, prefix)

    def read(self, path):
        e = self.paths[path]
        if e["type"] == 2 and e["size"] < self.mini_cutoff:
            return self._mini_chain(e["start"], e["size"])
        return self._chain(e["start"])[:e["size"]]

    def has(self, path):
        return path in self.paths

# ----------------------------- HORIBA DAS decoding -----------------------------
def _str(b):
    n = struct.unpack("<i", b[:4])[0]
    return b[4:4+n*2].decode("utf-16-le", "replace")

def _curve(b):
    """8-byte header (n_curves, n_channels) then float32 counts -> list[float]."""
    n = (len(b) - 8) // 4
    return list(struct.unpack("<%df" % n, b[8:8+4*n]))

def _f32vec(b):
    return list(struct.unpack("<%df" % (len(b)//4), b))

def read_das(path):
    c = _CFB(path)
    out = {"path": path, "meta": {}, "tcal_ns": None, "decays": [], "prompts": [], "fits": []}

    if c.has("/Information/TCal"):
        out["tcal_ns"] = struct.unpack("<f", c.read("/Information/TCal"))[0]
    for k in ("Name", "Measurement", "HubType"):
        p = "/Information/" + k
        if c.has(p):
            out["meta"][k] = _str(c.read(p))
    if c.has("/Version"):
        out["meta"]["Version"] = _str(c.read("/Version"))

    tcal = out["tcal_ns"] if out["tcal_ns"] else None  # 0.0 -> treat as unknown

    def axis(n):
        return [i*tcal for i in range(n)] if tcal else None

    def collect(group, dst):
        # single-decay layout: /<group>/1/Data   ;  batch layout: /<group>/1, /2, ...
        for k in range(1, 1000):
            data_p = "/%s/%d/Data" % (group, k)   # single
            flat_p = "/%s/%d" % (group, k)         # batch
            if c.has(data_p):
                name = _str(c.read("/%s/%d/Name" % (group, k))) if c.has("/%s/%d/Name" % (group, k)) else str(k)
                counts = _curve(c.read(data_p))
            elif c.has(flat_p) and c.paths[flat_p]["type"] == 2:
                name = str(k); counts = _curve(c.read(flat_p))
            else:
                break
            dst.append({"index": k, "name": name, "counts": counts, "time_ns": axis(len(counts))})

    collect("Decays", out["decays"])
    collect("Prompts", out["prompts"])

    # fit blocks (parameters only; semantic mapping is DAS6-defined)
    i = 1
    while c.has("/Fits/%d/1/Data" % i) or c.has("/Fits/%d/1/In" % i):
        j = 1; fits_i = []
        while c.has("/Fits/%d/%d/In" % (i, j)):
            base = "/Fits/%d/%d" % (i, j)
            fit = {"name": _str(c.read(base+"/Name")) if c.has(base+"/Name") else "",
                   "in":  _f32vec(c.read(base+"/In"))  if c.has(base+"/In")  else [],
                   "out": _f32vec(c.read(base+"/Out")) if c.has(base+"/Out") else []}
            fits_i.append(fit); j += 1
        out["fits"].append(fits_i); i += 1
    return out

def to_csv(das, csv_path):
    """Write channel, time_ns, and one column per curve (decays then prompts)."""
    curves = [("decay_%d_%s" % (d["index"], d["name"]), d["counts"]) for d in das["decays"]]
    curves += [("prompt_%d_%s" % (p["index"], p["name"]), p["counts"]) for p in das["prompts"]]
    if not curves:
        return
    n = max(len(v) for _, v in curves)
    tcal = das["tcal_ns"] if das["tcal_ns"] else None
    with open(csv_path, "w", encoding="utf-8") as f:
        hdr = ["channel"] + (["time_ns"] if tcal else []) + [nm for nm, _ in curves]
        f.write(",".join(hdr) + "\n")
        for ch in range(n):
            row = [str(ch)]
            if tcal: row.append("%.6f" % (ch*tcal))
            for _, v in curves:
                row.append("%.6g" % v[ch] if ch < len(v) else "")
            f.write(",".join(row) + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(0)
    for path in sys.argv[1:]:
        d = read_das(path)
        base = os.path.splitext(os.path.basename(path))[0]
        csv_path = os.path.join(os.path.dirname(path) or ".", base + ".csv")
        to_csv(d, csv_path)
        tc = d["tcal_ns"]
        print("== %s ==" % os.path.basename(path))
        print("   measurement : %s" % d["meta"].get("Measurement", "?"))
        print("   tcal        : %s" % ("%.6f ns/ch (%.1f ps/ch)" % (tc, tc*1000) if tc else "0 / not stored -- supply from single-decay file"))
        print("   decays      : %d   prompts: %d   fit-sets: %d" % (len(d["decays"]), len(d["prompts"]), len(d["fits"])))
        for dec in d["decays"]:
            v = dec["counts"]; pk = max(range(len(v)), key=lambda k: v[k])
            print("     decay %-2s peak=%.0f @ch%d  total=%.0f" % (dec["index"], v[pk], pk, sum(v)))
        print("   -> wrote %s" % csv_path)
