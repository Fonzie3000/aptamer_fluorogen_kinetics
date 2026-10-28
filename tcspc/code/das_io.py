"""
Reader for Horiba DAS6 `.das` files and their paired `*_fit.txt` reports.

A `.das` file is an OLE2 compound document. Relevant streams:

    Decays/<n>          float32[4096] after an 8-byte header, photon counts
    Prompts/<n>         float32[4096] after an 8-byte header, the measured IRF
                        (Ludox scattering standard at the excitation wavelength),
                        recorded once per acquisition
    Decays/ElementCount int16, number of acquisitions in the batch
    Fits/<n>/<k>/...    the vendor's own fits (k=1 bi, k=2 tri): In, Out, Info, Name
    Information/TCal    float32 channel width - NOT SET in these files (reads 0.0),
                        so the channel width is taken from the acquisition record

Header: bytes 0-3 = element index (int32), bytes 4-7 = channel count (int32).

Channel width is 0.055176 ns from the acquisition record (DeltaFlex, 4096 channels,
226 ns window); 4096 * 0.055176 = 226.0 ns, which is consistent. The `.das` files
themselves report "Time calibration = not set", so this value is external to the file.
"""

from __future__ import annotations

import os
import re
import struct

import numpy as np
import olefile

NS_PER_CHANNEL = 0.055176        # from the acquisition record, not from the file
N_CHANNELS = 4096


def _utf16_pstring(b: bytes) -> str:
    """Length-prefixed UTF-16LE string as used by the Information/* streams."""
    n = struct.unpack("<I", b[:4])[0]
    return b[4 : 4 + 2 * n].decode("utf-16-le", "replace")


def _counts(ole: olefile.OleFileIO, path: str) -> np.ndarray:
    raw = ole.openstream(path).read()
    idx, nch = struct.unpack("<ii", raw[:8])
    a = np.frombuffer(raw[8:], dtype="<f4").astype(float)
    assert a.size == nch, f"{path}: header says {nch} channels, got {a.size}"
    return a


def read_das(path: str) -> dict:
    """All acquisitions in one .das file.

    Returns {file, name, measurement, n_acq, decays, prompts, vendor_fits} where
    `decays` and `prompts` are (n_acq, 4096) arrays of counts, row i being acquisition i+1.
    """
    ole = olefile.OleFileIO(path)
    try:
        listing = {"/".join(s) for s in ole.listdir(streams=True)}
        n_acq = len([s for s in listing if re.fullmatch(r"Decays/\d+", s)])
        decays = np.vstack([_counts(ole, f"Decays/{i}") for i in range(1, n_acq + 1)])
        prompts = np.vstack([_counts(ole, f"Prompts/{i}") for i in range(1, n_acq + 1)])
        info = {}
        for key in ("Name", "Measurement", "BuiltBy"):
            p = f"Information/{key}"
            if p in listing:
                info[key.lower()] = _utf16_pstring(ole.openstream(p).read())
        tcal = None
        if "Information/TCal" in listing:
            tcal = struct.unpack("<f", ole.openstream("Information/TCal").read())[0]
        vendor = {}
        for i in range(1, n_acq + 1):
            for k in (1, 2):
                base = f"Fits/{i}/{k}"
                if f"{base}/Name" not in listing:
                    continue
                vendor[(i, k)] = dict(
                    name=_utf16_pstring(ole.openstream(f"{base}/Name").read()),
                    out=np.frombuffer(ole.openstream(f"{base}/Out").read(), "<f4").astype(float),
                    info=np.frombuffer(ole.openstream(f"{base}/Info").read(), "<f4").astype(float),
                )
        return dict(file=os.path.basename(path), n_acq=n_acq, decays=decays,
                    prompts=prompts, tcal_in_file=tcal, vendor_fits=vendor, **info)
    finally:
        ole.close()


_FIT_SCALARS = {
    "lo_ch": r"Prompt and decay LO\s*=\s*([-\d.Ee+]+)",
    "hi_ch": r"Prompt and decay HI\s*=\s*([-\d.Ee+]+)",
    "hi_reduced_ch": r"Hi reduced to:\s*([-\d.Ee+]+)",
    "shift_ch": r"SHIFT\s*=\s*([-\d.Ee+]+)",
    "chisq": r"CHISQ\s*=\s*([-\d.Ee+]+)",
    "dof": r"\[\s*(\d+)\s*degrees of freedom",
    "durbin_watson": r"Durbin-Watson Parameter\s*=\s*([-\d.Ee+]+)",
    "background_on_prompt": r"Background on prompt\s*=\s*([-\d.Ee+]+)",
}


def read_fit_txt(path: str) -> dict:
    """The DAS6 text report: fit window, shift, component lifetimes (channels),
    chi-square, Durbin-Watson, and the Chan/Prompt/Decay/Fit/Residuals table.

    These reports are PER ACQUISITION and carry NO scatter term, so they are the
    like-with-like reference for a no-scatter validation run.
    """
    txt = open(path, encoding="utf-8", errors="replace").read()
    out = {"file": os.path.basename(path)}
    for key, pat in _FIT_SCALARS.items():
        m = re.search(pat, txt)
        out[key] = (int(m.group(1)) if key == "dof" else float(m.group(1))) if m else None
    out["n_exp"] = int(re.search(r"Calculated using (\d+) exponentials", txt).group(1))
    # component lifetimes and relative amplitudes, in channels and per cent
    out["tau_ch"] = [float(v) for v in re.findall(r"^ T\d = ([-\d.Ee+]+) ch", txt, re.M)]
    out["rel_ampl_pct"] = [float(v) for v in re.findall(r"\[\s*([-\d.Ee+]+) Rel\.Ampl\]", txt)]
    out["B"] = [float(v) for v in re.findall(r"^ B\d = ([-\d.Ee+]+)", txt, re.M)]
    # the trailing per-channel table
    rows = re.findall(r"^(\d+)\t([-\d.Ee+]+)\t([-\d.Ee+]+)\t([-\d.Ee+]+)\t([-\d.Ee+]+)",
                      txt, re.M)
    if rows:
        arr = np.array(rows, dtype=float)
        out["table"] = dict(chan=arr[:, 0].astype(int), prompt=arr[:, 1],
                            decay=arr[:, 2], fit=arr[:, 3], residual=arr[:, 4])
    return out


def construct_from_filename(fn: str) -> tuple[str, int]:
    """'12bpSALAD_batch.das' -> ('SALAD1', 12). 'MII' is TO1-Biotin."""
    m = re.match(r"(\d+)bp(MII|SALAD)", fn)
    if not m:
        raise ValueError(f"cannot parse construct from {fn!r}")
    bp, tag = int(m.group(1)), m.group(2)
    return ("TO1-B" if tag == "MII" else "SALAD1"), bp


def load_folder(folder: str) -> list[dict]:
    """Every *_batch.das in `folder`, with its paired bi/tri fit reports attached."""
    out = []
    for fn in sorted(f for f in os.listdir(folder) if f.endswith("_batch.das")):
        rec = read_das(os.path.join(folder, fn))
        rec["dye"], rec["stem_bp"] = construct_from_filename(fn)
        stem = fn.replace("_batch.das", "")
        for kind in ("bi", "tri"):
            p = os.path.join(folder, f"{stem}_{kind}_fit.txt")
            rec[f"{kind}_fit"] = read_fit_txt(p) if os.path.exists(p) else None
        out.append(rec)
    return out
