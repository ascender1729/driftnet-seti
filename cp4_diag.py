#!/usr/bin/env python3
"""Diagnose the cp4 .h5 write/read: is the injected signal surviving the round-trip?"""
import glob, os, numpy as np
import cp4_cadence_roc as C4
import cp3_inject_real as C3
C3.SKIP_PNG = True

cads = C4.get_cadences("real_data/cadence")
cad = cads[0]
import h5py
with h5py.File(cad[0]) as hf: nch = hf["data"].shape[2]
rng = np.random.default_rng(1)
center = C4.edge_center(nch, rng)
frame = C3.bg_frame(cad[0], center, fchans=C4.FCHANS)

# inject SNR=30 line
arr, f0 = C4.inject_line(frame, 30.0, 0.2)
print("IN-MEMORY injected array: shape", arr.shape, "dtype", arr.dtype)
print("  bg median", np.median(frame.data), "bg std", np.std(frame.data))
col = C4.INJ_IDX
print(f"  injected col {col}: mem max {arr[:,col].max():.3e}  bg col max {np.asarray(frame.data)[:,col].max():.3e}")
print(f"  array global max {arr.max():.3e} at col {np.unravel_index(arr.argmax(), arr.shape)[1]}")

# write then read back
os.makedirs("diag", exist_ok=True)
C4.write_h5(arr, frame, "diag/t.h5", "SYNTH_ON", 59000.0)
from blimpy import Waterfall
wf = Waterfall("diag/t.h5", load_data=True)
rb = np.squeeze(wf.data).astype(np.float32)
print("READBACK: shape", rb.shape, "dtype", rb.dtype)
print(f"  readback col {col} max {rb[:,col].max():.3e}  global max {rb.max():.3e} at col {np.unravel_index(rb.argmax(), rb.shape)[1]}")
print("  readback == inmem?", np.allclose(rb, arr, rtol=1e-3), " maxdiff", np.abs(rb.astype(float)-arr.astype(float)).max())

# DriftNet scores on both
import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
net = C4.train_driftnet("real_data/cadence", dev)
with torch.no_grad():
    p_mem = torch.softmax(net(torch.tensor(arr[None]).to(dev))[0], 1)[0]
    p_rb = torch.softmax(net(torch.tensor(rb[None]).to(dev))[0], 1)[0]
print("DriftNet signal-prob: in-memory", float(p_mem[0]), " readback", float(p_rb[0]))
print("  full probs mem", [round(float(x),3) for x in p_mem], " rb", [round(float(x),3) for x in p_rb])
# FLIP test: does reversing the readback frequency axis recover it?
rbf = np.ascontiguousarray(rb[:, ::-1])
with torch.no_grad():
    p_flip = torch.softmax(net(torch.tensor(rbf[None]).to(dev))[0], 1)[0]
print("  flipped readback signal-prob", float(p_flip[0]), "probs", [round(float(x),3) for x in p_flip])
# turboSETI on the single frame: does it find the line at all?
from turbo_seti.find_doppler.find_doppler import FindDoppler
FindDoppler("diag/t.h5", max_drift=2.0, snr=8.0, n_coarse_chan=1, out_dir="diag/").search()
dat = "diag/t.dat"
hits = [l.strip() for l in open(dat) if l.strip() and not l.startswith("#")] if os.path.exists(dat) else []
print(f"turboSETI hits in single frame: {len(hits)}")
for h in hits[:5]: print("   ", h)
print("DIAG_DONE")
