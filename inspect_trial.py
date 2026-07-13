import glob, os
d = "cp4c/X/trial_signal_30_0"
for dat in sorted(glob.glob(f"{d}/f*.dat")):
    hits = [l for l in open(dat) if l.strip() and not l.startswith("#")]
    print(os.path.basename(dat), "hits:", len(hits))
    for h in hits[:1]: print("   ", h.strip()[:160])
csv = f"{d}/events.csv"
print("events.csv exists:", os.path.exists(csv))
if os.path.exists(csv):
    rows = open(csv).read().splitlines()
    print("csv lines:", len(rows))
    for r in rows[:3]: print("   CSV:", r[:180])
