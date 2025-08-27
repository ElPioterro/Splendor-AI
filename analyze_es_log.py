# analyze_es_log.py
import csv
from pathlib import Path


p = Path("logs/es_log.csv")
rows = []
with p.open() as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append({
            "gen": int(row["gen"]),
            "wr": float(row["wr"]),
            "wr_lb": float(row["wr_lb"]),
            "score": float(row["score"]),
        })

def rolling_mean(xs, k):
    out=[]
    for i in range(len(xs)):
        s = xs[max(0, i-k+1): i+1]
        out.append(sum(s)/len(s))
    return out

wr     = [r["wr"] for r in rows]
wr_lb  = [r["wr_lb"] for r in rows]
score  = [r["score"] for r in rows]
gens   = [r["gen"] for r in rows]

# Rolling
r10 = rolling_mean(wr_lb, 10)
r20 = rolling_mean(wr_lb, 20)

# Trend 20-gen (początek vs koniec)
beg20 = sum(wr_lb[:20])/min(20, len(wr_lb))
end20 = sum(wr_lb[-20:])/min(20, len(wr_lb))
delta = end20 - beg20

# Rozkład binów WR
bins = {0.0:0, 0.25:0, 0.5:0, 0.75:0, 1.0:0}
for w in wr:
    if w in bins: bins[w]+=1

print(f"Rolling wr_lb (last 10 avg): {r10[-1]:.3f}")
print(f"Trend 20-gen: begin={beg20:.3f} -> end={end20:.3f} (Δ={delta:+.3f})")
print("WR distribution (count per bin):", bins)

# Heurystyka GO/NO-GO:
go = (r10[-1] >= 0.35) or (end20 >= beg20 + 0.05)
print("Decision:", "GO (continue & increase eval fidelity)" if go else "NO-GO (tune eval or hyperparams)")