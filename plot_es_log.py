# plot_es_log.py
import csv
from pathlib import Path
import matplotlib.pyplot as plt

def load_csv(path):
    rows=[]
    with Path(path).open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def to_float(row, key, default=0.0):
    try: return float(row[key])
    except: return default

def main():
    es = load_csv("logs/es_log.csv")
    gens = [int(r["gen"]) for r in es]
    wr_lb = [to_float(r, "wr_lb") for r in es]
    wr    = [to_float(r, "wr") for r in es]
    score = [to_float(r, "score") for r in es]

    fig, ax = plt.subplots(1,1, figsize=(11,4))
    ax.plot(gens, wr_lb, label="wr_lb", lw=2)
    ax.plot(gens, wr, label="wr", alpha=0.5)
    ax.plot(gens, score, label="score", alpha=0.7)
    ax.set_xlabel("generation"); ax.set_title("ES training")
    ax.grid(True); ax.legend()

    hold = Path("logs/es_holdout_log.csv")
    if hold.exists():
        ho = load_csv(str(hold))
        hg = [int(r["gen"]) for r in ho]
        hwr_lb = [to_float(r, "wr_lb") for r in ho]
        ax2 = ax.twinx()
        ax2.plot(hg, hwr_lb, "k--", label="hold_wr_lb", alpha=0.8)
        ax2.set_ylabel("hold_wr_lb")
        ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()