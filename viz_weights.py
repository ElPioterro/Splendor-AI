# viz_weights.py
from __future__ import annotations
import os, re, glob, argparse
import numpy as np
import matplotlib.pyplot as plt

import nn_config as cfg
# Te wartości mogą nie zgadzać się ze snapshotem, używamy ich tylko jako domyślne do etykiet,
# jeśli suma zgadza się z in_dim snapshotu.
from feature_encoder import STATE_DIM as FE_STATE_DIM, MOVE_DIM as FE_MOVE_DIM

def param_shapes(in_dim: int, h1: int, h2: int):
    # W1, b1, W2, b2, W3, b3
    return [(in_dim, h1), (h1,), (h1, h2), (h2,), (h2, 1), (1,)]

def infer_arch_from_theta_len(theta_len: int, candidates=(16, 24, 32, 48, 64, 96, 128)):
    """
    Zwraca (in_dim, h1, h2) dopasowane do długości theta.
    Szuka po rozsądnym zbiorze kandydatów h1/h2.
    """
    sols = []
    for h1 in candidates:
        for h2 in candidates:
            # P = in_dim*h1 + h1*h2 + h1 + 2*h2 + 1
            num = theta_len - (h1*h2 + h1 + 2*h2 + 1)
            if num <= 0: 
                continue
            if num % h1 != 0:
                continue
            in_dim = num // h1
            if in_dim > 0:
                sols.append((in_dim, h1, h2))
    # heurystyka wyboru: najbliżej aktualnego cfg (jeśli jest), inaczej najmniejsza
    if not sols:
        return None
    # preferuj zgodność z cfg.HIDDEN1/HIDDEN2
    sols.sort(key=lambda t: (abs(t[1]-getattr(cfg, "HIDDEN1", 64)) + abs(t[2]-getattr(cfg, "HIDDEN2", 64)), t[0]))
    return sols[0]

def unflatten(theta: np.ndarray, in_dim: int, h1: int, h2: int):
    shapes = param_shapes(in_dim, h1, h2)
    params = []
    idx = 0
    for s in shapes:
        n = int(np.prod(s))
        params.append(theta[idx: idx + n].reshape(s))
        idx += n
    if idx != theta.size:
        raise ValueError(f"theta size {theta.size} != expected {idx} for (in={in_dim},h1={h1},h2={h2})")
    return params  # W1,b1,W2,b2,W3,b3

def plot_heatmaps(W1, W2, W3, outfile: str, title: str):
    vmax = np.max(np.abs(np.concatenate([W1.ravel(), W2.ravel(), W3.ravel()])))
    vmax = max(vmax, 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    im0 = axes[0].imshow(W1, cmap="coolwarm", vmin=-vmax, vmax=+vmax, aspect="auto")
    axes[0].set_title("W1: inputs → H1"); axes[0].set_xlabel("H1"); axes[0].set_ylabel("inputs")
    axes[1].imshow(W2, cmap="coolwarm", vmin=-vmax, vmax=+vmax, aspect="auto")
    axes[1].set_title("W2: H1 → H2"); axes[1].set_xlabel("H2"); axes[1].set_ylabel("H1")
    axes[2].imshow(W3.T, cmap="coolwarm", vmin=-vmax, vmax=+vmax, aspect="auto")
    axes[2].set_title("W3: H2 → out"); axes[2].set_xlabel("H2"); axes[2].set_ylabel("out")
    plt.suptitle(title)
    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.7)
    cbar.set_label("weight")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=150); plt.close(fig)

def plot_importance(W1, outfile: str, title: str, state_dim: int | None, move_dim: int | None):
    imp = np.sum(np.abs(W1), axis=1)  # [in_dim]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    if state_dim is not None and move_dim is not None and (state_dim + move_dim == W1.shape[0]):
        imp_state = float(np.sum(imp[:state_dim]))
        imp_move  = float(np.sum(imp[state_dim:state_dim+move_dim]))
        axes[0].bar(["state", "move"], [imp_state, imp_move], color=["#4e79a7", "#e15759"])
        axes[0].set_title("Input importance: state vs move")
        axes[0].set_ylabel("sum |weights|")
        # Top-k cech
        k = min(15, imp.shape[0])
        idx = np.argsort(imp)[-k:][::-1]
        labels = [f"s[{i}]" if i < state_dim else f"m[{i-state_dim}]" for i in idx]
        axes[1].bar(range(k), imp[idx], color="#76b7b2")
        axes[1].set_title("Top input features by |weight| (W1)")
        axes[1].set_xticks(range(k)); axes[1].set_xticklabels(labels, rotation=45, ha="right")
    else:
        # Nie znamy poprawnego podziału state/move -> pokazujemy tylko total i top-k
        axes[0].bar(["inputs"], [float(np.sum(imp))], color="#4e79a7")
        axes[0].set_title("Input importance: total (unknown split)")
        axes[0].set_ylabel("sum |weights|")
        k = min(15, imp.shape[0])
        idx = np.argsort(imp)[-k:][::-1]
        labels = [f"in[{i}]" for i in idx]
        axes[1].bar(range(k), imp[idx], color="#76b7b2")
        axes[1].set_title("Top inputs by |weight| (W1)")
        axes[1].set_xticks(range(k)); axes[1].set_xticklabels(labels, rotation=45, ha="right")
    plt.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=150); plt.close(fig)

def extract_gen(path: str) -> int:
    m = re.search(r"gen_(\d+)", path)
    return int(m.group(1)) if m else -1

def main():
    ap = argparse.ArgumentParser(description="Viz NN weights from ES snapshots")
    ap.add_argument("--snapdir", type=str, default=getattr(cfg, "SNAPSHOT_DIR", "snapshots"))
    ap.add_argument("--outdir", type=str, default="viz")
    ap.add_argument("--gens", type=int, nargs="*", default=None, help="np. --gens 10 20 30")
    ap.add_argument("--state-dim", type=int, default=None, help="Opcjonalnie: ręcznie podaj STATE_DIM snapshotu")
    ap.add_argument("--move-dim", type=int, default=None, help="Opcjonalnie: ręcznie podaj MOVE_DIM snapshotu")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.snapdir, "theta_gen_*.npy")))
    if not files:
        print(f"Brak snapshotów w {args.snapdir}. Włącz snapshoty w es_manager + nn_config.")
        return

    if args.gens:
        sel = [os.path.join(args.snapdir, f"theta_gen_{g:04d}.npy") for g in args.gens if os.path.exists(os.path.join(args.snapdir, f"theta_gen_{g:04d}.npy"))]
        files = sel if sel else files[:1]

    os.makedirs(args.outdir, exist_ok=True)

    for path in files:
        g = extract_gen(path)
        theta = np.load(path).astype(np.float32).reshape(-1)
        arch = infer_arch_from_theta_len(theta_len=theta.size)
        if arch is None:
            print(f"[viz] Nie potrafię dopasować architektury do len={theta.size}. Podaj ręcznie --state-dim/--move-dim i H1/H2 w nn_config.")
            continue
        in_dim, h1, h2 = arch
        if (h1, h2) != (getattr(cfg, "HIDDEN1", h1), getattr(cfg, "HIDDEN2", h2)):
            print(f"[viz] Uwaga: snapshot wygląda na H1={h1}, H2={h2}, in_dim={in_dim} (nie jak w nn_config).")
        W1, b1, W2, b2, W3, b3 = unflatten(theta, in_dim, h1, h2)

        # Ustal podział state/move do etykiet
        state_dim = args.state_dim
        move_dim  = args.move_dim
        if state_dim is None or move_dim is None:
            # Spróbuj użyć dimensów z feature_encodera, jeśli suma się zgadza
            if (FE_STATE_DIM + FE_MOVE_DIM) == in_dim:
                state_dim, move_dim = FE_STATE_DIM, FE_MOVE_DIM
            else:
                # nie znamy podziału – pokażemy tylko total i top-k
                state_dim, move_dim = None, None
                print(f"[viz] Nie znam poprawnego podziału state/move dla in_dim={in_dim}. Użyj --state-dim/--move-dim, jeśli chcesz etykiety.")

        title = f"Gen {g} | in={in_dim}, H1={h1}, H2={h2}"
        plot_heatmaps(W1, W2, W3, os.path.join(args.outdir, f"weights_gen_{g:04d}.png"), title)
        plot_importance(W1, os.path.join(args.outdir, f"importance_gen_{g:04d}.png"), title, state_dim, move_dim)
        print(f"[viz] zapisano: weights_gen_{g:04d}.png, importance_gen_{g:04d}.png")

if __name__ == "__main__":
    main()