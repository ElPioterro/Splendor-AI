# migrate_theta.py
# Użycie:
#   python migrate_theta.py --old nn_champion.npy --out nn_champion_migrated.npy
# Opcjonalnie: --old-h1 64 --old-h2 64 --new-h1 64 --new-h2 64
# Jeśli nie podasz H1/H2, użyje nn_config.py. in_dim(old) zostanie wywnioskowany z długości theta.
from __future__ import annotations
import argparse, sys
import numpy as np

import nn_config as cfg
from feature_encoder import STATE_DIM, MOVE_DIM

# Kształty parametrów: [W1 (in,h1), b1 (h1), W2 (h1,h2), b2(h2), W3(h2,1), b3(1)]
def param_shapes(in_dim: int, h1: int, h2: int):
    return [(in_dim, h1), (h1,), (h1, h2), (h2,), (h2, 1), (1,)]

def unflatten(theta: np.ndarray, shapes):
    params=[]; idx=0
    for s in shapes:
        n = int(np.prod(s))
        params.append(theta[idx: idx+n].reshape(s))
        idx += n
    if idx != theta.size:
        raise ValueError(f"theta size {theta.size} != expected {idx} for shapes {shapes}")
    return params

def flatten(params) -> np.ndarray:
    return np.concatenate([p.reshape(-1) for p in params]).astype(np.float32)

def infer_old_in_dim(theta_len: int, h1: int, h2: int) -> int:
    # len = in*h1 + h1*h2 + h1 + 2*h2 + 1  => in = (len - (h1*h2 + h1 + 2*h2 + 1)) / h1
    rest = theta_len - (h1*h2 + h1 + 2*h2 + 1)
    if rest <= 0 or rest % h1 != 0:
        raise ValueError("Nie można wywnioskować old_in_dim z podanych H1/H2 i długości theta.")
    return rest // h1

def main():
    ap = argparse.ArgumentParser(description="Warm-start migracja theta do większego in_dim (nowe cechy).")
    ap.add_argument("--old", type=str, default="nn_champion.npy", help="Ścieżka do starej thety")
    ap.add_argument("--out", type=str, default="nn_champion_migrated.npy", help="Ścieżka wyjściowa")
    ap.add_argument("--old-h1", type=int, default=None)
    ap.add_argument("--old-h2", type=int, default=None)
    ap.add_argument("--new-h1", type=int, default=None)
    ap.add_argument("--new-h2", type=int, default=None)
    args = ap.parse_args()

    theta_old = np.load(args.old).astype(np.float32).reshape(-1)

    # Ustal H1/H2
    old_h1 = args.old_h1 if args.old_h1 is not None else cfg.HIDDEN1
    old_h2 = args.old_h2 if args.old_h2 is not None else cfg.HIDDEN2
    new_h1 = args.new_h1 if args.new_h1 is not None else cfg.HIDDEN1
    new_h2 = args.new_h2 if args.new_h2 is not None else cfg.HIDDEN2

    if (old_h1 != new_h1) or (old_h2 != new_h2):
        print(f"[ERR] H1/H2 muszą być takie same do prostej migracji. old=({old_h1},{old_h2}) new=({new_h1},{new_h2})")
        sys.exit(1)

    new_in = int(STATE_DIM + MOVE_DIM)
    old_in = infer_old_in_dim(theta_len=theta_old.size, h1=old_h1, h2=old_h2)
    if new_in < old_in:
        print(f"[ERR] Nowe in_dim ({new_in}) < stare ({old_in}) – nieobsługiwane. Rozszerzamy, nie zwężamy.")
        sys.exit(1)

    # Rozbij starą thetę
    shapes_old = param_shapes(old_in, old_h1, old_h2)
    W1o, b1o, W2o, b2o, W3o, b3o = unflatten(theta_old, shapes_old)

    # Zbuduj nowy zestaw parametrów
    shapes_new = param_shapes(new_in, new_h1, new_h2)
    W1n = np.zeros(shapes_new[0], dtype=np.float32)  # (new_in, h1)
    b1n = b1o.copy()
    W2n = W2o.copy()
    b2n = b2o.copy()
    W3n = W3o.copy()
    b3n = b3o.copy()

    # Kopiujemy stare wiersze W1
    W1n[:old_in, :] = W1o
    # Nowe wiersze (dla nowych cech) zostają 0 – sieć zachowuje stare zachowanie i może stopniowo uczyć się z nowych sygnałów.

    theta_new = flatten([W1n, b1n, W2n, b2n, W3n, b3n])
    np.save(args.out, theta_new)
    print(f"[OK] Zapisano zmigrowaną thetę do: {args.out}")
    print(f"     old_in={old_in}, new_in={new_in}, H1/H2={new_h1}/{new_h2}, len_old={theta_old.size}, len_new={theta_new.size}")

if __name__ == "__main__":
    main()