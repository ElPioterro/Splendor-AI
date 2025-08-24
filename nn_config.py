# NN arch
STATE_DIM = 32   # zacznij prosto; zwiększymy po stabilizacji
MOVE_DIM  = 32
HIDDEN1   = 64
HIDDEN2   = 64
ACT       = "relu"  # "relu" | "silu" | "tanh"

# ES
ES_POP      = 64          # liczba perturbacji (liczymy pary mirrored => 2*ES_POP ewaluacji)
ES_SIGMA    = 0.1         # skala szumu
ES_LR       = 0.05        # learning rate
ES_ELITE_FRACTION = 0.2   # opcjonalne ważenie top-frakcji (rank-based)
ANNEAL_SIGMA = True

# Ewaluacja (możesz skopiować z config)
ANCHORS_COUNT = 4
EVAL_SEEDS_PER_ANCHOR = 2
WILSON_Z = 1.96
TURNS_BASELINE = 60.0
PPT_WEIGHT = 2.0  # lekki tie-break

# Losowość/MP
SEED_BASE = 12345
USE_MULTIPROCESSING = True
N_PROCESSES = None