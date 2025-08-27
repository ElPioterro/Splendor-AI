# nn_config.py
# Konfiguracja odnogi NN/ES (bez logiki). SSOT: STATE_DIM/MOVE_DIM z feature_encoder.

from __future__ import annotations
from feature_encoder import STATE_DIM, MOVE_DIM

# Architektura modelu NN
HIDDEN1: int = 64
HIDDEN2: int = 64
# Dozwolone: "relu", "silu", "tanh"
ACT: str = "silu"

# Łączny wymiar wejścia (state + move)
IN_DIM: int = STATE_DIM + MOVE_DIM

# Hyperparametry OpenES
ES_POP: int = 64
ES_SIGMA: float = 0.10
ES_LR: float = 0.05

# Annealing dla sigma (proste, w trenerze)
ANNEAL_SIGMA: bool = True
SIGMA_DECAY: float = 0.995
SIGMA_MIN: float = 0.03

# Opcjonalne (na przyszłość)
ES_ELITE_FRACTION: float = 0.0  # 0.0 = nieużywane

# Ewaluacja i metryki
ANCHORS_COUNT: int = 4
EVAL_SEEDS_PER_ANCHOR: int = 6
PPT_WEIGHT: float = 0.05
WILSON_Z: float = 1.96

# Losowość / MP (drukujemy w driverze, ES może nie używać MP)
SEED_BASE: int = 424242
USE_MULTIPROCESSING: bool = True
N_PROCESSES: int = 8

# XOR do odwracania ról (używany np. w eval_nn_vs_ga)
ROLE_XOR_SEED: int = 0xA5A5A5A5

# Logi i artefakty
LOG_DIR: str = "logs"
ES_LOG_CSV: str = "logs/es_log.csv"
MODEL_OUT_FILE: str = "nn_champion.npy"

# Opcjonalny hold-out (stabilny monitoring)
USE_HOLDOUT: bool = True
HOLDOUT_SEEDS_PER_ANCHOR: int = 32
HOLDOUT_LOG_CSV: str = "logs/es_holdout_log.csv"

SNAPSHOT_EVERY: int = 10   # co ile generacji zapisać theta (0 = wyłącz)
SNAPSHOT_DIR: str = "snapshots"