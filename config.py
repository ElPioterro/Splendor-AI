# config.py

# --- Konfiguracja Gry ---
DATA_FILE = 'data/splendor_data.json'
TARGET_PRESTIGE_POINTS = 15

# --- Konfiguracja Agenta Genetycznego i Konstytucja ---
DNA_SIZE = 20

############################################################################
#       KONFIGURACJA PROCESU EWOLUCYJNEGO - ARCHITEKTURA "GENIUSZ"       #
############################################################################

# --- GA: Populacja i Generacje ---
POP_SIZE = 60
GENERATIONS = 120

# --- GA: Selekcja i Reprodukcja ---
ELITE_COUNT = 6             # Ilu najlepszych agentów przechodzi do następnej generacji
IMMIGRANTS = 3              # Ilu losowych "imigrantów" dodać w każdej generacji dla różnorodności
TOURNAMENT_K = 5            # Rozmiar turnieju selekcyjnego
CROSSOVER_RATE = 0.9        # Prawdopodobieństwo krzyżowania

# --- GA: Mutacja (z mechanizmem Annealing) ---
MUTATION_RATE = 0.15        # Prawdopodobieństwo mutacji pojedynczego genu
MUTATION_SIGMA_START = 0.35 # Początkowa "siła" mutacji (odch. standardowe)
MUTATION_SIGMA_END = 0.10   # Końcowa "siła" mutacji
GENE_CLIP = 1.5             # Geny będą przycinane do zakresu [-1.5, 1.5]

# --- GA: Ewaluacja (Kotwice i Powtarzalność) ---
ANCHORS_COUNT = 4           # Ilu przeciwników-kotwic użyć w każdej generacji
HOF_MAX = 5                 # Maksymalny rozmiar Hall of Fame (przechowuje najlepsze historyczne DNA)
EVAL_SEEDS_PER_ANCHOR = 2   # Na każdą parę (kandydat, kotwica) zagramy N seedów x 2 role (startujący/drugi)
                            # Całkowita liczba gier na kandydata = ANCHORS_COUNT * EVAL_SEEDS_PER_ANCHOR * 2

# --- GA: Baseline DNA (optional starting anchors) ---
# Collection of DNA for agents with known, interesting strategies
BASELINE_DNA_POOL = [
    # Example 1: Aggressive point-chaser
    [1.0, 0.8, -0.2, 0.6, 0.4, 0.2, -0.1, 0.3, 0.9, 0.7, -0.3, 0.5, 0.2, -0.2, 0.1, 0.4],
    # Example 2: Engine-building card strategist
    [0.2, -0.6, 1.0, 0.7, -0.3, 0.4, 0.8, -0.5, 0.6, -0.1, 0.9, 0.3, -0.2, 0.5, 0.1, 0.7],
    # Example 3: Defensive resource hoarder
    [0.3, 0.1, -0.8, 0.4, 0.9, -0.2, 0.6, -0.4, 0.2, 0.5, -0.7, 0.3, 0.8, -0.1, 0.4, -0.3],
    # Example 4: Balanced opportunist
    [0.5, 0.4, 0.3, -0.2, 0.1, 0.6, -0.3, 0.5, -0.1, 0.4, 0.2, -0.5, 0.3, 0.7, -0.4, 0.2],
    # Example 5: High-risk combo seeker
    [-0.1, 0.9, 0.7, -0.3, 0.2, 0.8, -0.6, 0.4, 0.1, -0.2, 0.9, 0.5, -0.7, 0.3, 0.6, -0.5],
    # Example 6: Late-game scaler
    [0.4, -0.3, 0.6, 0.2, -0.5, 0.7, 0.1, -0.2, 0.8, 0.3, -0.4, 0.9, 0.2, -0.6, 0.5, 0.1],
]

# --- Ustawienia Techniczne ---
SEED_BASE = 12345
USE_MULTIPROCESSING = False   # Ustaw na True, aby potencjalnie przyspieszyć ewaluację (wymaga implementacji Pool.map)
N_PROCESSES = None            # None => cpu_count()-1, jeśli USE_MULTIPROCESSING = True

# --- Multiprocessing ---
USE_MULTIPROCESSING = False   # włącz, gdy będziesz chciał przyspieszyć ewaluację
N_PROCESSES = None            # None => cpu_count() - 1
CHUNK_SIZE = None             # None lub np. 4-16 dla lepszego throughput

# --- Re-score top-N ---
USE_TOP_RESCORE = True
TOP_RESCORE_N = 10
EVAL_SEEDS_PER_ANCHOR_TOP = 3  # dokładniejsza ewaluacja topki

# --- Seedy i RNG dla ról/agentów ---
DIFFERENT_SEED_FOR_SWAP = True
SWAP_ROLES_SEED_XOR = 0x9E3779B
AGENT_SEED_CAND_OFFSET = 101
AGENT_SEED_ANCH_OFFSET = 313

SHOW_PROGRESS = True  # włącz/wyłącz paski postępu

# --- Szybkość i długość gier (wpływ na fitness) ---
TURNS_BASELINE = 58.0           # realistyczna mediana długości gry
TURNS_PENALTY_WEIGHT = 2        # kara za bycie wolniejszym niż baseline
SPEED_BONUS_WEIGHT = 1.2        # bonus za bycie szybszym (skaluje się z points_norm)
PPT_WEIGHT = 2.0      # bonus za punkty/na turę

# --- Limit tur bezpieczeństwa ---
MAX_TURNS = 100#150                 # twardy limit tur w symulacji (seq + MP)

# --- Harmonogram trudności (anchors) ---
ANCHOR_SCHEDULE_ENABLED = True
ANCHOR_SCHEDULE_UP_THRESHOLD = 0.65
ANCHOR_SCHEDULE_DOWN_THRESHOLD = 0.50
ANCHOR_SCHEDULE_STREAK = 3
ANCHORS_MAX = 5                 # docelowa liczba kotwic po „utwardzeniu”

# --- Klips wag (delikatne otwarcie sufitu) ---
GENE_CLIP = 2.0                 # było 1.5
# (opcjonalnie) delikatniejsze końcowe mutacje:
# MUTATION_SIGMA_END = 0.09

WILSON_Z = 1.96  # 95% CI dla Wilson lower bound