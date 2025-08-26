# PROMETHEUS — Odnoga „Neuroewolucja” (nn-agent)

Agent NN trenowany metodą OpenES (CPU, numpy), kompatybilny z głównym silnikiem (SSOT), deterministyczny i lekki. Decyzje powstają przez ocenę f([state_vec, move_vec]) małą siecią MLP i wybór argmax.

Kluczowe pryncypia:

- SSOT: tylko Game (game_engine.py) zarządza stanem/losowością i regułami.
- Czyste warstwy: enkoder nie ma stanu, agent nie zna silnika, trener nie ma ukrytych efektów.
- Determinizm: seedy przekazywane jawnie, brak globalnego random.seed.

## Struktura odnogi NN

- feature_encoder.py

  - encode_state_and_moves(state, valid_moves) -> (state_vec, move_vecs)
  - eksportuje: STATE_DIM, MOVE_DIM, COLOR_ORDER
  - szybki, czysty, bez deepcopy; normalizacje proste, delty aproksymowane lokalnie

- nn_agent.py

  - NnAgent: lekka MLP (numpy), ocena ruchów i wybór argmax
  - API zgodne z GA: choose_action -> (Move, Optional[Noble]) — noble wybiera engine
  - param_count(), init_theta() do wygody

- nn_config.py

  - Zero logiki, tylko parametry NN/ES i ewaluacji
  - importuje STATE_DIM/MOVE_DIM z feature_encoder (SSOT wymiarów)

- es_manager.py

  - Trener OpenES (mirrored sampling, common random numbers)
  - Ewaluacja vs kotwice (GeneticAgent), logi, zapis nn_champion.npy
  - Opcjonalny hold‑out (stabilny monitoring)

- train_nn.py

  - Driver: bootstrap środowiska i start treningu ES

- eval_nn_vs_ga.py (opcjonalny)

  - Headless ewaluacja NN vs GA (WR/WR_LB, tempo)

- main.py
  - Tryb gry vs NN: --play-nn (jeśli dodałeś patch)

## Wymagania

- Python 3.10/3.11
- numpy
- tqdm (opcjonalnie, do progresu)
- pygame (tylko do trybu gry z GUI)
- pytest (opcjonalnie, testy)

Instalacja:

```bash
pip install numpy tqdm pygame pytest
# lub: pip install -r requirements.txt
```

## Szybki start: trening

1. Kotwice (przeciwnicy do ewaluacji):

- Najprościej: miej champion_agent.npy w katalogu głównym.
- Albo przekaż własne pliki DNA GA przez --anchors.

2. Odpal trening:

```bash
python train_nn.py --gens 150
# lub z własnymi kotwicami:
python train_nn.py --gens 120 --anchors champion_agent.npy hof/top1.npy
```

Artefakty:

- Model: nn_champion.npy
- Log: logs/es_log.csv
- (opcjonalnie) Hold‑out: logs/es_holdout_log.csv

Wznowienie:

```bash
python train_nn.py --gens 200 --resume nn_champion.npy
```

## Tryb gry: Człowiek vs NN

Wymagane: patch w main.py dodający flagę --play-nn (dostarczony wcześniej).
Uruchom:

```bash
python main.py --play-nn
```

Uwaga: NnAgent nie wybiera arystokraty — po apply_move wybór dokonuje engine/GUI (deterministycznie: max prestiż, tie po id).

## Headless ewaluacja (NN vs GA)

Bez GUI, wiele seedów, obie role:

```bash
python eval_nn_vs_ga.py --seeds 64 --nn nn_champion.npy --ga champion_agent.npy
```

Wynik: WR, Wilson WR_LB, średnie punkty/tura i tempo (pts/turn).

## Konfiguracja (nn_config.py)

- Architektura NN:

  - HIDDEN1, HIDDEN2, ACT in {"relu","silu","tanh"}
  - IN_DIM = STATE_DIM + MOVE_DIM (SSOT z feature_encoder)

- OpenES:

  - ES_POP, ES_SIGMA, ES_LR
  - ANNEAL_SIGMA, SIGMA_DECAY, SIGMA_MIN

- Ewaluacja:

  - ANCHORS_COUNT, EVAL_SEEDS_PER_ANCHOR, WILSON_Z
  - PPT_WEIGHT (lekki tie-break tempa)

- Losowość/MP:

  - SEED_BASE, USE_MULTIPROCESSING, N_PROCESSES
  - ROLE_XOR_SEED do odwracania ról

- Hold‑out (opcjonalny monitoring):
  - USE_HOLDOUT, HOLDOUT_SEEDS_PER_ANCHOR, HOLDOUT_LOG_CSV

## Determinizm i seedy

- Game(seed=...) steruje losowością (lokalny RNG silnika).
- OpenES używa deterministycznej siatki seedów (gen/anchor/rep, obie role via XOR).
- NnAgent tie-break ma własny lokalny seed (rng_seed) — powtarzalne argmax przy remisach.
- Brak globalnego random.seed — izolacja od procesów pobocznych.

## Diagnostyka i logi

- logs/es_log.csv: per generacja — score, wr_lb, wr, sigma, time_sec
- logs/es_holdout_log.csv (opcjonalnie): stabilny WR/WR_LB/tempo na stałym zbiorze seedów
- nn_champion.npy: najlepsza znana theta (automatycznie nadpisywana przy poprawie)

Wskazówka: WR_LB jest konserwatywne — dla top modeli zwiększ EVAL_SEEDS_PER_ANCHOR lub uruchom headless eval z większą próbką.

## Częste problemy i rozwiązania

- Brak champion_agent.npy:

  - Przekaż kotwice przez --anchors do train_nn.py.
  - Albo umieść pliki \*.npy w anchors/ lub hof/ (jeśli używasz wersji trenera z auto‑odkrywaniem).

- Pylance: „Variable not allowed in type expression” w feature_encoder:

  - Zastosowana wersja używa TYPE_CHECKING i importu runtime `import game_engine as ge` — ostrzeżenie znika.

- Multiprocessing na Windows:

  - Uruchamiaj train_nn.py bezpośrednio (zawiera if **name** == "**main**":).
  - Przy debugowaniu wyłącz MP: `python train_nn.py --no-mp`.

- Niestabilne WR:

  - Włącz USE_HOLDOUT=True i obserwuj hold‑out WR_LB.
  - Podnieś EVAL_SEEDS_PER_ANCHOR, zwiększ ANCHORS_COUNT.

- Wydajność:
  - feature_encoder nie używa deepcopy; wszystkie wektory float32.
  - Dostosuj N_PROCESSES do liczby rdzeni.

## Testy (opcjonalne)

Szybkie sanity:

```python
# tests/test_nn_branch.py
import numpy as np, config
from game_loader import GameLoader
from game_engine import Game
from feature_encoder import encode_state_and_moves, STATE_DIM, MOVE_DIM

def test_encoder_shapes():
    cards, nobles = GameLoader().load_definitions_from_json(config.DATA_FILE)
    game = Game(all_cards=cards, all_nobles=nobles, seed=123).setup_new_game(["A","B"])
    s, mv = encode_state_and_moves(game, [])
    assert s.shape[0] == STATE_DIM
    # dla prawdziwej gry podaj valid_moves z game.get_valid_moves()
```

Uruchom:

```bash
pytest -q
```

## Roadmap (skrót)

- Feature engineering: noble_distance, block_value, scarcity_pressure+
- Trener: rank-based/CMA‑ES, lepsze CRN/antithetic
- Ewaluacja: hold‑out window, cross‑play vs poprzednich championów NN/GA
