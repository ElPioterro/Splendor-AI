# eval_nn_vs_ga.py
# Headless ewaluacja NN vs GA: WR i Wilson LB bez GUI.

from __future__ import annotations
import argparse
import math
import numpy as np

import config
from game_loader import GameLoader
import game_engine as ge
from agents import GeneticAgent
from nn_agent import NnAgent
import nn_config as cfg
from feature_encoder import STATE_DIM, MOVE_DIM
import random
random.seed(cfg.SEED_BASE)
np.random.seed(cfg.SEED_BASE & 0xFFFFFFFF)

COLOR_ORDER = [ge.GemColor.WHITE, ge.GemColor.BLUE, ge.GemColor.GREEN, ge.GemColor.RED, ge.GemColor.BLACK]
COLOR_TO_IDX = {c: i for i, c in enumerate(COLOR_ORDER)}

def _move_sort_key(m: ge.Move):
    if isinstance(m, ge.BuyCard):
        return (0, m.card.level, m.card.id)
    if isinstance(m, ge.ReserveVisibleCard):
        return (1, m.card.level, m.card.id)
    if isinstance(m, ge.ReserveFromDeck):
        return (2, m.tier, 0)
    if isinstance(m, ge.TakeThreeGems):
        cols = tuple(sorted((COLOR_TO_IDX[c] for c in m.colors)))
        return (3, cols)
    if isinstance(m, ge.TakeTwoGems):
        return (4, COLOR_TO_IDX[m.color])
    return (99, repr(m))

def canonicalize_moves(moves: list[ge.Move]) -> list[ge.Move]:
    return sorted(moves, key=_move_sort_key)

def infer_hidden_dims(theta_len: int, in_dim: int) -> tuple[int,int] | None:
    for h1 in (32, 64, 128):
        for h2 in (32, 64, 128):
            p = in_dim*h1 + h1 + h1*h2 + h2 + h2 + 1
            if p == theta_len:
                return h1, h2
    return None

def wilson_lower_bound(p: float, n: int, z: float = cfg.WILSON_Z) -> float:
    if n <= 0:
        return 0.0
    p = float(max(0.0, min(1.0, p)))
    denom = 1.0 + (z*z) / n
    center = p + (z*z) / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) + (z*z)/(4.0*n)) / n)
    return (center - margin) / denom

def choose_noble_deterministic(eligible: list[ge.Noble]):
    if not eligible:
        return None
    return sorted(eligible, key=lambda n: (-n.prestige_points, n.id))[0]

def play_game(theta: np.ndarray, ga_dna: np.ndarray, as_first: bool, game_seed: int, tie_seed: int,
              cards, nobles) -> tuple[int, int, int]:
    # Per‑game determinism (gdyby GA używał global RNG)
    random.seed(tie_seed)
    np.random.seed(tie_seed & 0xFFFFFFFF)

    # Auto‑dopasowanie architektury (zostaw jak masz)
    try:
        nn_agent = NnAgent(theta=theta, hidden1=cfg.HIDDEN1, hidden2=cfg.HIDDEN2, act=cfg.ACT, rng_seed=tie_seed)
    except ValueError:
        inferred = infer_hidden_dims(len(theta), STATE_DIM + MOVE_DIM)
        if not inferred:
            raise
        h1, h2 = inferred
        print(f"[eval] Auto-detected NN dims: H1={h1}, H2={h2}")
        nn_agent = NnAgent(theta=theta, hidden1=h1, hidden2=h2, act=cfg.ACT, rng_seed=tie_seed)

    ga_agent = GeneticAgent(dna=ga_dna.copy())

    game = ge.Game(all_cards=cards, all_nobles=nobles, seed=game_seed)
    game.setup_new_game(player_names=["P1", "P2"])

    def is_nn_turn(state: ge.GameState) -> bool:
        return (state.current_player_index == 0) if as_first else (state.current_player_index == 1)

    # BEZPIECZNIK: limit tur (jak w trenerze)
    max_turns = int(getattr(config, "MAX_TURNS", 150))
    turns = 0

    while not game.is_game_over():
        state = game.game_state
        valid = canonicalize_moves(game.get_valid_moves())
        if not valid:
            game.finalize_turn(None)
            turns += 1
            if turns >= max_turns:
                break
            continue

        if is_nn_turn(state):
            move, _ = nn_agent.choose_action(state, valid)
        else:
            move, _ = ga_agent.choose_action(state, valid)

        eligible = game.apply_move(move)
        chosen = choose_noble_deterministic(eligible)
        game.finalize_turn(chosen)
        turns += 1
        if turns >= max_turns:
            break

    # Zakończenie: normalnie lub po limicie tur
    if game.is_game_over():
        winner = game.get_winner()
        p_nn = game.game_state.players[0 if as_first else 1]
        my_win = 1 if winner is p_nn else 0
        my_pts = int(p_nn.prestige_points)
        turns = int(game.game_state.turn_number)
        return my_win, my_pts, turns
    else:
        # Finisz po limicie tur – rozstrzygnięcie jak w engine (prestige, tie-break len(cards))
        players = game.game_state.players
        winner_like_engine = sorted(players, key=lambda p: (-p.prestige_points, len(p.cards)))[0]
        p_nn = players[0 if as_first else 1]
        my_win = 1 if winner_like_engine is p_nn else 0
        my_pts = int(p_nn.prestige_points)
        return my_win, my_pts, turns
    
def main():
    ap = argparse.ArgumentParser(description="Headless eval NN vs GA")
    ap.add_argument("--seeds", type=int, default=32, help="Ile seedów (każdy seed = 2 gry, obie role)")
    ap.add_argument("--nn", type=str, default="nn_champion.npy", help="Plik z parametrami NN (theta)")
    ap.add_argument("--ga", type=str, default="champion_agent.npy", help="Plik DNA GA (anchor)")
    args = ap.parse_args()

    loader = GameLoader()
    cards, nobles = loader.load_definitions_from_json(config.DATA_FILE)

    theta = np.load(args.nn).astype(np.float32)
    try:
        ga_dna = np.load(args.ga).astype(np.float32)
    except FileNotFoundError:
        try:
            from config import BASELINE_DNA_POOL, DNA_SIZE
            if BASELINE_DNA_POOL:
                ga_dna = np.array(BASELINE_DNA_POOL[0], dtype=np.float32)
                print("[eval] Używam BASELINE_DNA_POOL[0] jako GA kotwicy (brak champion_agent.npy).")
            else:
                ga_dna = np.zeros(DNA_SIZE, dtype=np.float32)
                print("[eval] Używam zerowego DNA (fallback), ustaw BASELINE_DNA_POOL albo podaj --ga.")
        except Exception:
            # ostateczny fallback, jeśli config nie ma DNA_SIZE
            DNA_SIZE = 20
            ga_dna = np.zeros(DNA_SIZE, dtype=np.float32)
            print("[eval] Fallback: zerowe DNA 20-dim (ustaw poprawne parametry w config.py).")

    # Dopasuj rozmiar DNA do oczekiwanego przez GeneticAgent
    try:
        from config import DNA_SIZE
        expected = int(DNA_SIZE)
    except Exception:
        expected = 20

    ga_dna = np.asarray(ga_dna, dtype=np.float32).reshape(-1)
    if ga_dna.size > expected:
        print(f"[eval] Ostrzeżenie: DNA GA ({ga_dna.size}) > {expected}. Przycinam do {expected}.")
        ga_dna = ga_dna[:expected]
    elif ga_dna.size < expected:
        print(f"[eval] Ostrzeżenie: DNA GA ({ga_dna.size}) < {expected}. Dopadam zerami do {expected}.")
        ga_dna = np.pad(ga_dna, (0, expected - ga_dna.size), mode="constant", constant_values=0.0)

    wins = 0
    games = 0
    sum_pts = 0
    sum_turns = 0

    for i in range(args.seeds):
        seed_first = int(cfg.SEED_BASE + 10007 * i)
        seed_second = seed_first ^ int(cfg.ROLE_XOR_SEED)
        tie_seed = int(cfg.SEED_BASE ^ 0x9E3779B1 ^ i)

        w, pts, t = play_game(theta, ga_dna, as_first=True, game_seed=seed_first, tie_seed=tie_seed,
                              cards=cards, nobles=nobles)
        wins += w; games += 1; sum_pts += pts; sum_turns += t

        w, pts, t = play_game(theta, ga_dna, as_first=False, game_seed=seed_second, tie_seed=tie_seed,
                              cards=cards, nobles=nobles)
        wins += w; games += 1; sum_pts += pts; sum_turns += t
        print(f"[eval] seed {i+1}/{args.seeds} done")

    wr = wins / games if games else 0.0
    wr_lb = wilson_lower_bound(wr, games, cfg.WILSON_Z) if games else 0.0
    avg_pts = sum_pts / games if games else 0.0
    avg_turns = sum_turns / games if games else 0.0
    pace = avg_pts / max(1.0, avg_turns)

    print("="*60)
    print(f"NN vs GA | seeds={args.seeds} (gry={games})")
    print(f"WR      : {wr:.4f}")
    print(f"WR_LB   : {wr_lb:.4f}")
    print(f"Avg pts : {avg_pts:.3f}")
    print(f"Avg turn: {avg_turns:.3f}")
    print(f"Pace    : {pace:.3f} (pts/turn)")
    print("="*60)

if __name__ == "__main__":
    main()