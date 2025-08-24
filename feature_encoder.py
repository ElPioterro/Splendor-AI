import numpy as np
from typing import List, Tuple
from game_engine import GameState, Move, BuyCard, TakeThreeGems, TakeTwoGems, ReserveVisibleCard, ReserveFromDeck, GemColor
import config

COLORS = list(GemColor)

def _player_scalar_feats(state: GameState, pid: int) -> np.ndarray:
    p = state.players[pid]
    opps = [state.players[i] for i in range(len(state.players)) if i != pid]
    v = []
    # moje
    v.append(p.prestige_points)
    v += [p.bonuses[c] for c in COLORS]            # 5
    v += [p.gems[c] for c in COLORS]               # 5
    v.append(len(p.reserved_cards))                # 1
    v.append(p.gold_gems)                          # 1
    # przeciwnicy (agregaty)
    if opps:
        opp_max_pts = max(o.prestige_points for o in opps)
        opp_avg_bon = sum(sum(o.bonuses.values()) for o in opps)/len(opps)
        opp_avg_tok = sum(o.total_gems for o in opps)/len(opps)
    else:
        opp_max_pts = opp_avg_bon = opp_avg_tok = 0.0
    v += [opp_max_pts, opp_avg_bon, opp_avg_tok]   # 3
    # stół – dostępne żetony
    v += [state.available_gems[c] for c in COLORS] # 5
    # długość gry
    v.append(getattr(state, "turn_number", 0))
    arr = np.array(v, dtype=float)
    # prosta normalizacja
    arr[0] /= max(1, getattr(config, "TARGET_PRESTIGE_POINTS", 15))
    arr[1:6] = np.clip(arr[1:6]/6.0, 0, 1)
    arr[6:11] = np.clip(arr[6:11]/7.0, 0, 1)
    arr[11] = np.clip(arr[11]/3.0, 0, 1)
    arr[12] = np.clip(arr[12]/5.0, 0, 1)
    arr[13] = np.clip(arr[13]/15.0, 0, 1)
    arr[14] = np.clip(arr[14]/5.0, 0, 1)
    arr[15] = np.clip(arr[15]/10.0, 0, 1)
    arr[16:21] = np.clip(arr[16:21]/7.0, 0, 1)
    # turn_number
    base = float(getattr(config, "TURNS_BASELINE", 60.0))
    arr[21] = np.clip(arr[21]/base, 0, 1)
    return arr

def _move_feats(state: GameState, pid: int, move: Move) -> np.ndarray:
    # wektor ruchu: one-hot typu + proste delty (punkty, bonus, zyski żetonów)
    v = np.zeros(32, dtype=float)
    # typy: buy, res_v, res_d, take3, take2 => 5
    if isinstance(move, BuyCard):
        v[0] = 1.0
        v[5] = move.card.prestige_points / 5.0                         # zysk punktów
        # bonus kolor
        color_idx = COLORS.index(move.card.bonus_color)
        v[6+color_idx] = 1.0                                           # 5 kanałów
    elif isinstance(move, ReserveVisibleCard):
        v[1] = 1.0
        # karta potencjalna: prestiż i kosztowość (przybliżenie)
        v[5] = getattr(move.card, "prestige_points", 0)/5.0
        cidx = COLORS.index(move.card.bonus_color)
        v[6+cidx] = 0.3
    elif isinstance(move, ReserveFromDeck):
        v[2] = 1.0
        tier = int(getattr(move, "tier", 1))
        v[11] = np.clip((tier-1)/2.0, 0, 1)
    elif isinstance(move, TakeThreeGems):
        v[3] = 1.0
        for c in move.colors:
            cidx = COLORS.index(c)
            v[12 + cidx] += 1.0/3.0
    elif isinstance(move, TakeTwoGems):
        v[4] = 1.0
        cidx = COLORS.index(move.color)
        v[12 + cidx] += 1.0
    # prosta presja/tempo z kontekstu (kopie z state): sloty 20–21 wolne
    target = getattr(config, "TARGET_PRESTIGE_POINTS", 15)
    opps = [state.players[i] for i in range(len(state.players)) if i != pid]
    opp_max = max((o.prestige_points for o in opps), default=0)/max(1, target)
    v[20] = opp_max
    v[21] = np.clip(getattr(state, "turn_number", 0)/max(1.0, getattr(config, "TURNS_BASELINE", 60.0)), 0, 1)
    return v

def encode_state_and_moves(state: GameState, valid_moves: List[Move]) -> Tuple[np.ndarray, List[np.ndarray]]:
    pid = state.current_player_index
    s = _player_scalar_feats(state, pid)
    ms = [_move_feats(state, pid, m) for m in valid_moves]
    # dopasuj do NN dims
    s = _fit_dim(s, target=32)
    ms = [_fit_dim(mv, target=32) for mv in ms]
    return s, ms

def _fit_dim(arr: np.ndarray, target: int) -> np.ndarray:
    if arr.shape[0] == target:
        return arr
    if arr.shape[0] < target:
        return np.pad(arr, (0, target-arr.shape[0]))
    return arr[:target]