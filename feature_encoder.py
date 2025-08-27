# feature_encoder.py
# Czysty, szybki enkoder cech dla odnogi NN (PROMETHEUS / nn-agent).
# Kontrakt publiczny: encode_state_and_moves(state, valid_moves) -> (state_vec, [move_vecs])

from __future__ import annotations

from typing import List, Tuple, Optional, TYPE_CHECKING
import numpy as np
import game_engine as ge  # runtime: isinstance, dostęp do klas

if TYPE_CHECKING:
    from game_engine import (
        GameState, Player, Card, Noble, Move,
        TakeThreeGems, TakeTwoGems, ReserveVisibleCard, ReserveFromDeck, BuyCard,
        GemColor,
    )

# =========================
# Stałe i kolejność kolorów
# =========================

# Kolejność spójna z ge.GemColor
COLOR_ORDER = (
    ge.GemColor.WHITE, ge.GemColor.BLUE, ge.GemColor.GREEN, ge.GemColor.RED, ge.GemColor.BLACK
)
COLOR_TO_IDX = {c: i for i, c in enumerate(COLOR_ORDER)}
N_COLORS = len(COLOR_ORDER)

# Normalizacje (stałe, proste)
MAX_POINTS = 15.0            # próg zwycięstwa w engine
MAX_CARD_POINTS = 5.0        # typowy max punktów na karcie
MAX_BONUS_PER_COLOR = 8.0    # prosta skala (wystarczająca do stabilizacji)
MAX_TOKENS_IN_HAND = 10.0    # limit żetonów u gracza
MAX_BANK_COLOR = 7.0         # bank przy 4 graczach
MAX_BANK_GOLD = 5.0          # startowe złoto
TURNS_BASELINE = 40.0        # normalizacja numeru tury
SHORTFALL_NORM = 7.0         # skala dla "bliskości" zakupu

MAX_VISIBLE_PER_COLOR = 12.0   # 3 rzędy * 4 sloty

def _clip01(x: float) -> float:
    return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else x)

def _norm(x: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return _clip01(float(x) / float(denom))

def _to_f32(seq) -> np.ndarray:
    return np.asarray(seq, dtype=np.float32)

def _vec_from_color_dict(d: dict) -> np.ndarray:
    # d: Dict[ge.GemColor, int]
    out = np.zeros(N_COLORS, dtype=np.float32)
    if d:
        for c, v in d.items():
            idx = COLOR_TO_IDX.get(c, None)
            if idx is not None:
                out[idx] = float(v)
    return out

# --- NEW: widoczne karty (po bonus_color) ---
def _visible_bonus_counts(state: "GameState") -> np.ndarray:
    cnt = np.zeros(N_COLORS, dtype=np.float32)
    for row in state.visible_cards.values():
        for card in row:
            idx = COLOR_TO_IDX[card.bonus_color]
            cnt[idx] += 1.0
    return cnt

# --- NEW: kolory „potrzebne” do najbliższego nobles (dla synergy w move) ---
def _needed_colors_for_closest_noble(me: "Player", nobles: List["Noble"]) -> Optional[set]:
    if not nobles:
        return None
    my_bon = _player_bonuses_vec(me)
    best_noble = None
    best_tot = None
    for n in nobles:
        tot = 0.0
        for c, r in n.requirements.items():
            idx = COLOR_TO_IDX[c]
            need = float(r) - my_bon[idx]
            if need > 0:
                tot += need
        if best_tot is None or tot < best_tot:
            best_tot = tot; best_noble = n
    if best_noble is None:
        return None
    needed = set()
    for c, r in best_noble.requirements.items():
        idx = COLOR_TO_IDX[c]
        if my_bon[idx] < float(r):
            needed.add(idx)
    return needed if needed else None

def _nobles_req_sum(state: "GameState") -> np.ndarray:
    req = np.zeros(N_COLORS, dtype=np.float32)
    for n in state.nobles:
        for c, r in n.requirements.items():
            req[COLOR_TO_IDX[c]] += float(r)
    return req  # surowa suma per kolor

def _nobles_req_share(state: "GameState") -> np.ndarray:
    vec = _nobles_req_sum(state)
    tot = float(np.sum(vec))
    return vec / tot if tot > 0 else vec  # rozkład (suma=1) albo zera

def _noble_min_distance_total(me: "Player", nobles: list["Noble"]) -> Tuple[float, float]:
    """
    Zwraca: (best_dist, denom), gdzie:
      - best_dist: minimalny łączny brak bonusów do najbliższego nobla (po odjęciu naszych bonusów),
      - denom: skala normalizacji = maksymalna suma wymagań dowolnego nobla dostępnego na stole.
    Gdy nie ma nobles, zwraca (0.0, 1.0).
    """
    if not nobles:
        return 0.0, 1.0  # ważne: zawsze krotka!

    my_bon = _player_bonuses_vec(me)
    best_dist = None
    max_req_per_noble = 1.0
    for n in nobles:
        tot_req = float(sum(n.requirements.values()))
        if tot_req > max_req_per_noble:
            max_req_per_noble = tot_req
        dist = 0.0
        for c, r in n.requirements.items():
            need = float(r) - my_bon[COLOR_TO_IDX[c]]
            if need > 0:
                dist += need
        best_dist = dist if (best_dist is None or dist < best_dist) else best_dist

    return float(best_dist or 0.0), float(max_req_per_noble)

# =========================
# Dostępy do pól stanu (bezpieczne i szybkie)
# =========================

def _current_player(state: GameState) -> Player:
    return state.get_current_player()

def _opponents(state: GameState) -> List[Player]:
    me_idx = state.current_player_index
    return [p for i, p in enumerate(state.players) if i != me_idx]

def _turn_number(state: GameState) -> int:
    return int(state.turn_number)

def _player_prestige(p: Player) -> int:
    return int(p.prestige_points)

def _player_bonuses_vec(p: Player) -> np.ndarray:
    return _vec_from_color_dict(p.bonuses)

def _player_tokens(p: Player) -> Tuple[np.ndarray, int]:
    return _vec_from_color_dict(p.gems), int(p.gold_gems)

def _player_reserved_count(p: Player) -> int:
    return len(p.reserved_cards)

def _bank_tokens(state: GameState) -> Tuple[np.ndarray, int]:
    return _vec_from_color_dict(state.available_gems), int(state.gold_gems)

def _card_cost_vec(card: Card) -> np.ndarray:
    out = np.zeros(N_COLORS, dtype=np.float32)
    for color, amount in card.cost:
        idx = COLOR_TO_IDX[color]
        out[idx] = float(amount)
    return out

def _card_points(card: Card) -> int:
    return int(card.prestige_points)

def _card_bonus_idx(card: Card) -> int:
    return COLOR_TO_IDX[card.bonus_color]

# =========================
# Shortfall i wydatki przy kupnie
# =========================

def _cost_after_bonus(cost: np.ndarray, bon: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, cost - bon)

def _shortfall_total(cost: np.ndarray, bon: np.ndarray, tok_c: np.ndarray, tok_g: int) -> int:
    need = _cost_after_bonus(cost, bon).astype(np.int32)
    pay_c = np.minimum(need, tok_c.astype(np.int32))
    remain = need - pay_c
    lack = int(np.sum(remain))
    use_gold = min(tok_g, lack)
    return max(0, lack - use_gold)

def _tokens_spent_for_buy(cost: np.ndarray, bon: np.ndarray, tok_c: np.ndarray, tok_g: int) -> Tuple[np.ndarray, int]:
    need = _cost_after_bonus(cost, bon).astype(np.int32)
    pay_c = np.minimum(need, tok_c.astype(np.int32))
    remain = need - pay_c
    lack = int(np.sum(remain))
    pay_gold = min(tok_g, lack)
    return pay_c.astype(np.float32), int(pay_gold)

def _opponents_tokens_gold_reserved(state: "GameState") -> tuple[np.ndarray, float, float]:
    opps = _opponents(state)  # poprawna nazwa helpera
    if not opps:
        return np.zeros(N_COLORS, dtype=np.float32), 0.0, 0.0
    tok_vecs, golds, resv = [], [], []
    for p in opps:
        c, g = _player_tokens(p)
        tok_vecs.append(c)
        golds.append(float(g))
        resv.append(float(_player_reserved_count(p)))  # używamy istniejącego helpera
    tok_max = np.max(np.stack(tok_vecs, axis=0), axis=0)  # [5]
    gold_max = float(np.max(golds))
    res_max = float(np.max(resv))
    return tok_max, gold_max, res_max

# =========================
# Cechy stanu
# =========================

def _encode_state(state: GameState) -> np.ndarray:
    me = _current_player(state)
    opps = _opponents(state)
    bank_c, bank_g = _bank_tokens(state)

    my_pts = _player_prestige(me)
    my_bon = _player_bonuses_vec(me)
    my_tok_c, my_tok_g = _player_tokens(me)
    my_res = _player_reserved_count(me)

 

    if opps:
        opp_pts_arr = np.asarray([_player_prestige(p) for p in opps], dtype=np.float32)
        opp_pts_max = float(np.max(opp_pts_arr))
        opp_pts_avg = float(np.mean(opp_pts_arr))
        opp_bon_mat = np.stack([_player_bonuses_vec(p) for p in opps], axis=0)
        opp_bon_avg = np.mean(opp_bon_mat, axis=0)
        opp_bon_max = np.max(opp_bon_mat, axis=0)
    else:
        opp_pts_max = 0.0
        opp_pts_avg = 0.0
        opp_bon_avg = np.zeros(N_COLORS, dtype=np.float32)
        opp_bon_max = np.zeros(N_COLORS, dtype=np.float32)

    # --- NEW: widoczne karty i nobles ---
    vis_bonus_cnt = _visible_bonus_counts(state)          # [5], normalizujemy /12 (max 12 kart na stole)
    nobles_share  = _nobles_req_share(state)              # [5], rozkład po kolorach (suma=1)
    noble_dist_raw, noble_dist_denom = _noble_min_distance_total(me, state.nobles)  # scalar, denom dynamiczny

    # --- NEW: relacja do lidera ---
    rel_points = 0.5 + (my_pts - opp_pts_max) / (2.0 * MAX_POINTS)
    rel_points = _clip01(rel_points)

    opp_tok_c_max, opp_tok_g_max, opp_res_max = _opponents_tokens_gold_reserved(state)

    tnum = _turn_number(state)

    feats: List[float] = []
    # 1) Ja
    feats.append(_norm(my_pts, MAX_POINTS))
    feats.extend((my_bon / MAX_BONUS_PER_COLOR).tolist())
    feats.extend((my_tok_c / MAX_TOKENS_IN_HAND).tolist())
    feats.append(_norm(my_tok_g, MAX_BANK_GOLD))
    feats.append(_norm(my_res, 3.0))

    # 2) Oponenci (zostawiamy – sieć może się nauczyć ignorować, ale rel_points daje lepszy sygnał)
    feats.append(_norm(opp_pts_max, MAX_POINTS))
    feats.append(_norm(opp_pts_avg, MAX_POINTS))
    feats.extend((opp_bon_avg / MAX_BONUS_PER_COLOR).tolist())
    feats.extend((opp_bon_max / MAX_BONUS_PER_COLOR).tolist())

    # 3) Bank
    feats.extend((bank_c / MAX_BANK_COLOR).tolist())
    feats.append(_norm(bank_g, MAX_BANK_GOLD))

    # 4) NEW: rynek kart i nobles
    feats.extend((vis_bonus_cnt / MAX_VISIBLE_PER_COLOR).tolist())   # [5]
    feats.extend(nobles_share.tolist())                              # [5] – już w [0,1], suma=1
    feats.append(_clip01(noble_dist_raw / max(1.0, noble_dist_denom)))  # [1] dystans / max wymagań noble

    feats.append(_clip01(rel_points))                                 # [1]

    # ... po sekcji Oponenci (punkty/bonusy) i Bank – wstaw:
    feats.extend((opp_tok_c_max / MAX_TOKENS_IN_HAND).tolist())  # [5] ile przeciwnik MA żetonów (kolor)
    feats.append(_norm(opp_tok_g_max, MAX_BANK_GOLD))            # [1] ile przeciwnik MA złota
    feats.append(_norm(opp_res_max, 3.0))                        # [1] ile przeciwnik MA rezerw

    # 5) Tura
    feats.append(_norm(tnum, TURNS_BASELINE))

    return _to_f32(feats)

# =========================
# Cechy ruchu
# =========================

class MoveCat:
    BUY = 0
    RESERVE = 1
    TAKE3 = 2
    TAKE2 = 3

def _infer_move_category(m: Move) -> int:
    if isinstance(m, ge.BuyCard):
        return MoveCat.BUY
    if isinstance(m, (ge.ReserveVisibleCard, ge.ReserveFromDeck)):
        return MoveCat.RESERVE
    if isinstance(m, ge.TakeThreeGems):
        return MoveCat.TAKE3
    if isinstance(m, ge.TakeTwoGems):
        return MoveCat.TAKE2
    return MoveCat.RESERVE

def _move_card(m: Move) -> Optional[Card]:
    if isinstance(m, (ge.BuyCard, ge.ReserveVisibleCard)):
        return m.card
    return None

def _best_target_shortfall_from_valid_moves(valid_moves: List[Move], me: Player) -> Optional[int]:
    tok_c, tok_g = _player_tokens(me)
    bon = _player_bonuses_vec(me)
    best: Optional[int] = None
    for mv in valid_moves:
        if isinstance(mv, ge.BuyCard):
            cost = _card_cost_vec(mv.card)
        elif isinstance(mv, ge.ReserveVisibleCard):
            cost = _card_cost_vec(mv.card)
        else:
            continue
        sf = _shortfall_total(cost, bon, tok_c, tok_g)
        best = sf if (best is None or sf < best) else best
    return best

def _opponent_pressure_for_card(card: Optional[Card], opps: List[Player]) -> float:
    if not card or not opps:
        return 0.0
    cost = _card_cost_vec(card)
    best_closeness = 0.0
    for p in opps:
        bon = _player_bonuses_vec(p)
        tok_c, tok_g = _player_tokens(p)
        sf = _shortfall_total(cost, bon, tok_c, tok_g)
        closeness = 1.0 - _norm(sf, SHORTFALL_NORM)
        if closeness > best_closeness:
            best_closeness = closeness
    return _clip01(best_closeness)

def _scarcity_pressure(colors: List[int], bank_c: np.ndarray) -> float:
    if not colors:
        return 0.0
    vals = []
    for i in colors:
        scarcity = 1.0 - _norm(float(bank_c[i]), MAX_BANK_COLOR)
        vals.append(_clip01(scarcity))
    return float(np.mean(vals)) if vals else 0.0

def _turn_progress_take(delta_tok_c: np.ndarray, delta_tok_g: int, me: Player, valid_moves: List[Move]) -> float:
    best_sf = _best_target_shortfall_from_valid_moves(valid_moves, me)
    if best_sf is None or best_sf <= 0:
        return 0.0
    gained = float(np.sum(delta_tok_c) + delta_tok_g)
    return _clip01(gained / float(best_sf + 1e-6))

def _encode_move(move: Move, state: GameState, valid_moves: List[Move]) -> np.ndarray:
    me = _current_player(state)
    opps = _opponents(state)
    bank_c, bank_g = _bank_tokens(state)

    needed = _needed_colors_for_closest_noble(me, state.nobles)  # --- NEW
    noble_synergy = 0.0   

    cat = _infer_move_category(move)
    type_one_hot = [0.0, 0.0, 0.0, 0.0]
    type_one_hot[cat] = 1.0

    delta_points = 0.0
    bonus_gain_oh = np.zeros(N_COLORS, dtype=np.float32)
    delta_tok_c = np.zeros(N_COLORS, dtype=np.float32)
    delta_tok_g = 0
    opp_pressure = 0.0
    turn_progress = 0.0

    if isinstance(move, ge.BuyCard):
        pts = _card_points(move.card)
        delta_points = _norm(pts, MAX_CARD_POINTS)
        bonus_gain_oh[_card_bonus_idx(move.card)] = 1.0

        bon = _player_bonuses_vec(me)
        tok_c, tok_g = _player_tokens(me)
        cost = _card_cost_vec(move.card)
        pay_c, pay_g = _tokens_spent_for_buy(cost, bon, tok_c, tok_g)
        delta_tok_c = -pay_c
        delta_tok_g = -pay_g

        opp_pressure = _opponent_pressure_for_card(move.card, opps)
        turn_progress = 1.0
        # --- NEW: synergy ---
        if needed is not None:
            bidx = _card_bonus_idx(move.card)
            noble_synergy = 1.0 if bidx in needed else 0.0

    elif isinstance(move, (ge.ReserveVisibleCard, ge.ReserveFromDeck)):
        if bank_g > 0:
            delta_tok_g = 1
        card = _move_card(move)
        opp_pressure = _opponent_pressure_for_card(card, opps) if isinstance(move, ge.ReserveVisibleCard) else 0.0
        turn_progress = 0.0
        if isinstance(move, ge.ReserveVisibleCard) and needed is not None:
            bidx = _card_bonus_idx(move.card)
            noble_synergy = 1.0 if bidx in needed else 0.0

    elif isinstance(move, ge.TakeThreeGems):
        cols_idx = [COLOR_TO_IDX[c] for c in move.colors]
        for i in cols_idx:
            delta_tok_c[i] += 1.0
        opp_pressure = _scarcity_pressure(cols_idx, bank_c)
        turn_progress = _turn_progress_take(delta_tok_c, 0, me, valid_moves)
        if needed:
            cols_idx = [COLOR_TO_IDX[c] for c in move.colors]
            hits = [1.0 if i in needed else 0.0 for i in cols_idx]
            noble_synergy = float(np.mean(hits)) if hits else 0.0

    elif isinstance(move, ge.TakeTwoGems):
        cidx = COLOR_TO_IDX[move.color]
        delta_tok_c[cidx] += 2.0
        opp_pressure = _scarcity_pressure([cidx], bank_c)
        turn_progress = _turn_progress_take(delta_tok_c, 0, me, valid_moves)
        if needed and cidx is not None:
            noble_synergy = 1.0 if cidx in needed else 0.0

    feats: List[float] = []
    feats.extend(type_one_hot)                                 # 4
    feats.append(_clip01(delta_points))                        # 1
    feats.extend(bonus_gain_oh.tolist())                       # N_COLORS
    feats.extend((delta_tok_c / MAX_TOKENS_IN_HAND).tolist())  # N_COLORS
    feats.append(_norm(delta_tok_g, MAX_BANK_GOLD))            # 1
    feats.append(_clip01(opp_pressure))                        # 1
    feats.append(_clip01(turn_progress))                       # 1
    feats.append(_clip01(noble_synergy))                       # --- NEW ---
    return _to_f32(feats)

# =========================
# API publiczne
# =========================

def encode_state_and_moves(
    state: GameState,
    valid_moves: List[Move]
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Czysty enkoder cech (bez efektów ubocznych).
    Zwraca:
      - state_vec: [STATE_DIM] float32
      - move_vecs: list wektorów [MOVE_DIM] float32, w tej samej kolejności co valid_moves
    """
    state_vec = _encode_state(state)
    move_vecs = [_encode_move(mv, state, valid_moves) for mv in valid_moves]
    return state_vec, move_vecs

# =========================
# Wymiary cech
# =========================

STATE_DIM: int = (
    1 + N_COLORS + N_COLORS + 1 + 1               # ja
    + 1 + 1 + N_COLORS + N_COLORS                 # oponenci punkty/bonusy (max/avg)
    + N_COLORS + 1                                # bank
    + N_COLORS + N_COLORS + 1                     # NEW: visible + nobles_share + noble_dist
    + N_COLORS + 1 + 1                            # NEW: opp tokens max + opp gold max + opp reserved max
    + 1                                           # NEW: rel_points (opcjonalnie, jeśli dodałeś)
    + 1                                           # turn
)

MOVE_DIM: int = 4 + 1 + N_COLORS + N_COLORS + 1 + 1 + 1 + 1  # ostatnie +1 to noble_synergy

__all__ = ["encode_state_and_moves", "STATE_DIM", "MOVE_DIM", "COLOR_ORDER"]