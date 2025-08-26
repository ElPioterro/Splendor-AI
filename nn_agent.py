# nn_agent.py
# Minimalny, szybki agent NN z MLP (numpy) dla odnogi nn-agent.
# Kontrakt: choose_action(state, valid_moves) -> (Move, Optional[Noble])
# - SSOT: brak modyfikacji GameState
# - Czystość: deterministyczny tie-break z lokalnym RNG
# - Zero frameworków ML: tylko numpy

from __future__ import annotations

from typing import List, Tuple, Optional, TYPE_CHECKING
import numpy as np
import random

# Runtime-only dostęp do klas silnika (do isinstance, kompatybilność z Pylance)
import game_engine as ge
import nn_config as cfg

# Enkoder cech
from feature_encoder import encode_state_and_moves, STATE_DIM, MOVE_DIM

if TYPE_CHECKING:
    from game_engine import GameState, Move, Noble

# =========================
# Aktywacje
# =========================

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0, out=x)

def _silu(x: np.ndarray) -> np.ndarray:
    # x * sigmoid(x); implementacja stabilna numerycznie
    # unika overflow przy dużych |x|
    out = x.copy()
    pos = x >= 0
    neg = ~pos
    # sigmoid for pos: 1/(1+exp(-x)); for neg: exp(x)/(1+exp(x))
    out[pos] = x[pos] / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = x[neg] * (ex / (1.0 + ex))
    return out

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x, out=x)

_ACTS = {
    "relu": _relu,
    "silu": _silu,
    "tanh": _tanh,
}

# =========================
# Pomoc: kształty parametrów i (un)flatten
# =========================

def param_shapes(in_dim: int, h1: int, h2: int) -> List[Tuple[int, ...]]:
    return [
        (in_dim, h1), (h1,),
        (h1, h2),     (h2,),
        (h2, 1),      (1,),
    ]

def param_count(in_dim: int = None, h1: int = None, h2: int = None) -> int:
    """
    Zwraca liczbę parametrów MLP. Jeśli brak argumentów, używa
    STATE_DIM+MOVE_DIM oraz HIDDEN1/HIDDEN2 z nn_config.
    """
    if in_dim is None:
        in_dim = STATE_DIM + MOVE_DIM
    if h1 is None:
        h1 = cfg.HIDDEN1
    if h2 is None:
        h2 = cfg.HIDDEN2
    return sum(int(np.prod(s)) for s in param_shapes(in_dim, h1, h2))

def _unflatten(theta: np.ndarray, shapes: List[Tuple[int, ...]]) -> List[np.ndarray]:
    params = []
    idx = 0
    for s in shapes:
        n = int(np.prod(s))
        chunk = theta[idx: idx + n]
        params.append(chunk.reshape(s))
        idx += n
    if idx != theta.size:
        raise ValueError(f"Rozmiar theta ({theta.size}) nie zgadza się z sumą kształtów ({idx}).")
    return params

def _flatten(params: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([p.reshape(-1) for p in params]).astype(np.float32, copy=False)

def glorot_init(rng: np.random.RandomState, in_dim: int, h1: int, h2: int) -> np.ndarray:
    shapes = param_shapes(in_dim, h1, h2)
    pars: List[np.ndarray] = []

    def xavier(shape):
        if len(shape) == 2:
            fan_in, fan_out = shape
        elif len(shape) == 1:
            # bias
            return np.zeros(shape, dtype=np.float32)
        else:
            raise ValueError("Nieoczekiwany kształt parametru")
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, size=shape).astype(np.float32)

    for s in shapes:
        pars.append(xavier(s))
    return _flatten(pars)

# =========================
# MLP
# =========================

class TinyMLP:
    def __init__(self, in_dim: int, h1: int, h2: int, act: str = "silu"):
        if act not in _ACTS:
            raise ValueError(f"Nieznana aktywacja: {act}. Dozwolone: {list(_ACTS.keys())}")
        self.in_dim = in_dim
        self.h1 = h1
        self.h2 = h2
        self.act_name = act
        self.act_fn = _ACTS[act]
        self.shapes = param_shapes(in_dim, h1, h2)
        self.params: List[np.ndarray] = []  # [W1, b1, W2, b2, W3, b3]

    def set_params_from_theta(self, theta: np.ndarray):
        theta = np.asarray(theta, dtype=np.float32)
        if theta.size != param_count(self.in_dim, self.h1, self.h2):
            raise ValueError("Rozmiar theta niezgodny z architekturą sieci.")
        self.params = _unflatten(theta, self.shapes)

    def theta(self) -> np.ndarray:
        if not self.params:
            raise ValueError("Parametry sieci nie ustawione.")
        return _flatten(self.params)

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        # X: [N, in_dim] -> out: [N] (skalar per próbka)
        if not self.params:
            raise ValueError("Parametry sieci nie ustawione.")
        W1, b1, W2, b2, W3, b3 = self.params
        H1 = X @ W1 + b1  # [N, h1]
        H1 = self.act_fn(H1)
        H2 = H1 @ W2 + b2  # [N, h2]
        H2 = self.act_fn(H2)
        Y = H2 @ W3 + b3   # [N, 1]
        return Y.reshape(-1)  # [N]

# =========================
# Agent NN
# =========================

class NnAgent:
    """
    Polityka: ocena f([state_vec, move_vec]) -> skalar; wybór argmax.
    - Interfejs zgodny z GA: choose_action -> (Move, Optional[Noble])
    - Noble wybiera silnik (SSOT) po apply_move, finalize_turn
    """
    def __init__(
        self,
        theta: np.ndarray,
        hidden1: int = 64,
        hidden2: int = 64,
        act: str = "silu",
        rng_seed: Optional[int] = None,
        name: str = "NnAgent",
    ):
        self.name = name
        self.rng = random.Random(rng_seed)
        self.in_dim = int(STATE_DIM + MOVE_DIM)
        self.h1 = int(hidden1)
        self.h2 = int(hidden2)
        self.net = TinyMLP(self.in_dim, self.h1, self.h2, act=act)
        self.net.set_params_from_theta(theta.astype(np.float32))

    @staticmethod
    def param_count(hidden1: int = 64, hidden2: int = 64) -> int:
        return param_count(STATE_DIM + MOVE_DIM, hidden1, hidden2)

    @staticmethod
    def init_theta(
        rng_seed: int = 1234, hidden1: int = 64, hidden2: int = 64
    ) -> np.ndarray:
        rng = np.random.RandomState(rng_seed)
        return glorot_init(rng, STATE_DIM + MOVE_DIM, hidden1, hidden2)

    def set_params(self, theta: np.ndarray):
        self.net.set_params_from_theta(theta)

    def get_params(self) -> np.ndarray:
        return self.net.theta()

    def _scores_for_moves(self, state_vec: np.ndarray, move_vecs: List[np.ndarray]) -> np.ndarray:
        # Batch: konkatenacja state z każdym move -> [K, in_dim]
        k = len(move_vecs)
        X = np.empty((k, self.in_dim), dtype=np.float32)
        # kopiowanie state_vec do wszystkich wierszy
        X[:, :STATE_DIM] = state_vec.reshape(1, -1)
        # wstawienie wektorów ruchów
        for i, mv in enumerate(move_vecs):
            X[i, STATE_DIM:] = mv
        return self.net.forward_batch(X)

    def choose_action(
        self,
        state: "GameState",
        valid_moves: List["Move"],
    ) -> Tuple["Move", Optional["Noble"]]:
        """
        Zwraca (Move, None). Noble wybiera silnik (SSOT) po apply_move.
        """
        if not valid_moves:
            raise ValueError("Brak legalnych ruchów.")

        state_vec, move_vecs = encode_state_and_moves(state, valid_moves)
        scores = self._scores_for_moves(state_vec, move_vecs)

        # Argmax + deterministyczny tie-break z lokalnym RNG
        best_val = float(np.max(scores))
        ties = np.flatnonzero(np.isclose(scores, best_val, rtol=1e-7, atol=1e-9))
        if ties.size == 1:
            idx = int(ties[0])
        else:
            idx = int(self.rng.choice(list(ties)))  # lokalny RNG; deterministyczny po seed

        return valid_moves[idx], None