import numpy as np
from typing import Tuple, Optional, List
from game_engine import GameState, Move
from feature_encoder import encode_state_and_moves
import nn_config as NN

def _act(x: np.ndarray) -> np.ndarray:
    if NN.ACT == "relu":
        return np.maximum(0, x)
    if NN.ACT == "silu":
        return x/(1.0 + np.exp(-x))
    if NN.ACT == "tanh":
        return np.tanh(x)
    return np.maximum(0, x)

def _shapes() -> Tuple[List[Tuple[int,int]], List[int]]:
    in_dim = NN.STATE_DIM + NN.MOVE_DIM
    shapes = [
        (in_dim, NN.HIDDEN1),
        (NN.HIDDEN1, NN.HIDDEN2),
        (NN.HIDDEN2, 1)
    ]
    biases = [NN.HIDDEN1, NN.HIDDEN2, 1]
    return shapes, biases

def param_count() -> int:
    shp, bs = _shapes()
    return sum(w*h + b for (w,h), b in zip(shp, bs))

def unflatten(theta: np.ndarray):
    shp, bs = _shapes()
    params = []
    idx = 0
    for (w, h), b in zip(shp, bs):
        W = theta[idx: idx + w*h].reshape(w, h); idx += w*h
        bvec = theta[idx: idx + b]; idx += b
        params.append((W, bvec))
    return params

def forward(params, x: np.ndarray) -> float:
    W1,b1 = params[0]; W2,b2 = params[1]; W3,b3 = params[2]
    h1 = _act(x @ W1 + b1)
    h2 = _act(h1 @ W2 + b2)
    y  = h2 @ W3 + b3
    return float(y.squeeze())

class NnAgent:
    def __init__(self, theta: np.ndarray, rng_seed: Optional[int]=None, name: str="NnAgent"):
        self.theta = theta.copy()
        self.params = unflatten(self.theta)
        self.rng = np.random.default_rng(rng_seed)
        self.name = name

    def choose_action(self, state: GameState, valid_moves: List[Move]) -> Tuple[Move, Optional[object]]:
        if not valid_moves:
            raise ValueError("NnAgent: no valid moves")
        s, moves = encode_state_and_moves(state, valid_moves)
        scores = []
        for mv in moves:
            x = np.concatenate([s, mv], axis=0)
            scores.append(forward(self.params, x))
        idxs = np.flatnonzero(np.isclose(scores, np.max(scores)))
        idx = int(self.rng.choice(idxs))
        # NN nie wybiera szlachcica; engine poda eligible po apply_move; zwrócimy None
        return valid_moves[idx], None