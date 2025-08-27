# quick_check.py
import numpy as np
from feature_encoder import STATE_DIM, MOVE_DIM
import nn_config as cfg

theta = np.load("nn_champion.npy")
in_dim = STATE_DIM + MOVE_DIM
expected = in_dim*cfg.HIDDEN1 + cfg.HIDDEN1 + cfg.HIDDEN1*cfg.HIDDEN2 + cfg.HIDDEN2 + cfg.HIDDEN2 + 1
print("theta_len =", theta.size, "expected =", expected)