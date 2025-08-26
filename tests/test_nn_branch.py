# tests/test_nn_branch.py
import numpy as np
import config
from game_loader import GameLoader
from game_engine import Game
from feature_encoder import encode_state_and_moves, STATE_DIM, MOVE_DIM

def test_encoder_shapes():
    loader = GameLoader()
    cards, nobles = loader.load_definitions_from_json(config.DATA_FILE)
    game = Game(all_cards=cards, all_nobles=nobles, seed=123)
    game.setup_new_game(["A","B"])
    valid = game.get_valid_moves()
    s, mv = encode_state_and_moves(game.game_state, valid)
    assert s.shape[0] == STATE_DIM
    assert all(v.shape[0] == MOVE_DIM for v in mv)