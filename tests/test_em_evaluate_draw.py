# tests/test_em_evaluate_draw.py
import numpy as np
import config
from agents import GeneticAgent
from evolution_manager import EvolutionManager

def test_evaluate_candidate_counts_draws(all_cards, all_nobles, monkeypatch):
    em = EvolutionManager(all_cards, all_nobles)

    # Patch: każda gra to remis 10:10 w 30 turach
    def fake_play_one_game(self, dnaA, anchor, seed, swap_roles):
        return (-1, 10, 10, 30)

    monkeypatch.setattr(EvolutionManager, "_play_one_game", fake_play_one_game, raising=True)

    anchors = [GeneticAgent(dna=np.zeros(config.DNA_SIZE))]
    metrics = em._evaluate_candidate(np.zeros(config.DNA_SIZE), anchors, gen_idx=0)

    assert abs(metrics["win_rate"] - 0.5) < 1e-9
    assert abs(metrics["avg_margin"] - 0.0) < 1e-9

    wr = 0.5
    points_norm = 10/15
    margin_norm = 0.0
    tb = getattr(config, "TURNS_BASELINE", 60.0)
    tpw = getattr(config, "TURNS_PENALTY_WEIGHT", 1.2)
    sbw = getattr(config, "SPEED_BONUS_WEIGHT", 0.8)
    penalty = max(0.0, (30 - tb) / tb)
    speed_bonus = max(0.0, (tb - 30) / tb)
    expected = (12.0*wr) + (3.0*margin_norm) + (1.5*points_norm) - (tpw*penalty) + (sbw*points_norm*speed_bonus)

    assert abs(metrics["fitness"] - expected) < 1e-6