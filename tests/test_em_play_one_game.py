# tests/test_em_play_one_game.py
import numpy as np
import config
from agents import GeneticAgent
from evolution_manager import EvolutionManager

def test_play_one_game_is_deterministic(all_cards, all_nobles, monkeypatch):
    # Przyspiesz grę: 5 punktów zwycięstwa
    original_target = getattr(config, "TARGET_PRESTIGE_POINTS", 15)
    monkeypatch.setattr(config, "TARGET_PRESTIGE_POINTS", 5, raising=False)

    em = EvolutionManager(all_cards, all_nobles)

    dna_cand = np.zeros(config.DNA_SIZE)
    anchor = GeneticAgent(dna=np.zeros(config.DNA_SIZE), rng_seed=999)

    res1 = em._play_one_game(dna_cand, anchor, seed=2024, swap_roles=False)
    res2 = em._play_one_game(dna_cand, anchor, seed=2024, swap_roles=False)
    assert res1 == res2, "Ten sam seed i układ ról => identyczny wynik"

    res3 = em._play_one_game(dna_cand, anchor, seed=2024, swap_roles=True)
    res4 = em._play_one_game(dna_cand, anchor, seed=2024, swap_roles=True)
    assert res3 == res4, "Ten sam seed (z XOR dla swap) => identyczny wynik przy odwróconych rolach"

    # Posprzątaj
    monkeypatch.setattr(config, "TARGET_PRESTIGE_POINTS", original_target, raising=False)