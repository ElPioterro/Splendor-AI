# tests/test_game_determinism.py
import random
from game_engine import Game

def snapshot(gs):
    return dict(
        visible={tier: [c.id for c in gs.visible_cards[tier]] for tier in (1, 2, 3)},
        decks={tier: [c.id for c in gs.decks[tier]] for tier in (1, 2, 3)},
        nobles=[n.id for n in gs.nobles],
    )

def test_game_same_seed_is_deterministic(all_cards, all_nobles):
    seed = 123
    g1 = Game(all_cards, all_nobles, seed=seed, target_points=15)
    s1 = snapshot(g1.setup_new_game(["A", "B"]))

    g2 = Game(all_cards, all_nobles, seed=seed, target_points=15)
    s2 = snapshot(g2.setup_new_game(["A", "B"]))

    assert s1 == s2, "Ten sam seed musi dać identyczny układ kart i arystokratów"

def test_game_not_affected_by_global_random(all_cards, all_nobles):
    seed = 777
    random.seed(999)
    g1 = Game(all_cards, all_nobles, seed=seed, target_points=15)
    s1 = snapshot(g1.setup_new_game(["A", "B"]))

    random.seed(42)
    g2 = Game(all_cards, all_nobles, seed=seed, target_points=15)
    s2 = snapshot(g2.setup_new_game(["A", "B"]))

    assert s1 == s2, "Globalny random.seed nie może wpływać na Game z lokalnym RNG"

def test_game_different_seeds_vary_setup(all_cards, all_nobles):
    g1 = Game(all_cards, all_nobles, seed=1, target_points=15)
    s1 = snapshot(g1.setup_new_game(["A", "B"]))
    g2 = Game(all_cards, all_nobles, seed=2, target_points=15)
    s2 = snapshot(g2.setup_new_game(["A", "B"]))

    # Różna próbka: albo nobles, albo widoczne karty/deck
    different = (s1["nobles"] != s2["nobles"]) or any(
        s1["visible"][t] != s2["visible"][t] for t in (1, 2, 3)
    )
    assert different, "Różne seedy powinny w praktyce dawać inny setup"