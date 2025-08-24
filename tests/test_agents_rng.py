# tests/test_agents_rng.py
import numpy as np
from game_engine import GameState, Player, GemColor
from game_engine import ReserveFromDeck
from agents import GeneticAgent
from config import DNA_SIZE

def empty_state_2p():
    players = [Player(name="P0"), Player(name="P1")]
    available_gems = {c: 0 for c in GemColor}
    decks = {1: [], 2: [], 3: []}
    visible = {1: [], 2: [], 3: []}
    nobles = []
    return GameState(players=players, available_gems=available_gems, gold_gems=0,
                     decks=decks, visible_cards=visible, nobles=nobles)

def test_genetic_agent_tiebreak_is_seeded():
    state = empty_state_2p()
    moves = [ReserveFromDeck(tier=1), ReserveFromDeck(tier=2), ReserveFromDeck(tier=3)]
    dna =  np.zeros(DNA_SIZE)

    a1 = GeneticAgent(dna=dna, rng_seed=123)
    c1 = a1.choose_action(state, moves)

    a2 = GeneticAgent(dna=dna, rng_seed=123)
    c2 = a2.choose_action(state, moves)

    assert c1 == c2, "Ten sam rng_seed => deterministyczny wybór w remisie"

    a3 = GeneticAgent(dna=dna, rng_seed=124)
    c3 = a3.choose_action(state, moves)
    # Nie wymuszamy różnicy (by uniknąć flakiness), ale drukujemy dla debug
    assert c3 in [(m, None) for m in moves]