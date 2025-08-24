# tests/conftest.py

import pytest
from typing import List, Dict, Tuple
from game_engine import Card, Noble, GemColor

def make_cards(n_per_level: int = 6) -> List[Card]:
    cards = []
    cid = 1
    palette = [GemColor.WHITE, GemColor.BLUE, GemColor.GREEN, GemColor.RED, GemColor.BLACK]
    for level in (1, 2, 3):
        for i in range(n_per_level):
            color = palette[i % len(palette)]
            cost = ((color, level),)  # proste koszty: level sztuk danego koloru
            pp = 1 if level == 3 and (i % 3 == 0) else 0
            cards.append(Card(id=cid, level=level, prestige_points=pp, bonus_color=color, cost=tuple(cost)))
            cid += 1
    return cards

def make_nobles(count: int = 5):
    nobles = []
    palette = [GemColor.WHITE, GemColor.BLUE, GemColor.GREEN, GemColor.RED, GemColor.BLACK]
    for i in range(count):
        c = palette[i % len(palette)]
        nobles.append(Noble(id=i+1, prestige_points=3, requirements={c: 3}))
    return nobles

@pytest.fixture
def all_cards():
    return make_cards(6)

@pytest.fixture
def all_nobles():
    return make_nobles(5)