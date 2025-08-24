# tests/test_engine_rules.py
import pytest
from game_engine import Game, GameState, Player, GemColor, Card, Noble, BuyCard

def test_finalize_turn_rejects_ineligible_noble(all_cards, all_nobles):
    # Zapewnij, że pierwszy szlachcic wymaga np. WHITE:3
    noble = Noble(id=999, prestige_points=3, requirements={GemColor.WHITE: 1})
    g = Game(all_cards, all_nobles + [noble], seed=123, target_points=15)
    gs = g.setup_new_game(["A", "B"])
    gs.nobles.append(noble)  # dodajemy naszego testowego

    # Gracz nie spełnia wymagań -> błąd
    with pytest.raises(ValueError):
        g.finalize_turn(noble)

def test_apply_move_returns_eligible_nobles(all_cards, all_nobles):
    # Przygotuj darmową kartę z bonusem WHITE, która odblokuje nobles WHITE:1
    free_white_card = Card(
        id=10001, level=1, prestige_points=0, bonus_color=GemColor.WHITE, cost=tuple()
    )
    noble_white = Noble(id=10002, prestige_points=3, requirements={GemColor.WHITE: 1})

    g = Game(all_cards, all_nobles + [noble_white], seed=123, target_points=15)
    gs = g.setup_new_game(["A", "B"])
    # Wstaw kartę na stół, aby BuyCard był spójny
    gs.visible_cards[1][0] = free_white_card
    gs.nobles.append(noble_white)

    # Kupno darmowej karty powinno zwrócić nobles do których gracz się kwalifikuje
    nobles = g.apply_move(BuyCard(card=free_white_card))
    assert noble_white in nobles, "Po zakupie bonusu WHITE gracz powinien kwalifikować się do nobles WHITE:1"

    # finalize_turn powinien przyjąć wybranego (kwalifikowalnego) arystokratę bez błędu
    g.finalize_turn(noble_white)