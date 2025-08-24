# game_engine.py
#
# Serce i Rdzeń Logiki Gry.
# Wersja z poprawnym rozdzieleniem odpowiedzialności za wybór arystokraty.

import random
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Dict, List, Optional, Tuple

# --- Podstawowe Typy i Enumy ---
class GemColor(str, Enum):
    WHITE = "white"; BLUE = "blue"; GREEN = "green"; RED = "red"; BLACK = "black"

# --- Struktury Danych Gry ---
@dataclass(frozen=True, eq=True)
class Card:
    id: int; level: int; prestige_points: int; bonus_color: GemColor; cost: Tuple[Tuple[GemColor, int], ...]

@dataclass(frozen=True, eq=True)
class Noble:
    id: int; prestige_points: int; requirements: Dict[GemColor, int]

@dataclass
class Player:
    name: str
    gems: Dict[GemColor, int] = field(default_factory=lambda: {c: 0 for c in GemColor})
    gold_gems: int = 0
    cards: List[Card] = field(default_factory=list)
    reserved_cards: List[Card] = field(default_factory=list)
    nobles: List[Noble] = field(default_factory=list)
    @property
    def prestige_points(self) -> int: return sum(c.prestige_points for c in self.cards) + sum(n.prestige_points for n in self.nobles)
    @property
    def bonuses(self) -> Dict[GemColor, int]:
        b = {c: 0 for c in GemColor}; [b.update({c.bonus_color: b[c.bonus_color] + 1}) for c in self.cards]; return b
    @property
    def total_gems(self) -> int: return sum(self.gems.values()) + self.gold_gems

@dataclass
@dataclass
class GameState:
    players: List[Player]
    available_gems: Dict[GemColor, int]
    gold_gems: int
    decks: Dict[int, List[Card]]
    visible_cards: Dict[int, List[Card]]
    nobles: List[Noble]
    current_player_index: int = 0
    turn_number: int = 0
    def get_current_player(self) -> Player: return self.players[self.current_player_index]

# --- Definicje Ruchów ---
@dataclass(frozen=True)
class Move: pass
@dataclass(frozen=True)
class TakeThreeGems(Move): colors: Tuple[GemColor, GemColor, GemColor]
@dataclass(frozen=True)
class TakeTwoGems(Move): color: GemColor
@dataclass(frozen=True)
class ReserveVisibleCard(Move): card: Card
@dataclass(frozen=True)
class ReserveFromDeck(Move): tier: int
@dataclass(frozen=True)
class BuyCard(Move): card: Card

# --- Główny Silnik Gry (z kluczowymi zmianami) ---

class Game:
    def __init__(self, all_cards: List[Card], all_nobles: List[Noble], seed: Optional[int] = None, target_points: int = 15):
        self.all_cards = all_cards
        self.all_nobles = all_nobles
        self.game_state: Optional[GameState] = None
        self._game_end_triggered: bool = False
        self._final_player_index: Optional[int] = None
        self.rng = random.Random(seed)         # lokalny RNG
        self.target_points = target_points     # próg zwycięstwa (patrz też config.TARGET_PRESTIGE_POINTS)

    def setup_new_game(self, player_names: List[str]) -> GameState:
        num_players = len(player_names)
        if not 2 <= num_players <= 4:
            raise ValueError("Liczba graczy musi być pomiędzy 2 a 4.")
        players = [Player(name=name) for name in player_names]
        gem_counts = {2: 4, 3: 5, 4: 7}
        available_gems = {color: gem_counts[num_players] for color in GemColor}

        decks = {1: [], 2: [], 3: []}
        for card in self.all_cards:
            decks[card.level].append(card)
        for deck in decks.values():
            self.rng.shuffle(deck)  # zamiast globalnego random

        visible_cards = {tier: [decks[tier].pop() for _ in range(4) if decks[tier]] for tier in decks}
        nobles = self.rng.sample(self.all_nobles, k=num_players + 1)  # zamiast globalnego random

        self.game_state = GameState(
            players=players, available_gems=available_gems, gold_gems=5,
            decks=decks, visible_cards=visible_cards, nobles=nobles
        )
        self._game_end_triggered = False
        self._final_player_index = None
        return self.game_state

    def get_valid_moves(self) -> List[Move]:
        state = self.game_state; player = state.get_current_player()
        moves: List[Move] = []
        if player.total_gems <= 7:
            available_colors = [c for c, count in state.available_gems.items() if count > 0]
            for combo in combinations(available_colors, 3): moves.append(TakeThreeGems(colors=tuple(sorted(combo, key=lambda x: x.value))))
        if player.total_gems <= 8:
            for color, count in state.available_gems.items():
                if count >= 4: moves.append(TakeTwoGems(color=color))
        if len(player.reserved_cards) < 3:
            for cards in state.visible_cards.values():
                for card in cards: moves.append(ReserveVisibleCard(card=card))
            for tier, deck in state.decks.items():
                if deck: moves.append(ReserveFromDeck(tier=tier))
        buyable_cards = [card for tier_cards in state.visible_cards.values() for card in tier_cards]
        buyable_cards.extend(player.reserved_cards)
        for card in set(buyable_cards): # set() to handle duplicates from visible and reserved
            if self._can_player_afford(player, card): moves.append(BuyCard(card=card))
        return moves

    def _can_player_afford(self, player: Player, card: Card) -> bool:
        shortfall = 0; bonuses = player.bonuses
        for color, cost_val in card.cost:
            effective_cost = cost_val - bonuses.get(color, 0)
            if effective_cost > 0:
                shortfall += max(0, effective_cost - player.gems.get(color, 0))
        return shortfall <= player.gold_gems

    def apply_move(self, move: Move) -> List[Noble]:
        """
        Wykonuje główną akcję gracza i ZWRACA listę arystokratów,
        do których gracz się kwalifikuje. NIE kończy tury.
        """
        state = self.game_state; player = state.get_current_player()
        if isinstance(move, TakeThreeGems):
            for color in move.colors: state.available_gems[color] -= 1; player.gems[color] += 1
        elif isinstance(move, TakeTwoGems):
            state.available_gems[move.color] -= 2; player.gems[move.color] += 2
        elif isinstance(move, ReserveVisibleCard):
            self._remove_card_from_visible(move.card); player.reserved_cards.append(move.card)
            if state.gold_gems > 0: state.gold_gems -= 1; player.gold_gems += 1
        elif isinstance(move, ReserveFromDeck):
            if state.decks[move.tier]:
                card = state.decks[move.tier].pop(); player.reserved_cards.append(card)
                if state.gold_gems > 0: state.gold_gems -= 1; player.gold_gems += 1
        elif isinstance(move, BuyCard):
            bonuses = player.bonuses; gold_to_pay = 0
            for color, cost_val in move.card.cost:                
                cost_after_bonus = max(0, cost_val - bonuses.get(color, 0))
                gems_paid = min(player.gems.get(color, 0), cost_after_bonus)
                player.gems[color] -= gems_paid
                state.available_gems[color] += gems_paid
                gold_to_pay += cost_after_bonus - gems_paid
            player.gold_gems -= gold_to_pay; state.gold_gems += gold_to_pay
            if move.card in player.reserved_cards: player.reserved_cards.remove(move.card)
            else: self._remove_card_from_visible(move.card)
            player.cards.append(move.card)
        
        # Sprawdź kwalifikujących się arystokratów i zwróć ich do agenta
        player_bonuses = player.bonuses
        eligible_nobles = [n for n in state.nobles if all(player_bonuses.get(c, 0) >= r for c, r in n.requirements.items())]
        return eligible_nobles

    def finalize_turn(self, chosen_noble: Optional[Noble]):
        """
        Finalizuje turę gracza po dokonaniu wyboru arystokraty (lub jego braku).
        Ta metoda MUSI być wywołana po `apply_move`.
        """
        state = self.game_state
        player = state.get_current_player()

        # Wyznacz arystokratów, do których gracz się kwalifikuje
        player_bonuses = player.bonuses
        eligible_nobles = [n for n in state.nobles if all(player_bonuses.get(c, 0) >= r for c, r in n.requirements.items())]

        # 1) Przypisz wybranego arystokratę – ale tylko jeśli jest kwalifikowalny
        if chosen_noble:
            if chosen_noble not in state.nobles:
                raise ValueError("Wybrany arystokrata nie jest dostępny na stole.")
            if chosen_noble not in eligible_nobles:
                raise ValueError("Wybrany arystokrata nie jest kwalifikowalny dla gracza.")
            state.nobles.remove(chosen_noble)
            player.nobles.append(chosen_noble)

        # 2) Warunek końca gry – użyj target_points
        if not self._game_end_triggered and player.prestige_points >= self.target_points:
            self._game_end_triggered = True
            self._final_player_index = state.current_player_index

        # 3) Następny gracz
        state.current_player_index = (state.current_player_index + 1) % len(state.players)
        state.turn_number += 1 


    def _remove_card_from_visible(self, card_to_remove: Card):
        state = self.game_state
        tier = card_to_remove.level
        row = state.visible_cards.get(tier, [])
        if card_to_remove in row:
            idx = row.index(card_to_remove)
            row.pop(idx)
            if state.decks.get(tier):
                new_card = state.decks[tier].pop()
                row.insert(idx, new_card)  # <-- zamiast append
        # jeśli karta była kupowana z rezerwy, nic nie zmieniamy na stole

    def is_game_over(self) -> bool:
        return self._game_end_triggered and self.game_state.current_player_index == self._final_player_index

    def get_winner(self) -> Optional[Player]:
        if not self.is_game_over(): return None
        return sorted(self.game_state.players, key=lambda p: (-p.prestige_points, len(p.cards)))[0]
