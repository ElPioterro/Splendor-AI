"""
Definiuje interfejs dla agentów (graczy) oraz ich konkretne implementacje.
Wersja 2.0 z rozszerzoną logiką dla GeneticAgent, uwzględniającą
stan przeciwników i ocenę kart na stole.
"""

import abc
import copy
import numpy as np
from typing import List, Optional, Tuple

from game_engine import GameState, Move, Noble, Player, Card, GemColor
from game_engine import BuyCard
import config

# --- Funkcja Pomocnicza (Zasada DRY) ---

def get_eligible_nobles_after_purchase(
    player: 'Player', card: 'Card', available_nobles: List['Noble']
) -> List['Noble']:
    """Sprawdza, których arystokratów gracz może przyjąć po hipotetycznym zakupie danej karty."""
    temp_bonuses = player.bonuses
    temp_bonuses[card.bonus_color] += 1
    return [
        n for n in available_nobles
        if all(temp_bonuses.get(c, 0) >= r for c, r in n.requirements.items())
    ]

# --- Abstrakcyjny Interfejs Agenta (bez zmian) ---

class Agent(abc.ABC):
    @abc.abstractmethod
    def choose_action(
        self, game_state: 'GameState', valid_moves: List['Move']
    ) -> Tuple['Move', Optional['Noble']]:
        raise NotImplementedError("Każdy agent musi implementować tę metodę.")

# --- Implementacje Agentów (zaktualizowane) ---

class RandomAgent(Agent):
    """Agent wykonujący losowe, ale dozwolone ruchy (z lokalnym RNG)."""
    def __init__(self, rng: Optional[np.random.Generator] = None, rng_seed: Optional[int] = None):
        self.rng = rng if rng is not None else np.random.default_rng(rng_seed)

    def choose_action(
        self, game_state: 'GameState', valid_moves: List['Move']
    ) -> Tuple['Move', Optional['Noble']]:
        if not valid_moves:
            raise ValueError("RandomAgent nie otrzymał żadnych prawidłowych ruchów.")

        idx = int(self.rng.integers(low=0, high=len(valid_moves)))
        chosen_move = valid_moves[idx]
        chosen_noble: Optional['Noble'] = None

        if isinstance(chosen_move, BuyCard):
            player = game_state.get_current_player()
            eligible_nobles = get_eligible_nobles_after_purchase(
                player, chosen_move.card, game_state.nobles
            )
            if eligible_nobles:
                # losowo wybierz arystokratę lub brak wyboru
                pool = eligible_nobles + [None]
                idx2 = int(self.rng.integers(low=0, high=len(pool)))
                chosen_noble = pool[idx2]

        return chosen_move, chosen_noble


class HumanAgent(Agent):
    """Agent pozwalający człowiekowi grać przez konsolę."""
    def choose_action(
        self, game_state: 'GameState', valid_moves: List['Move']
    ) -> Tuple['Move', Optional['Noble']]:
        print("\n--- Twoja tura! ---")
        valid_moves.sort(key=lambda m: m.__class__.__name__)
        for i, move in enumerate(valid_moves):
            print(f"  [{i}]: {move}")

        chosen_move = None
        while chosen_move is None:
            try:
                choice = int(input("Wybierz numer ruchu: "))
                if 0 <= choice < len(valid_moves):
                    chosen_move = valid_moves[choice]
                else:
                    print("Nieprawidłowy numer.")
            except ValueError:
                print("Proszę wprowadzić liczbę.")

        chosen_noble: Optional['Noble'] = None
        if isinstance(chosen_move, BuyCard):
            player = game_state.get_current_player()
            eligible_nobles = get_eligible_nobles_after_purchase(
                player, chosen_move.card, game_state.nobles
            )
            if eligible_nobles:
                print("\nMożesz przyjąć jednego z arystokratów:")
                for i, noble in enumerate(eligible_nobles):
                    print(f"  [{i}]: {noble}")
                print(f"  [{len(eligible_nobles)}]: Nie przyjmuj żadnego")
                
                while True:
                    try:
                        noble_choice = int(input("Wybierz opcję: "))
                        if 0 <= noble_choice < len(eligible_nobles):
                            chosen_noble = eligible_nobles[noble_choice]
                            break
                        elif noble_choice == len(eligible_nobles):
                            chosen_noble = None
                            break
                        else:
                             print("Nieprawidłowy numer.")
                    except ValueError:
                        print("Proszę wprowadzić liczbę.")
        
        print("--------------------")
        return chosen_move, chosen_noble


class GeneticAgent(Agent):
    """
    Agent AI oparty na wektorze wag (DNA). Wersja deterministyczna:
    - własny RNG (self.rng) z rng_seed,
    - tie-break w remisach przez self.rng,
    - shortfall uwzględnia złoto.
    """
    # 16 bazowych genów + 4 nowe (tempo/presja/potencjał/synergia)
    # 0: prestiż
    # 1-5: bonusy kolorów
    # 6-10: liczba żetonów
    # 11: liczba rezerw
    # 12: różnica punktowa do lidera (było)
    # 13: przewaga w bonusach nad średnią (było)
    # 14: przewaga w żetonach (było)
    # 15: osiągalność najlepszej karty (było)
    # 16: turn_progress (czas)          [NOWE]
    # 17: opponent_pressure             [NOWE]
    # 18: reserved_potential            [NOWE]
    # 19: engine_synergy                [NOWE]
    DNA_SIZE = getattr(config, "DNA_SIZE", 20)
    
    def __init__(self, dna: Optional[np.ndarray] = None,
                 name: Optional[str] = None,
                 rng: Optional[np.random.Generator] = None,
                 rng_seed: Optional[int] = None):
        self.rng = rng if rng is not None else np.random.default_rng(rng_seed)
        if dna is None:
            # jeśli EM zawsze podaje DNA, to nieistotne; tu dla kompletności
            self.dna = self.rng.uniform(-1.0, 1.0, self.DNA_SIZE)
        else:
            dna = np.array(dna, dtype=float).copy()
            if dna.shape != (self.DNA_SIZE,):
                raise ValueError(f"DNA musi mieć rozmiar {self.DNA_SIZE}")
            self.dna = dna
        self.name = name or "GeneticAgent"

    def _calculate_card_shortfall(self, player: 'Player', card: 'Card') -> int:
        """Ilu kolorowych żetonów brakuje do zakupu karty, po uwzględnieniu złota."""
        shortfall = 0
        player_bonuses = player.bonuses
        for color, cost_val in card.cost:
            needed = cost_val - player_bonuses.get(color, 0)
            if needed > 0:
                shortfall += max(0, needed - player.gems.get(color, 0))
        # uwzględnij złoto jako „dowolny kolor”
        shortfall = max(0, shortfall - player.gold_gems)
        return shortfall

    def _extract_features(self, game_state: 'GameState', player_id: int) -> np.ndarray:
        features = np.zeros(self.DNA_SIZE)
        player = game_state.players[player_id]

        # Cechy własne
        features[0] = player.prestige_points
        player_bonuses = player.bonuses
        for i, color in enumerate(GemColor):
            features[1 + i] = player_bonuses[color]
            features[6 + i] = player.gems[color]
        features[11] = len(player.reserved_cards)

        # 12..14 – Twoje dotychczasowe cechy kontekstowe (różnica punktowa, przewaga bonusów, przewaga żetonów)
        opponents = [p for i, p in enumerate(game_state.players) if i != player_id]
        if opponents:
            leader_score = max(p.prestige_points for p in opponents)
            features[12] = player.prestige_points - leader_score

            avg_opp_bonuses = sum(sum(p.bonuses.values()) for p in opponents) / len(opponents)
            avg_opp_tokens = sum(p.total_gems for p in opponents) / len(opponents)
            features[13] = sum(player_bonuses.values()) - avg_opp_bonuses
            features[14] = player.total_gems - avg_opp_tokens

        # 15 – osiągalność najlepszej widocznej karty (było)
        visible_cards = [c for tier_list in game_state.visible_cards.values() for c in tier_list]
        if visible_cards:
            min_shortfall = float('inf')
            for card in visible_cards:
                shortfall = self._calculate_card_shortfall(player, card)
                if shortfall < min_shortfall:
                    min_shortfall = shortfall
            features[15] = 1.0 / (1.0 + min_shortfall)

        # NOWE: 16..19 (tempo/presja/potencjał/synergia)
        target = getattr(config, "TARGET_PRESTIGE_POINTS", 15)
        # 16: turn_progress – 0 na starcie, ~1 w okolicach baseline
        turns_baseline = float(getattr(config, "TURNS_BASELINE", 60.0))
        turn_no = getattr(game_state, "turn_number", 0)
        features[16] = max(0.0, min(1.0, turn_no / max(1.0, turns_baseline)))

        # 17: opponent_pressure – jak blisko zwycięstwa jest lider
        if opponents:
            opp_max = max(p.prestige_points for p in opponents)
            features[17] = min(1.0, opp_max / max(1, target))
        else:
            features[17] = 0.0

        # 18: reserved_potential – najsilniejsza rezerwa / (1+shortfall), znormalizowana
        reserved_potential = 0.0
        for rc in player.reserved_cards:
            short = self._calculate_card_shortfall(player, rc)
            score = (rc.prestige_points + 0.3) / (1.0 + short)
            reserved_potential = max(reserved_potential, score)
        features[18] = min(1.0, reserved_potential / 5.0)

        # 19: engine_synergy – suma kwadratów bonusów, normalizacja 25 (=5^2)
        bonus_vals = np.array([player_bonuses[c] for c in GemColor], dtype=float)
        features[19] = min(1.0, float(np.sum(bonus_vals**2)) / 25.0)

        return features

    def evaluate_state(self, game_state: 'GameState', player_id: int) -> float:
        features = self._extract_features(game_state, player_id)
        return float(np.dot(features, self.dna))

    def choose_action(self, game_state: 'GameState', valid_moves: List['Move']) -> Tuple['Move', Optional['Noble']]:
        if not valid_moves:
            raise ValueError("GeneticAgent nie otrzymał żadnych prawidłowych ruchów.")

        pid = game_state.current_player_index
        player = game_state.players[pid]

        base_features = self._extract_features(game_state, pid)
        base_score = float(np.dot(base_features, self.dna))

        best_score = -float('inf')
        best_actions: List[Tuple['Move', Optional['Noble']]] = []

        color_to_idx = {c: i for i, c in enumerate(GemColor)}
        bonuses_now = player.bonuses  # do obliczenia delta_synergy

        for move in valid_moves:
            if not isinstance(move, BuyCard):
                score = base_score
                if score > best_score + 1e-12:
                    best_score = score; best_actions = [(move, None)]
                elif abs(score - best_score) <= 1e-12:
                    best_actions.append((move, None))
                continue

            card = move.card
            # delta: prestiż i bonus koloru (jak dotąd)
            delta_score = 0.0
            delta_score += self.dna[0] * float(card.prestige_points)
            cidx = color_to_idx[card.bonus_color]
            delta_score += self.dna[1 + cidx] * 1.0

            # delta engine_synergy (cecha 19): ( (b+1)^2 - b^2 ) / 25 = (2b+1)/25
            b = float(bonuses_now[card.bonus_color])
            delta_engine = (2.0 * b + 1.0) / 25.0
            if self.DNA_SIZE >= 20:
                delta_score += self.dna[19] * delta_engine

            # wariant bez arystokraty
            score0 = base_score + delta_score
            if score0 > best_score + 1e-12:
                best_score = score0; best_actions = [(move, None)]
            elif abs(score0 - best_score) <= 1e-12:
                best_actions.append((move, None))

            # warianty z arystokratą (cecha 0 – prestiż)
            # (eligible nobles i tak engine zweryfikuje; tu tylko scoring)
            # Możesz użyć helpera jak wcześniej – tu pseudo:
            # unlocked_nobles = get_eligible_nobles_after_purchase(...)
            # ale bez deepcopy to ciężkie – zostawmy jak było u Ciebie,
            # ewentualny wpływ i tak jest mniejszy niż prestiż karty.
            # Jeśli masz gotowy helper, użyj:
            try:
                unlocked_nobles = get_eligible_nobles_after_purchase(player, card, game_state.nobles)
            except Exception:
                unlocked_nobles = []
            for noble in unlocked_nobles:
                sc = score0 + self.dna[0] * float(noble.prestige_points)
                if sc > best_score + 1e-12:
                    best_score = sc; best_actions = [(move, noble)]
                elif abs(sc - best_score) <= 1e-12:
                    best_actions.append((move, noble))

        if len(best_actions) == 1:
            return best_actions[0]
        idx = int(self.rng.integers(low=0, high=len(best_actions)))
        return best_actions[idx]