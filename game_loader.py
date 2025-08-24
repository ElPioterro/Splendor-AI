# game_loader.py
#
# Kustosz Danych.
# Odpowiada za wczytanie danych z pliku json, przetworzenie ich i stworzenie
# z nich obiektów (Card, Noble) gotowych do użycia przez silnik gry.

import json
import os
from typing import List, Dict, Any, Tuple

# Aby ten plik działał, musi mieć dostęp do definicji klas z game_engine.
# Używamy warunkowego importu, aby uniknąć błędów cyklicznych,
# chociaż w tej architekturze nie powinny one wystąpić.
try:
    from game_engine import Card, Noble, GemColor
except ImportError:
    print("Błąd: Nie można zaimportować klas z 'game_engine.py'.")
    print("Upewnij się, że plik 'game_engine.py' znajduje się w tym samym katalogu.")
    # Definiujemy atrapy klas, aby edytor kodu nie zgłaszał błędów.
    # W rzeczywistym wykonaniu program zakończy działanie, jeśli import się nie powiedzie.
    from enum import Enum
    class GemColor(str, Enum): pass
    class Card: pass
    class Noble: pass


class GameLoader:
    """
    Klasa odpowiedzialna za wczytywanie definicji gry (kart i arystokratów)
    z zewnętrznych źródeł danych, takich jak pliki JSON.
    """

    def _parse_cards(self, card_data: List[Dict[str, Any]]) -> List[Card]:
        """Konwertuje listę słowników z danymi kart na listę obiektów Card."""
        if not card_data:
            return []
            
        parsed_cards = []
        for data in card_data:
            # Sprawdzenie, czy wszystkie wymagane klucze istnieją
            required_keys = {"id", "level", "prestige_points", "bonus_color", "cost"}
            if not required_keys.issubset(data.keys()):
                raise KeyError(f"Brakujący klucz w danych karty. Wymagane: {required_keys}. Dane: {data}")
            
            # ZMIANA: Konwertujemy słownik kosztów na posortowaną krotkę krotek
            card_cost_dict = {GemColor(k): v for k, v in data.get('cost', {}).items() if v > 0}
            # Sortowanie (sorted) zapewnia, że karty o tym samym koszcie zawsze będą miały tę samą reprezentację
            # co jest dobre dla haszowania i porównań.
            sorted_cost = tuple(sorted(card_cost_dict.items(), key=lambda item: item[0].value))

            card = Card(
                id=data['id'],
                level=data['level'],
                prestige_points=data['prestige_points'],
                bonus_color=GemColor(data['bonus_color']),
                # Tworzymy słownik kosztów, pomijając te o wartości 0 dla czystości
                cost=sorted_cost  # <-- TUTAJ PRZYPISUJEMY KROTKĘ            
                )
            parsed_cards.append(card)
        return parsed_cards

    def _parse_nobles(self, noble_data: List[Dict[str, Any]]) -> List[Noble]:
        """Konwertuje listę słowników z danymi arystokratów na listę obiektów Noble."""
        if not noble_data:
            return []
        parsed_nobles = []
        for data in noble_data:
            required_keys = {"id", "prestige_points", "requirements"}  # <--- dodane "id"
            if not required_keys.issubset(data.keys()):
                raise KeyError(f"Brakujący klucz w danych arystokraty. Wymagane: {required_keys}. Dane: {data}")

            noble = Noble(
                id=data['id'],  # <--- nowy atrybut
                prestige_points=data['prestige_points'],
                requirements={GemColor(k): int(v) for k, v in data.get('requirements', {}).items() if v > 0}
            )
            parsed_nobles.append(noble)
        return parsed_nobles

    def load_definitions_from_json(self, filepath: str) -> Tuple[List[Card], List[Noble]]:
        """
        Wczytuje pełną definicję kart i arystokratów z pliku JSON.

        Args:
            filepath: Ścieżka do pliku JSON.

        Returns:
            Krotka zawierająca (listę wszystkich obiektów Card, listę wszystkich obiektów Noble).
        
        Raises:
            FileNotFoundError: Jeśli plik pod podaną ścieżką nie zostanie znaleziony.
            json.JSONDecodeError: Jeśli plik ma nieprawidłową składnię JSON.
            KeyError: Jeśli plik JSON ma nieprawidłową strukturę (np. brak klucza "cards").
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Plik definicji gry nie został znaleziony pod ścieżką: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if "cards" not in data or "nobles" not in data:
            raise KeyError("Plik JSON musi zawierać główne klucze 'cards' i 'nobles'.")

        all_cards = self._parse_cards(data['cards'])
        all_nobles = self._parse_nobles(data['nobles'])
        
        return all_cards, all_nobles
