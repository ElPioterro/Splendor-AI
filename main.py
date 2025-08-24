'''
Moduł 4: main.py (Główny Plik)

    Punkt wejścia do aplikacji.

    Pozwala wybrać tryb:

        --train: Uruchamia evolution_manager do trenowania AI.

        --play: Pozwala zagrać człowiekowi przeciwko AI (ładuje najlepsze zapisane DNA).'''
import argparse
import sys
import os
import numpy as np
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import pygame
from typing import Optional

# --- Importy modułów projektu ---
import config
from game_loader import GameLoader # ZMIANA: Importujemy klasę, nie funkcję
from game_engine import Game, Move, TakeThreeGems, TakeTwoGems, BuyCard, ReserveVisibleCard, ReserveFromDeck, GemColor
from agents import HumanAgent, GeneticAgent
from evolution_manager import EvolutionManager
from gui import SplendorGUI

# Stała definiująca ścieżkę do pliku z DNA mistrza
CHAMPION_FILE = 'champion_agent.npy'

def run_training():
    """
    Uruchamia proces ewolucji w celu wytrenowania najlepszego agenta.
    """
    print("="*50)
    print(" uruchamianie trybu treningu: Ewolucja Agentów ".center(50))
    print("="*50)
    
    # ZMIANA 1: Poprawne wczytywanie danych gry za pomocą GameLoader
    print("Wczytywanie definicji gry...")
    try:
        loader = GameLoader()
        all_cards, all_nobles = loader.load_definitions_from_json(config.DATA_FILE)
        print("Definicje gry wczytane pomyślnie.")
    except Exception as e:
        print(f"\n[BŁĄD] Nie udało się wczytać danych gry z pliku '{config.DATA_FILE}': {e}")
        sys.exit(1)
        
    # ZMIANA 2: Poprawne inicjowanie EvolutionManager
    # Konstruktor oczekuje danych gry, a parametry ewolucji pobiera z config.py
    manager = EvolutionManager(all_cards=all_cards, all_nobles=all_nobles)
    
    # Uruchomienie procesu ewolucji (bez zmian)
    manager.run_evolution()
    
    # Zmieniamy nazwę finalnego pliku, aby była zgodna z nowym evolution_manager.py
    final_champion_file = "champion_agent.npy"
    print("\nTrening zakończony.")
    print(f"DNA najlepszego agenta zostało zapisane w pliku: {final_champion_file}")
    print("="*50)


# NOWA KLASA POMOCNICZA (można ją umieścić tuż przed funkcją run_game)
class HumanMoveConstructor:
    """
    Zbiera intencje z GUI i buduje Move.
    Zasady stanu:
    - reset() wywołuje TYLKO: sam konstruktor po zwróceniu gotowego Move albo 'clear_selection' z GUI,
      oraz pętla główna w przypadku wyjątku z engine.
    - 'Niepełne' kliknięcia nie resetują wyboru.
    """
    def __init__(self):
        self.selected_gems = []  # List[GemColor]

    def reset(self):
        self.selected_gems = []
        print("[Konstruktor] Zresetowano wybór.")

    def process_intent(self, intent: tuple) -> Optional[Move]:
        intent_type = intent[0]

        if intent_type == "clear_selection":
            self.reset()
            return None

        if intent_type == "buy_card":
            card = intent[1]
            self.reset()
            return BuyCard(card=card)

        if intent_type == "reserve_visible":
            card = intent[1]
            self.reset()
            return ReserveVisibleCard(card=card)

        if intent_type == "reserve_deck":
            tier = intent[1]
            self.reset()
            return ReserveFromDeck(tier=tier)

        if intent_type == "select_gem":
            gem_color = intent[1]

            # 1) 0 -> 1
            if len(self.selected_gems) == 0:
                self.selected_gems.append(gem_color)
                print(f"[Konstruktor] Wybrane: {self.selected_gems}")
                return None

            # 2) 1 -> 2
            if len(self.selected_gems) == 1:
                # Dwa takie same = TakeTwo
                if self.selected_gems[0] == gem_color:
                    move = TakeTwoGems(color=gem_color)
                    print(f"[Konstruktor] Buduję ruch: {move}")
                    self.reset()
                    return move
                # Dwa różne = czekamy na trzeci
                self.selected_gems.append(gem_color)
                print(f"[Konstruktor] Wybrane: {self.selected_gems}")
                return None

            # 3) 2 -> 3 (tylko jeśli dwa są różne i trzeci też inny)
            if len(self.selected_gems) == 2:
                a, b = self.selected_gems
                # jeśli mamy już dwa różne, trzeci musi być nowy
                if a != b:
                    if gem_color in (a, b):
                        # ignorujemy klik, nie resetujemy
                        print("[Konstruktor] Trzeci kolor powtórzony — ignoruję klik.")
                        return None
                    colors = tuple(sorted((a, b, gem_color), key=lambda x: x.value))
                    move = TakeThreeGems(colors=colors)
                    print(f"[Konstruktor] Buduję ruch: {move}")
                    self.reset()
                    return move
                else:
                    # teoretycznie nie trafimy tu, bo 2x ten sam kończyło wcześniej
                    # ale na wszelki wypadek: ignoruj
                    print("[Konstruktor] Niespójny stan (dwa takie same bez budowy) — ignoruję.")
                    return None

        return None
        
def run_game():
    """
    Uruchamia główną pętlę gry w trybie Człowiek vs AI z interfejsem graficznym.
    """
    print("="*50)
    print(" Uruchamianie trybu gry: Splendor AI ".center(50))
    print("="*50)
    
    # --- Inicjalizacja zmiennych, aby miały szerszy zasięg ---
    ai_agent = None
    human_player = None
    game = None
    players = []

    # --- Krok 1: Wczytanie agenta AI ---
    print(f"Wczytywanie mistrza AI z pliku '{CHAMPION_FILE}'...")
    if not os.path.exists(CHAMPION_FILE):
        print("\n[BŁĄD] Plik mistrza nie został znaleziony!")
        print("Proszę najpierw uruchomić trening za pomocą komendy:")
        print("  python main.py --train")
        sys.exit(1)

    try:
        champion_dna = np.load(CHAMPION_FILE)
        ai_agent = GeneticAgent(dna=champion_dna)
        ai_agent.name = "Mistrz AI"
        print("Wczytywanie mistrza AI zakończone sukcesem.")
    except Exception as e:
        print(f"\n[BŁĄD] Wystąpił błąd podczas wczytywania pliku '{CHAMPION_FILE}': {e}")
        sys.exit(1)

    # --- Krok 2: Inicjalizacja komponentów gry ---
    print("Inicjalizowanie komponentów gry...")
    try:
        loader = GameLoader()
        all_cards, all_nobles = loader.load_definitions_from_json(config.DATA_FILE)
        
        human_player = HumanAgent()
        human_player.name = "Gracz"
        
        players = [human_player, ai_agent]
        player_names = [p.name for p in players]
        
        game = Game(all_cards=all_cards, all_nobles=all_nobles)
        game.setup_new_game(player_names=player_names)
        
        print("Inicjalizacja zakończona. Gracze w grze:")
        for player in game.game_state.players:
            print(f"- {player.name}")
        print("="*50)
    except Exception as e:
        print(f"\n[BŁĄD] Nie udało się zainicjować gry: {e}")
        sys.exit(1)

    # --- Krok 3: Inicjalizacja GUI i pętli gry ---
    gui = SplendorGUI()
    move_constructor = HumanMoveConstructor()
    status_message = ""
    human_player_index = 0

    # Nowe
    game_mode = "PLAY"  # albo "CHOOSE_NOBLE"
    eligible_nobles_pending = []
    event_log = []  # lista stringów
    MAX_LOG = 9

    def log(msg: str):
        nonlocal event_log
        event_log.append(msg)
        if len(event_log) > MAX_LOG:
            event_log = event_log[-MAX_LOG:]

    running = True
    while running and not game.is_game_over():
        gui.draw_game_state(
            gs=game.game_state,
            human_player_index=human_player_index,
            selected_gems=move_constructor.selected_gems,
            status_message=status_message,
            mode=game_mode,
            eligible_nobles=eligible_nobles_pending,
            log_lines=event_log
        )
        if status_message:
            status_message = ""

        current_player_agent = players[game.game_state.current_player_index]

        # 1) TRYB WYBORU ARYSTOKRATY (tylko gdy tura człowieka)
        if current_player_agent is human_player and game_mode == "CHOOSE_NOBLE":
            intent = gui.handle_events(game.game_state, mode=game_mode, human_player_index=human_player_index)
            if not intent:
                continue
            if intent[0] == "QUIT":
                running = False
                continue
            if intent[0] == "choose_noble":
                chosen_noble = intent[1]  # może być None
                game.finalize_turn(chosen_noble)
                if chosen_noble:
                    log(f"{human_player.name} wybrał arystokratę #{chosen_noble.id} (+{chosen_noble.prestige_points})")
                game_mode = "PLAY"
                eligible_nobles_pending = []
                # NIE resetujemy konstruktora tutaj (happy path)
            continue  # wróć do rysowania po wyborze/klikach

        # 2) NORMALNA TURA
        if current_player_agent is human_player:
            intent = gui.handle_events(game.game_state, mode=game_mode, human_player_index=human_player_index)
            if not intent:
                continue
            if intent[0] == "QUIT":
                running = False
                continue

            move = move_constructor.process_intent(intent)
            if move:
                try:
                    valid_moves = game.get_valid_moves()
                    if move not in valid_moves:
                        msg = "Ruch niedozwolony!"
                        # Doprecyzuj powód dla ruchów pobrania żetonów
                        if isinstance(move, (TakeThreeGems, TakeTwoGems)):
                            added = 3 if isinstance(move, TakeThreeGems) else 2
                            cur_total = game.game_state.players[human_player_index].total_gems
                            if cur_total + added > 10:
                                msg = f"Nie możesz wziąć {added} żetonów: limit 10 (masz teraz {cur_total})."
                            else:
                                # Inne przyczyny: np. za mało żetonów w banku, albo 2 tego samego gdy w banku < 4 itp.
                                msg = "Ruch niedozwolony: sprawdź dostępność żetonów i zasady (3 różne lub 2 tego samego przy ≥4 w banku)."
                        raise ValueError(msg)

                    # Snapshot banku i ewentualnego slotu karty (przed apply_move)
                    prev_bank = dict(game.game_state.available_gems)
                    prev_gold = game.game_state.gold_gems

                    slot_info = None
                    if isinstance(move, BuyCard):
                        card = move.card
                        # jeśli karta pochodzi ze stołu (nie z rezerwy), złap jej slot
                        row = game.game_state.visible_cards.get(card.level, [])
                        if card in row:
                            slot_info = (card.level, row.index(card))

                    eligible_nobles = game.apply_move(move)

                    # Po udanym apply_move: delty banku
                    new_bank = game.game_state.available_gems
                    deltas = {c: new_bank[c] - prev_bank.get(c, 0) for c in GemColor if new_bank[c] != prev_bank.get(c, 0)}
                    gold_delta = game.game_state.gold_gems - prev_gold
                    if gold_delta:
                        deltas['gold'] = gold_delta
                    if deltas:
                        gui.push_token_deltas(deltas)

                    # Flash slotu (jeśli kupiono ze stołu)
                    if isinstance(move, BuyCard) and slot_info:
                        tier, idx = slot_info
                        gui.flash_card_slot(tier, idx)

                    # Arystokrata / finalize_turn bez resetu konstruktora (on już to zrobił)
                    if eligible_nobles:
                        game_mode = "CHOOSE_NOBLE"
                        eligible_nobles_pending = eligible_nobles
                        log(f"{human_player.name} wykonał ruch: {move}. Wybierz arystokratę.")
                    else:
                        game.finalize_turn(None)
                        log(f"{human_player.name} wykonał ruch: {move}")

                except Exception as e:
                    status_message = str(e)
                    print(f"[BŁĄD GŁÓWNEJ PĘTLI] {e}")
                    move_constructor.reset()  # reset TYLKO na ścieżce błędu

        else:
            # TURA AI
            pygame.time.wait(800)
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                game.finalize_turn(None)
                continue
            move, chosen_noble = ai_agent.choose_action(game.game_state, valid_moves)
            status_message = f"{ai_agent.name} wykonał ruch: {move}"
            print(status_message)

            prev_bank = dict(game.game_state.available_gems)
            prev_gold = game.game_state.gold_gems

            slot_info = None
            if isinstance(move, BuyCard):
                row = game.game_state.visible_cards.get(move.card.level, [])
                if move.card in row:
                    slot_info = (move.card.level, row.index(move.card))

            game.apply_move(move)
            game.finalize_turn(chosen_noble)

            # FX
            new_bank = game.game_state.available_gems
            deltas = {c: new_bank[c] - prev_bank.get(c, 0) for c in GemColor if new_bank[c] != prev_bank.get(c, 0)}
            gold_delta = game.game_state.gold_gems - prev_gold
            if gold_delta:
                deltas['gold'] = gold_delta
            if deltas:
                gui.push_token_deltas(deltas)
            if isinstance(move, BuyCard) and slot_info:
                gui.flash_card_slot(*slot_info)

            if chosen_noble:
                log(f"{ai_agent.name} wybrał arystokratę #{chosen_noble.id} (+{chosen_noble.prestige_points})")
            else:
                log(f"{ai_agent.name} wykonał ruch: {move}")
            pygame.time.wait(400)

    # --- Koniec gry ---
    if game.is_game_over():
        print("\nKoniec gry!")
        winner = game.get_winner()
        if winner:
            log(f"Zwycięzca: {winner.name} ({winner.prestige_points} pkt)")
            print(f"Zwycięzcą jest: {winner.name} z {winner.prestige_points} punktami!")
        
        # ZMIANA TUTAJ: Uzupełniamy brakujące argumenty
        gui.draw_game_state(
            gs=game.game_state,
            human_player_index=human_player_index,
            selected_gems=[], # Na ekranie końcowym nie ma już wyboru
            status_message=f"Koniec gry! Wygrywa {winner.name}!" if winner else "Koniec gry!",
            mode="GAME_OVER", # Możemy dodać specjalny tryb dla ekranu końcowego
            eligible_nobles=[],
            log_lines=event_log
        )
        pygame.time.wait(5000) # Wydłużamy czas, aby gracz mógł przeczytać wynik

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    
    pygame.quit()
    sys.exit()

def main():
    """
    Główny punkt wejścia aplikacji. Parsuje argumenty i uruchamia odpowiedni tryb.
    """
    parser = argparse.ArgumentParser(
        description="Splendor AI - Graj lub trenuj agentów opartych o algorytmy genetyczne."
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--train', 
        action='store_true', 
        help="Uruchom tryb treningu (ewolucji agentów)."
    )
    mode_group.add_argument(
        '--play', 
        action='store_true', 
        help="Uruchom grę przeciwko wytrenowanemu agentowi AI."
    )
    
    args = parser.parse_args()

    if args.train:
        run_training()
    elif args.play:
        run_game()

if __name__ == "__main__":
    main()