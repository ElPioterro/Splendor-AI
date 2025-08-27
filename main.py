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
from game_loader import GameLoader
from game_engine import Game, Move, TakeThreeGems, TakeTwoGems, BuyCard, ReserveVisibleCard, ReserveFromDeck, GemColor, Noble
from agents import HumanAgent, GeneticAgent
from evolution_manager import EvolutionManager
from gui import SplendorGUI

# Nowe importy dla odnogi NN:
from nn_agent import NnAgent
import nn_config as nn_cfg

# Stałe plików mistrzów
CHAMPION_FILE = 'champion_agent.npy'
NN_CHAMPION_FILE = 'nn_champion.npy'  # nowy: mistrz NN

def run_training():
    """
    Uruchamia proces ewolucji w celu wytrenowania najlepszego agenta.
    """
    print("="*50)
    print(" uruchamianie trybu treningu: Ewolucja Agentów ".center(50))
    print("="*50)
    
    print("Wczytywanie definicji gry...")
    try:
        loader = GameLoader()
        all_cards, all_nobles = loader.load_definitions_from_json(config.DATA_FILE)
        print("Definicje gry wczytane pomyślnie.")
    except Exception as e:
        print(f"\n[BŁĄD] Nie udało się wczytać danych gry z pliku '{config.DATA_FILE}': {e}")
        sys.exit(1)
        
    manager = EvolutionManager(all_cards=all_cards, all_nobles=all_nobles)
    
    manager.run_evolution()
    
    final_champion_file = "champion_agent.npy"
    print("\nTrening zakończony.")
    print(f"DNA najlepszego agenta zostało zapisane w pliku: {final_champion_file}")
    print("="*50)


class HumanMoveConstructor:
    """
    Zbiera intencje z GUI i buduje Move.
    """
    def __init__(self):
        self.selected_gems = []

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
            if len(self.selected_gems) == 0:
                self.selected_gems.append(gem_color)
                print(f"[Konstruktor] Wybrane: {self.selected_gems}")
                return None
            if len(self.selected_gems) == 1:
                if self.selected_gems[0] == gem_color:
                    move = TakeTwoGems(color=gem_color)
                    print(f"[Konstruktor] Buduję ruch: {move}")
                    self.reset()
                    return move
                self.selected_gems.append(gem_color)
                print(f"[Konstruktor] Wybrane: {self.selected_gems}")
                return None
            if len(self.selected_gems) == 2:
                a, b = self.selected_gems
                if a != b:
                    if gem_color in (a, b):
                        print("[Konstruktor] Trzeci kolor powtórzony — ignoruję klik.")
                        return None
                    colors = tuple(sorted((a, b, gem_color), key=lambda x: x.value))
                    move = TakeThreeGems(colors=colors)
                    print(f"[Konstruktor] Buduję ruch: {move}")
                    self.reset()
                    return move
                else:
                    print("[Konstruktor] Niespójny stan (dwa takie same bez budowy) — ignoruję.")
                    return None
        return None

def choose_noble_deterministic(eligible: list[Noble]) -> Optional[Noble]:
    """
    Deterministyczny wybór arystokraty: max prestiż, tie-break po id.
    """
    if not eligible:
        return None
    return sorted(eligible, key=lambda n: (-n.prestige_points, n.id))[0]

def run_game():
    """
    Uruchamia główną pętlę gry w trybie Człowiek vs AI z interfejsem graficznym.
    """
    print("="*50)
    print(" Uruchamianie trybu gry: Splendor AI ".center(50))
    print("="*50)
    
    ai_agent = None; human_player = None; game = None; players = []

    print(f"Wczytywanie mistrza AI z pliku '{CHAMPION_FILE}'...")
    if not os.path.exists(CHAMPION_FILE):
        print("\n[BŁĄD] Plik mistrza nie został znaleziony! Uruchom --train.")
        sys.exit(1)
    try:
        champion_dna = np.load(CHAMPION_FILE)
        ai_agent = GeneticAgent(dna=champion_dna)
        ai_agent.name = "Mistrz AI"
        print("Wczytywanie mistrza AI zakończone sukcesem.")
    except Exception as e:
        print(f"\n[BŁĄD] Wystąpił błąd podczas wczytywania pliku '{CHAMPION_FILE}': {e}")
        sys.exit(1)

    print("Inicjalizowanie komponentów gry...")
    try:
        loader = GameLoader()
        all_cards, all_nobles = loader.load_definitions_from_json(config.DATA_FILE)
        human_player = HumanAgent(); human_player.name = "Gracz"
        players = [human_player, ai_agent]
        player_names = [p.name for p in players]
        game = Game(all_cards=all_cards, all_nobles=all_nobles)
        game.setup_new_game(player_names=player_names)
        print("Inicjalizacja zakończona.")
    except Exception as e:
        print(f"\n[BŁĄD] Nie udało się zainicjować gry: {e}")
        sys.exit(1)

    gui = SplendorGUI(); move_constructor = HumanMoveConstructor()
    status_message = ""; human_player_index = 0; game_mode = "PLAY"
    eligible_nobles_pending = []; event_log = []; MAX_LOG = 9

    def log(msg: str):
        nonlocal event_log
        event_log.append(msg)
        if len(event_log) > MAX_LOG:
            event_log = event_log[-MAX_LOG:]

    running = True
    while running and not game.is_game_over():
        gui.draw_game_state(gs=game.game_state, human_player_index=human_player_index, selected_gems=move_constructor.selected_gems,
                            status_message=status_message, mode=game_mode, eligible_nobles=eligible_nobles_pending, log_lines=event_log)
        if status_message:
            status_message = ""

        current_player_agent = players[game.game_state.current_player_index]

        # === POCZĄTEK PATCHA ===
        if game_mode != "CHOOSE_NOBLE":
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                log(f"{players[game.game_state.current_player_index].name} nie ma dostępnych ruchów – tura pominięta.")
                print(f"{players[game.game_state.current_player_index].name} nie ma ruchów, tura pominięta.")
                game.finalize_turn(None)
                pygame.time.wait(800)
                continue
        # === KONIEC PATCHA ===

        if game_mode == "CHOOSE_NOBLE":
            intent = gui.handle_events(game.game_state, mode=game_mode, human_player_index=human_player_index)
            if not intent: continue
            if intent[0] == "QUIT": running = False; continue
            if intent[0] == "choose_noble":
                chosen_noble = intent[1]
                game.finalize_turn(chosen_noble)
                if chosen_noble: log(f"{human_player.name} wybrał arystokratę #{chosen_noble.id} (+{chosen_noble.prestige_points})")
                game_mode = "PLAY"; eligible_nobles_pending = []
            continue

        if current_player_agent is human_player:
            intent = gui.handle_events(game.game_state, mode=game_mode, human_player_index=human_player_index)
            if not intent: continue
            if intent[0] == "QUIT": running = False; continue

            move = move_constructor.process_intent(intent)
            if move:
                try:
                    if move not in valid_moves:
                        raise ValueError("Ruch niedozwolony!")
                    
                    # Logika efektów wizualnych
                    prev_bank = dict(game.game_state.available_gems); prev_gold = game.game_state.gold_gems
                    slot_info = None
                    if isinstance(move, BuyCard):
                        row = game.game_state.visible_cards.get(move.card.level, [])
                        if move.card in row: slot_info = (move.card.level, row.index(move.card))
                    
                    eligible_nobles = game.apply_move(move)
                    
                    # FX
                    new_bank = game.game_state.available_gems
                    deltas = {c: new_bank[c] - prev_bank.get(c, 0) for c in GemColor if new_bank[c] != prev_bank.get(c, 0)}
                    gold_delta = game.game_state.gold_gems - prev_gold
                    if gold_delta: deltas['gold'] = gold_delta
                    if deltas: gui.push_token_deltas(deltas)
                    if isinstance(move, BuyCard) and slot_info: gui.flash_card_slot(*slot_info)

                    if eligible_nobles:
                        game_mode = "CHOOSE_NOBLE"; eligible_nobles_pending = eligible_nobles
                        log(f"{human_player.name} wykonał ruch: {move}. Wybierz arystokratę.")
                    else:
                        game.finalize_turn(None)
                        log(f"{human_player.name} wykonał ruch: {move}")

                except Exception as e:
                    status_message = str(e); print(f"[BŁĄD GŁÓWNEJ PĘTLI] {e}"); move_constructor.reset()
        else:
            # TURA AI
            pygame.time.wait(800)
            move, chosen_noble = ai_agent.choose_action(game.game_state, valid_moves)
            log(f"{ai_agent.name} wykonał ruch: {move}")
            
            # FX
            prev_bank = dict(game.game_state.available_gems); prev_gold = game.game_state.gold_gems
            slot_info = None
            if isinstance(move, BuyCard):
                row = game.game_state.visible_cards.get(move.card.level, [])
                if move.card in row: slot_info = (move.card.level, row.index(move.card))
            
            game.apply_move(move); game.finalize_turn(chosen_noble)
            
            # FX
            new_bank = game.game_state.available_gems
            deltas = {c: new_bank[c] - prev_bank.get(c, 0) for c in GemColor if new_bank[c] != prev_bank.get(c, 0)}
            gold_delta = game.game_state.gold_gems - prev_gold
            if gold_delta: deltas['gold'] = gold_delta
            if deltas: gui.push_token_deltas(deltas)
            if isinstance(move, BuyCard) and slot_info: gui.flash_card_slot(*slot_info)

            if chosen_noble: log(f"{ai_agent.name} wybrał arystokratę #{chosen_noble.id} (+{chosen_noble.prestige_points})")
            pygame.time.wait(400)

    if game.is_game_over():
        winner = game.get_winner()
        log(f"Zwycięzca: {winner.name} ({winner.prestige_points} pkt)" if winner else "Koniec gry!")
        gui.draw_game_state(gs=game.game_state, human_player_index=human_player_index, selected_gems=[],
                            status_message=f"Koniec gry! Wygrywa {winner.name}!" if winner else "Koniec gry!",
                            mode="GAME_OVER", eligible_nobles=[], log_lines=event_log)
        pygame.time.wait(5000)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
    
    pygame.quit(); sys.exit()

def run_game_nn():
    """
    Uruchamia grę Człowiek vs NnAgent (nn_champion.npy) w GUI.
    """
    print("="*50); print(" Uruchamianie trybu gry: Splendor NN ".center(50)); print("="*50)

    print(f"Wczytywanie mistrza NN z pliku '{NN_CHAMPION_FILE}'...")
    if not os.path.exists(NN_CHAMPION_FILE):
        print("\n[BŁĄD] Plik NN mistrza nie został znaleziony! Uruchom train_nn.py.")
        sys.exit(1)
    try:
        theta = np.load(NN_CHAMPION_FILE)
        ai_agent = NnAgent(theta=theta, hidden1=nn_cfg.HIDDEN1, hidden2=nn_cfg.HIDDEN2, act=nn_cfg.ACT, rng_seed=nn_cfg.SEED_BASE)
        ai_agent.name = "NN Agent"
        print("Wczytywanie mistrza NN zakończone sukcesem.")
    except Exception as e:
        print(f"\n[BŁĄD] Wystąpił błąd podczas wczytywania pliku '{NN_CHAMPION_FILE}': {e}")
        sys.exit(1)

    print("Inicjalizowanie komponentów gry...")
    try:
        loader = GameLoader(); all_cards, all_nobles = loader.load_definitions_from_json(config.DATA_FILE)
        human_player = HumanAgent(); human_player.name = "Gracz"
        players = [human_player, ai_agent]
        player_names = [p.name for p in players]
        game = Game(all_cards=all_cards, all_nobles=all_nobles)
        game.setup_new_game(player_names=player_names)
        print("Inicjalizacja zakończona.")
    except Exception as e:
        print(f"\n[BŁĄD] Nie udało się zainicjować gry: {e}")
        sys.exit(1)

    gui = SplendorGUI(); move_constructor = HumanMoveConstructor()
    status_message = ""; human_player_index = 0; game_mode = "PLAY"
    eligible_nobles_pending = []; event_log = []; MAX_LOG = 9

    def log(msg: str):
        nonlocal event_log
        event_log.append(msg)
        if len(event_log) > MAX_LOG:
            event_log = event_log[-MAX_LOG:]

    running = True
    while running and not game.is_game_over():
        gui.draw_game_state(gs=game.game_state, human_player_index=human_player_index, selected_gems=move_constructor.selected_gems,
                            status_message=status_message, mode=game_mode, eligible_nobles=eligible_nobles_pending, log_lines=event_log)
        if status_message: status_message = ""
        
        current_player_agent = players[game.game_state.current_player_index]

        # === POCZĄTEK PATCHA ===
        if game_mode != "CHOOSE_NOBLE":
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                log(f"{players[game.game_state.current_player_index].name} nie ma dostępnych ruchów – tura pominięta.")
                print(f"{players[game.game_state.current_player_index].name} nie ma ruchów, tura pominięta.")
                game.finalize_turn(None)
                pygame.time.wait(800)
                continue
        # === KONIEC PATCHA ===

        if game_mode == "CHOOSE_NOBLE":
            intent = gui.handle_events(game.game_state, mode=game_mode, human_player_index=human_player_index)
            if not intent: continue
            if intent[0] == "QUIT": running = False; continue
            if intent[0] == "choose_noble":
                chosen_noble = intent[1]
                game.finalize_turn(chosen_noble)
                if chosen_noble: log(f"{human_player.name} wybrał arystokratę #{chosen_noble.id} (+{chosen_noble.prestige_points})")
                game_mode = "PLAY"; eligible_nobles_pending = []
            continue

        if current_player_agent is human_player:
            intent = gui.handle_events(game.game_state, mode=game_mode, human_player_index=human_player_index)
            if not intent: continue
            if intent[0] == "QUIT": running = False; continue

            move = move_constructor.process_intent(intent)
            if move:
                try:
                    if move not in valid_moves:
                        raise ValueError("Ruch niedozwolony!")
                    
                    prev_bank = dict(game.game_state.available_gems); prev_gold = game.game_state.gold_gems
                    slot_info = None
                    if isinstance(move, BuyCard):
                        row = game.game_state.visible_cards.get(move.card.level, [])
                        if move.card in row: slot_info = (move.card.level, row.index(move.card))
                    
                    eligible_nobles = game.apply_move(move)

                    new_bank = game.game_state.available_gems
                    deltas = {c: new_bank[c] - prev_bank.get(c, 0) for c in GemColor if new_bank[c] != prev_bank.get(c, 0)}
                    gold_delta = game.game_state.gold_gems - prev_gold
                    if gold_delta: deltas['gold'] = gold_delta
                    if deltas: gui.push_token_deltas(deltas)
                    if isinstance(move, BuyCard) and slot_info: gui.flash_card_slot(*slot_info)

                    if eligible_nobles:
                        game_mode = "CHOOSE_NOBLE"; eligible_nobles_pending = eligible_nobles
                        log(f"{human_player.name} wykonał ruch: {move}. Wybierz arystokratę.")
                    else:
                        game.finalize_turn(None)
                        log(f"{human_player.name} wykonał ruch: {move}")

                except Exception as e:
                    status_message = str(e); print(f"[BŁĄD GŁÓWNEJ PĘTLI] {e}"); move_constructor.reset()
        else:
            # TURA NN
            pygame.time.wait(800)
            move, _ = ai_agent.choose_action(game.game_state, valid_moves)
            log(f"{ai_agent.name} wykonał ruch: {move}")

            prev_bank = dict(game.game_state.available_gems); prev_gold = game.game_state.gold_gems
            slot_info = None
            if isinstance(move, BuyCard):
                row = game.game_state.visible_cards.get(move.card.level, [])
                if move.card in row: slot_info = (move.card.level, row.index(move.card))
            
            eligible = game.apply_move(move)
            chosen_noble = choose_noble_deterministic(eligible)
            game.finalize_turn(chosen_noble)
            
            new_bank = game.game_state.available_gems
            deltas = {c: new_bank[c] - prev_bank.get(c, 0) for c in GemColor if new_bank[c] != prev_bank.get(c, 0)}
            gold_delta = game.game_state.gold_gems - prev_gold
            if gold_delta: deltas['gold'] = gold_delta
            if deltas: gui.push_token_deltas(deltas)
            if isinstance(move, BuyCard) and slot_info: gui.flash_card_slot(*slot_info)

            if chosen_noble: log(f"{ai_agent.name} wybrał arystokratę #{chosen_noble.id} (+{chosen_noble.prestige_points})")
            pygame.time.wait(400)

    if game.is_game_over():
        winner = game.get_winner()
        log(f"Zwycięzca: {winner.name} ({winner.prestige_points} pkt)" if winner else "Koniec gry!")
        gui.draw_game_state(gs=game.game_state, human_player_index=human_player_index, selected_gems=[],
                            status_message=f"Koniec gry! Wygrywa {winner.name}!" if winner else "Koniec gry!",
                            mode="GAME_OVER", eligible_nobles=[], log_lines=event_log)
        pygame.time.wait(5000)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
    
    pygame.quit(); sys.exit()

def main():
    parser = argparse.ArgumentParser(description="Splendor AI - Graj lub trenuj agentów opartych o algorytmy genetyczne.")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help="Uruchom tryb treningu (ewolucji agentów).")
    mode_group.add_argument('--play', action='store_true', help="Uruchom grę przeciwko wytrenowanemu agentowi AI.")
    mode_group.add_argument('--play-nn', action='store_true', help="Uruchom grę Człowiek vs NN (nn_champion.npy).")
    
    args = parser.parse_args()

    if args.train:
        run_training()
    elif args.play:
        run_game()
    elif args.play_nn:
        run_game_nn()

if __name__ == "__main__":
    main()