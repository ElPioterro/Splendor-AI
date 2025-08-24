import pygame
from typing import Optional

# Importujemy komponenty gry, aby GUI wiedziało, co rysować
from game_engine import GameState, Move, Card, GemColor
# Na przyszłość, gdy będziemy tworzyć obiekty Move
from game_engine import TakeThreeGems, BuyCard 

# --- Ustawienia i Stałe Wizualne ---
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 800
BACKGROUND_COLOR = (20, 40, 60) # Ciemny granat
FONT_COLOR = (220, 220, 220)

# Słownik kolorów dla klejnotów/kart
GEM_COLORS_RGB = {
    GemColor.BLUE: (0, 100, 255),
    GemColor.GREEN: (0, 150, 50),
    GemColor.RED: (200, 0, 0),
    GemColor.WHITE: (230, 230, 230),
    GemColor.BLACK: (30, 30, 30),
    'gold': (255, 215, 0)
}

class SplendorGUI:
    """
    Klasa odpowiedzialna wyłącznie za prezentację wizualną gry Splendor.
    Nie zawiera żadnej logiki gry.
    """
    def __init__(self):
        pygame.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Splendor AI")
        
        self.font_small = pygame.font.SysFont('Arial', 18)
        self.font_medium = pygame.font.SysFont('Arial', 24)
        
        print("GUI zainicjalizowane.")

    def draw_game_state(self, game_state: GameState):
        """
        Główna metoda rysująca. Bierze stan gry i renderuje go na ekranie.
        """
        # 1. Rysowanie tła
        self.screen.fill(BACKGROUND_COLOR)

        # 2. Rysowanie planszy (miejsce na karty, żetony, etc.) - na razie proste prostokąty
        # Strefa kart
        pygame.draw.rect(self.screen, (40, 60, 80), (100, 50, 800, 600))
        # Strefa gracza
        pygame.draw.rect(self.screen, (40, 60, 80), (50, 670, 1180, 120))
        # Strefa żetonów
        pygame.draw.rect(self.screen, (40, 60, 80), (950, 50, 280, 400))
        
        # Przykład wyświetlania tekstu
        text_surface = self.font_medium.render("Witaj w Splendor AI!", True, FONT_COLOR)
        self.screen.blit(text_surface, (SCREEN_WIDTH // 2 - text_surface.get_width() // 2, 10))
        
        # 3. Odświeżenie ekranu
        pygame.display.flip()

    def handle_events(self) -> Optional[Move]:
        """
        Przechwytuje i obsługuje zdarzenia (np. kliknięcia).
        Jeśli gracz wykonał akcję, która tworzy ruch, ta metoda go zwróci.
        W przeciwnym razie zwraca None.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return "QUIT" # Specjalny sygnał do zamknięcia gry

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                print(f"Kliknięto w pozycję: {mouse_pos}")
                
                # --- TUTAJ BĘDZIE LOGIKA TŁUMACZENIA KLIKNIĘĆ NA RUCHY ---
                # Na przykład:
                # if card_rect.collidepoint(mouse_pos):
                #     return BuyCard(card=card_object)

        return None # W tej klatce nie wygenerowano żadnego ruchu