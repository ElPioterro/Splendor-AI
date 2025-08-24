**Splendor AI**

1.  **Cel Główny:** Stworzenie AI do gry w Splendor przy użyciu Algorytmu Genetycznego. AI będzie trenowane poprzez symulację wielu gier przeciwko sobie. Finalnym celem jest możliwość gry człowiek vs AI.

2.  **Stos Technologiczny:**

    - Język: Python 3.10+
    - Biblioteki: NumPy (do operacji na DNA), Multiprocessing (do równoległego treningu), `tqdm` (opcjonalnie, do pasków postępu). Na razie nie używamy żadnych zewnętrznych frameworków.
    - Styl kodu: Nowoczesny Python z użyciem type hints (np. `def play(self, game_state: GameState) -> Move:`) i docstringów.

3.  **Architektura Modułowa:** Projekt będzie składał się z następujących, oddzielnych plików:

    - `game_engine.py`: Zawiera całą logikę i stan gry Splendor. Jest całkowicie niezależny od AI. Definiuje klasy takie jak `GameState`, `Card`, `Player`, `Noble`.
    - `agents.py`: Definiuje interfejs gracza oraz jego konkretne implementacje: `HumanPlayer`, `RandomAgent`, `GeneticAgent`.
    - `evolution_manager.py`: Odpowiada za orkiestrację procesu ewolucji: tworzenie populacji, prowadzenie turniejów, selekcję, krzyżowanie i mutację.
    - `main.py`: Punkt wejścia aplikacji, obsługuje argumenty linii poleceń (`--train`, `--play`).
    - `config.py` (opcjonalnie): Przechowuje stałe, np. parametry algorytmu genetycznego.

4.  **Kluczowe Struktury Danych i API:**
    - `GameState`: Reprezentuje pełny, aktualny stan gry (dostępne klejnoty, karty na stole, stan graczy).
    - Każdy agent w `agents.py` musi implementować metodę `choose_move(self, game_state: GameState, valid_moves: list) -> Move`.
    - Silnik gry w `game_engine.py` musi udostępniać dwie kluczowe metody: `get_valid_moves(player)` oraz `apply_move(move)`.
