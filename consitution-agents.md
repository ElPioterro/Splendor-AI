**Splendor AI**

1.  **Cel Główny:** Stworzenie AI do gry w Splendor przy użyciu Algorytmu Genetycznego. AI będzie trenowane poprzez symulację wielu gier przeciwko sobie. Finalnym celem jest możliwość gry człowiek vs AI.

2.  **Stos Technologiczny:**

    - Język: Python 3.10+
    - Biblioteki: NumPy (do operacji na DNA), Multiprocessing (do równoległego treningu), `tqdm` (opcjonalnie, do pasków postępu). Na razie nie używamy żadnych zewnętrznych frameworków.
    - Styl kodu: Nowoczesny Python z użyciem type hints (np. `def play(self, game_state: GameState) -> Move:`) i docstringów.

3.  **Architektura Modułowa:** Projekt będzie składał się z następujących, oddzielnych plików:

- `data/splendor_data.json`: **Źródło Prawdy.** Plik z danymi definiującymi wszystkie karty i arystokratów w grze.

- `game_loader.py`: **Kustosz Danych.** Odpowiada za wczytanie danych z pliku `json`, przetworzenie ich i stworzenie z nich obiektów (`Card`, `Noble`) gotowych do użycia przez silnik gry.

- `game_engine.py`: **Serce i Rdzeń Logiki Gry.** Definiuje wszystkie zasady i stan gry. Zawiera klasy `GameState`, `Player`, `Card`, `Noble` i `Move`. Jest całkowicie "ślepy" na to, kto gra (człowiek czy AI) i jak gra jest wyświetlana. Jego jedyne zadania to zarządzanie stanem, walidacja ruchów (`get_valid_moves`) i aplikowanie ich (`apply_move`).

- `agents.py`: **Mózgi Graczy.** Definiuje abstrakcyjny interfejs dla gracza (`Agent`) z metodą `choose_move`. Zawiera jego konkretne implementacje:

  - `HumanAgent`: Pobiera ruch od człowieka (przez konsolę lub GUI).
  - `RandomAgent`: Wykonuje losowe, ale dozwolone ruchy. Idealny jako punkt odniesienia.
  - `GeneticAgent`: Nasz główny cel. Używa swojej sieci wag (DNA) do oceny stanu gry i wyboru najlepszego ruchu.

- `evolution_manager.py`: **Wielki Architekt Ewolucji.** Zarządza całym procesem uczenia maszynowego "offline":

  - Tworzy początkową populację `GeneticAgent`.
  - Organizuje turnieje (symuluje tysiące gier między agentami).
  - Ocenia agentów (fitness score).
  - Przeprowadza selekcję, krzyżowanie i mutację, tworząc nowe, lepsze pokolenia.
  - Zapisuje najlepsze "mózgi" (DNA) do pliku.

- `gui.py`: **Okno na Świat Gry.** Odpowiada za wizualną reprezentację `GameState`. Renderuje planszę, karty, żetony. Przechwytuje akcje gracza (kliknięcia) i tłumaczy je na obiekty `Move`, które przekazuje do silnika. Nie zawiera żadnej logiki gry.

- `main.py`: **Główny Dyrygent.** Punkt wejścia aplikacji, który spina wszystko razem. Parsuje argumenty linii poleceń, aby zdecydować, co robić:

  - `--train`: Uruchamia `evolution_manager`.
  - `--play-gui --agent path/to/best_agent.dna`: Uruchamia `gui.py`, tworzy grę między człowiekiem a wczytanym, najlepszym `GeneticAgent`.
  - `--play-console`: Uruchamia grę w trybie tekstowym.

- `config.py`: (Opcjonalnie) **Centrum Ustawień.** Przechowuje globalne stałe i parametry, np. populacja, współczynnik mutacji, ścieżki do plików, ustawienia okna GUI.

4.  **Kluczowe Struktury Danych i API:**
    - `GameState`: Reprezentuje pełny, aktualny stan gry (dostępne klejnoty, karty na stole, stan graczy).
    - Każdy agent w `agents.py` musi implementować metodę `choose_move(self, game_state: GameState, valid_moves: list) -> Move`.
    - Silnik gry w `game_engine.py` musi udostępniać dwie kluczowe metody: `get_valid_moves(player)` oraz `apply_move(move)`.
