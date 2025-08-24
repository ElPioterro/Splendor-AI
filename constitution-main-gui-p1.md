### **Konstytucja Projektu Splendor AI (Wersja 3.0 - Finalna)**

**Preambuła:** Po pomyślnym zakończeniu implementacji modułu `evolution_manager.py`, który dostarczył nam zdolność do tworzenia inteligentnych agentów, projekt wkracza w finalną fazę integracji i tworzenia interfejsu użytkownika. Niniejsza wersja konstytucji kodyfikuje rolę wszystkich ukończonych modułów i ustanawia niewzruszone zasady dla implementacji warstwy aplikacji (`main.py`) i prezentacji (`gui.py`), aby zapewnić spójny i solidny produkt końcowy.

**Artykuł I: Niezmienna Struktura Modułowa (Potwierdzony)**
Struktura projektu jest kompletna. Obowiązki każdego modułu są ostateczne:

- `data/splendor_data.json`: Źródło prawdy o komponentach gry.
- `game_loader.py`: Kustosz Danych, tłumacz z danych na obiekty.
- `game_engine.py`: Serce Logiki Gry, bezstronny sędzia zasad.
- `agents.py`: Mózgi Graczy, repozytorium strategii decyzyjnych.
- `evolution_manager.py`: Fabryka Inteligencji, narzędzie do treningu offline.
- `gui.py`: Okno na Świat Gry, warstwa interakcji z człowiekiem.
- `main.py`: Główny Dyrygent, punkt wejścia i spoiwo aplikacji.
- `config.py`: Centrum Ustawień, repozytorium parametrów.

**Artykuł II: Kluczowe Zasady Architektoniczne (Potwierdzone)**
Wszystkie zaimplementowane moduły i te, które powstaną, muszą przestrzegać następujących zasad:

1.  **Zasada Delegacji Decyzji (Dot. `game_engine.py`):** Obowiązuje dwuetapowy proces finalizacji tury (`apply_move` -> `finalize_turn`), delegujący pełną odpowiedzialność strategiczną do agenta.
2.  **Zasada Kompletnej Akcji (Dot. `agents.py`):** Każdy agent poprzez metodę `choose_action` musi zwracać kompletną, gotową do wykonania decyzję na daną turę.

**Artykuł III: Standard DNA Agenta Genetycznego (Potwierdzony)**
Oficjalny standard DNA (`Dodatek Techniczny A`, 16 genów) pozostaje wiążący. Jest to fundamentalny kontrakt między `agents.py` a `evolution_manager.py`.

**Artykuł IV: Rola i Artefakty Fabryki Inteligencji (Nowy Artykuł)**
Niniejszym formalnie definiuje się rolę ukończonego modułu `evolution_manager.py`:

1.  **Tryb Działania:** Jest to narzędzie działające wyłącznie w trybie "offline" (`--train`). Nie jest częścią interaktywnej pętli gry.
2.  **Cel:** Jego jedynym celem jest przeprowadzenie symulacji ewolucyjnej i wyprodukowanie zoptymalizowanego wektora DNA.
3.  **Artefakt Wyjściowy:** Jedynym produktem i "publicznym API" modułu po zakończeniu pracy jest plik binarny (`champion_agent.npy`), który zawiera DNA najlepszego wytrenowanego agenta.

**Artykuł V: Architektura Warstwy Aplikacji i Prezentacji (Nowy Artykuł)**
Ustanawia się następujące, nadrzędne zasady dla finalnych modułów `main.py` i `gui.py`:

1.  **Rola Głównego Dyrygenta (`main.py`):**

    - **Jedyny Punkt Wejścia:** `main.py` jest jedynym plikiem wykonywalnym aplikacji.
    - **Przełącznik Trybów:** Jest odpowiedzialny za parsowanie argumentów linii poleceń (np. `--train`, `--play`) i uruchamianie odpowiedniej logiki – albo `EvolutionManager`, albo głównej pętli gry.
    - **Inicjalizator Systemu:** W trybie gry (`--play`), `main.py` jest odpowiedzialny za zainicjowanie wszystkich niezbędnych komponentów: wczytanie danych gry (`game_loader`), stworzenie instancji silnika (`Game`), stworzenie graczy (`HumanAgent`) oraz wczytanie artefaktu (`.npy`) i stworzenie z niego `GeneticAgent`.

2.  **Zasada Czystej Prezentacji ("Głupi Widok") dla `gui.py`:**
    - **Absolutna Separacja Logiki od Widoku:** Moduł `gui.py` jest odpowiedzialny **wyłącznie** za prezentację wizualną i przechwytywanie intencji użytkownika.
    - **Zakaz Logiki Gry:** W `gui.py` **nie może** znajdować się żadna logika gry. Nie może on sprawdzać, czy ruch jest dozwolony, obliczać kosztów kart ani zarządzać stanem gry.
    - **Mechanizm Działania:**
      a) Otrzymuje obiekt `GameState` od głównej pętli gry.
      b) Renderuje stan gry na ekranie na podstawie otrzymanego obiektu.
      c) Przechwytuje akcje użytkownika (np. kliknięcia myszką).
      d) Tłumaczy te akcje na obiekty `Move` (np. `BuyCard(card=...)`).
      e) Zwraca stworzony obiekt `Move` do głównej pętli gry, która przekaże go do `game_engine` w celu walidacji i wykonania.
