### **Konstytucja Projektu PROMETHEUS (Wersja 4.0 - Finalna, Skonsolidowana)**

**Preambuła:**
Po pomyślnym ukończeniu implementacji wszystkich kluczowych modułów, od silnika gry po interfejs graficzny, projekt PROMETHEUS osiągnął status w pełni funkcjonalnego i grywalnego produktu. Niniejsza, finalna i skonsolidowana wersja konstytucji służy jako ostateczne źródło prawdy o architekturze projektu. Kodyfikuje ona wszystkie kluczowe zasady, standardy i decyzje projektowe, które doprowadziły projekt do sukcesu i które muszą być przestrzegane we wszelkich przyszłych pracach rozwojowych i konserwacyjnych.

**Artykuł I: Niezmienna Struktura Modułowa**
Struktura projektu jest kompletna. Obowiązki każdego modułu są ostateczne:

- `data/splendor_data.json`: Źródło prawdy o komponentach gry.
- `game_loader.py`: Kustosz Danych, tłumacz z danych na obiekty.
- `game_engine.py`: Serce Logiki Gry, bezstronny sędzia zasad.
- `agents.py`: Mózgi Graczy, repozytorium strategii decyzyjnych.
- `evolution_manager.py`: Fabryka Inteligencji, narzędzie do treningu offline.
- `gui.py`: Okno na Świat Gry, warstwa interakcji z człowiekiem.
- `main.py`: Główny Dyrygent, punkt wejścia i spoiwo aplikacji.
- `config.py`: Centrum Ustawień, repozytorium parametrów.

**Artykuł II: Kluczowe Zasady Architektoniczne**
Wszystkie moduły muszą przestrzegać następujących zasad:

1.  **Zasada Delegacji Decyzji:** Obowiązuje dwuetapowy proces finalizacji tury w `game_engine.py` (`apply_move` -> `finalize_turn`), delegujący pełną odpowiedzialność strategiczną do agenta.
2.  **Zasada Kompletnej Akcji:** Każdy agent (`agents.py`) poprzez metodę `choose_action` musi zwracać kompletną, gotową do wykonania decyzję na daną turę.

**Artykuł III: Standard DNA Agenta Genetycznego**
Ustanawia się oficjalny standard struktury genetycznej dla `GeneticAgent` (zgodnie z "Dodatkiem Technicznym A", 16 genów). Jest to wiążący kontrakt między `agents.py` a `evolution_manager.py`.

**Artykuł IV: Rola i Artefakty Fabryki Inteligencji**
Rola `evolution_manager.py` jest ściśle zdefiniowana:

1.  **Tryb Działania:** Jest to narzędzie działające wyłącznie w trybie "offline" (`--train`).
2.  **Cel:** Jego jedynym celem jest przeprowadzenie symulacji ewolucyjnej.
3.  **Artefakt Wyjściowy:** Jedynym produktem jego pracy jest plik binarny (`champion_agent.npy`) zawierający DNA najlepszego agenta.

**Artykuł V: Architektura Warstwy Aplikacji i Prezentacji**
Ustanawia się następujące zasady dla `main.py` i `gui.py`:

1.  **Rola Głównego Dyrygenta (`main.py`):** Jest jedynym punktem wejścia, przełącznikiem trybów (`--train`, `--play`) oraz inicjalizatorem wszystkich komponentów systemu w trybie gry.
2.  **Zasada Czystej Prezentacji (`gui.py`):** Moduł GUI jest odpowiedzialny wyłącznie za prezentację wizualną i przechwytywanie intencji użytkownika. Nie może zawierać żadnej logiki gry.

**Artykuł VI: Dziennik Pokładowy Ewolucji**
Ustanawia się mechanizm logowania szczegółowych statystyk każdej generacji do ustrukturyzowanego formatu `CSV` (`logs/`) w celu późniejszej analizy. Odpowiedzialność za to spoczywa na `evolution_manager.py`.

**Artykuł VII: Architektura Intencji Użytkownika**
Formalizuje się architekturę interakcji między `gui.py` a `main.py`:

1.  **Zasada Komunikacji Jednokierunkowej:** `main.py` przekazuje do `gui.py` pełny stan do narysowania. `gui.py` zwraca do `main.py` wyłącznie proste, zrozumiałe dla człowieka **"intencje"** (np. krotka `('buy_card', card_object)`), a **nigdy** gotowe obiekty `Move`.
2.  **Rola Konstruktora Ruchu:** Obowiązek przetłumaczenia sekwencji intencji na kompletny i walidowalny obiekt `Move` spoczywa w całości na `main.py` (np. w dedykowanej do tego klasie `HumanMoveConstructor`).
3.  **Walidacja Ruchu:** Ostateczna i jedyna weryfikacja poprawności ruchu należy do `game_engine.py`, wywoływanego z poziomu `main.py`.
