### **Konstytucja Projektu Splendor AI (Wersja 2.0 - Zaktualizowana)**

**Preambuła:** Po pomyślnym zakończeniu implementacji modułów `game_engine.py` i `agents.py`, niniejszym uroczyście aktualizujemy konstytucję naszego projektu. Celem tej aktualizacji jest skodyfikowanie kluczowych decyzji architektonicznych i standardów technicznych, które zapewnią spójność, modularność i skalowalność projektu w dalszych fazach rozwoju.

**Artykuł I: Niezmienna Struktura Modułowa**

Struktura projektu pozostaje nienaruszona i jest fundamentem naszej pracy. Każdy moduł ma jasno określoną i rozłączną odpowiedzialność:

- `data/splendor_data.json`: Definicje danych gry.
- `game_loader.py`: Wczytywanie i tworzenie obiektów gry z danych.
- `game_engine.py`: **Serce Logiki Gry.** Strażnik zasad, całkowicie agnostyczny wobec graczy i interfejsu.
- `agents.py`: **Mózgi Graczy.** Implementacje strategii decyzyjnych.
- `evolution_manager.py`: **Fabryka Inteligencji.** Orkiestrator procesu treningu genetycznego.
- `gui.py`: **Okno na Świat Gry.** Warstwa wizualna i interakcji z człowiekiem.
- `main.py`: **Główny Dyrygent.** Punkt wejścia i spoiwo aplikacji.
- `config.py`: Centrum stałych konfiguracyjnych.

**Artykuł II: Kluczowe Zasady Architektoniczne (Nowe Ustalenia)**

W toku prac nad `game_engine.py` i `agents.py` ustanowiono następujące, nadrzędne zasady:

1.  **Zasada Delegacji Decyzji (Dot. `game_engine.py`):** Silnik gry implementuje **dwuetapowy proces finalizacji tury**.

    - Metoda `apply_move(move)` wykonuje główną akcję i zwraca listę potencjalnych konsekwencji (np. dostępnych arystokratów).
    - Metoda `finalize_turn(chosen_noble)` zamyka turę po otrzymaniu od agenta ostatecznej decyzji.
    - **Uzasadnienie:** Taka architektura w pełni deleguje odpowiedzialność za strategiczne decyzje do agenta, utrzymując silnik w roli bezstronnego arbitra. Zapobiega to "przeciekaniu" logiki decyzyjnej do silnika.

2.  **Zasada Kompletnej Akcji (Dot. `agents.py`):** Wszyscy agenci, bez wyjątku, muszą implementować interfejs `Agent` i jego metodę `choose_action`.
    - Sygnatura: `choose_action(game_state, valid_moves) -> Tuple[Move, Optional[Noble]]`.
    - **Uzasadnienie:** Metoda ta wymusza na agencie zwrócenie **kompletnego pakietu decyzyjnego** na daną turę. Pętla główna gry nie musi interpretować stanu ani podejmować żadnych dodatkowych decyzji w imieniu agenta. Otrzymuje gotową, pełną akcję do wykonania.

**Artykuł III: Standard DNA Agenta Genetycznego (Nowy Artykuł)**

Ustanawia się oficjalny standard struktury genetycznej dla `GeneticAgent`, który jest wiążący dla wszystkich modułów wchodzących w interakcję z DNA agenta (w szczególności dla `evolution_manager.py`). Szczegółowa specyfikacja znajduje się w poniższym dodatku.

---

### **Dodatek Techniczny A: Standard DNA Agenta Genetycznego (wersja 2.0)**

**1. Cel Dokumentu:**
Niniejszy dodatek jest jedynym i ostatecznym źródłem prawdy dotyczącym struktury, rozmiaru i interpretacji wektora DNA używanego przez `GeneticAgent`. Moduł `evolution_manager.py` MUSI operować na DNA zgodnym z tą specyfikacją.

**2. Specyfikacja Ogólna:**

- **Stały Rozmiar (DNA_SIZE):** 16
- **Typ Danych:** `numpy.ndarray`
- **Typ Elementów:** `float`
- **Zalecany Zakres Inicjalizacji Genów:** [-1.0, 1.0]

**3. Mapa Genów:**
Wektor DNA jest podzielony na dwie logiczne sekcje, które pozwalają agentowi na ocenę stanu gry zarówno z perspektywy własnej, jak i w kontekście otoczenia.

**Sekcja 1: Cechy Własne (Geny 0-11) - Ocena Wewnętrzna**

- `Gen 0`: **Waga Punktów Prestiżu.** Kluczowy gen determinujący dążenie do wygranej.
- `Geny 1-5`: **Wagi Bonusów z Kart.** Określają wartość posiadania stałej "produkcji" dla każdego z pięciu kolorów klejnotów (kolejność: WHITE, BLUE, GREEN, RED, BLACK).
- `Geny 6-10`: **Wagi Posiadanych Żetonów.** Określają wartość posiadania płynnych zasobów dla każdego z pięciu kolorów (kolejność j.w.).
- `Gen 11`: **Waga Kart Zarezerwowanych.** Określa, czy agent preferuje posiadanie opcji na przyszłość, czy uważa to za balast.

**Sekcja 2: Cechy Kontekstowe (Geny 12-15) - Ocena Otoczenia**

- `Gen 12`: **Waga Różnicy Punktowej do Lidera.** Określa, jak bardzo agent przejmuje się dystansem do najsilniejszego przeciwnika. Ujemna wartość DNA lidera będzie tu mnożona przez ujemną cechę, dając pozytywny wynik.
- `Gen 13`: **Waga Przewagi w Bonusach.** Określa wartość posiadania większej liczby stałych bonusów niż średnia przeciwników. Pozwala na ocenę długoterminowego potencjału.
- `Gen 14`: **Waga Przewagi w Żetonach.** Określa wartość posiadania większej liczby żetonów niż średnia przeciwników. Pozwala na ocenę siły nabywczej w krótkim terminie.
- `Gen 15`: **Waga Osiągalności Karty.** Określa, jak bardzo agent ceni sytuację, w której najlepsze karty na stole są w jego zasięgu (niski koszt dojścia). Wartość cechy jest normalizowana do przedziału.
