# AI Expert Asks About Project

Doskonałe pytanie. Przechodzimy teraz na "meta-poziom": nie chodzi już o to, _co_ zbudować, ale _jak najefektywniej współpracować z AI (takim jak ja), aby to zbudować_. To jest kluczowa umiejętność w nowoczesnym programowaniu.

Twoja intuicja jest słuszna. Sposób, w jaki zorganizujesz pracę z AI, będzie miał ogromny wpływ na jakość kodu, spójność projektu i twoją własną frustrację (lub jej brak).

Oto moja odpowiedź, podzielona na strategię, taktykę i konkretne przykłady.

### **Strategia ogólna: Myśl o sobie jako o Architekcie, a o AI jako o Błyskotliwym Programiście-Stażyście**

Twój stażysta (AI) jest niesamowicie szybki, zna na pamięć całą dokumentację świata, ale brakuje mu nadrzędnego kontekstu projektu. Czasem zapomina, co ustaliliście godzinę temu i potrzebuje precyzyjnych, jednoznacznych poleceń. Nie zrozumie niejasnego "wiesz, o co chodzi".

Dlatego twoim głównym zadaniem jest **zarządzanie kontekstem**.

### **Jedna długa konwersacja czy wiele dedykowanych chatów?**

Odpowiedź brzmi: **stosuj strategię hybrydową.**

1.  **Główny Wątek Projektowy (Jeden, długi chat):**

    - **Cel:** Utrzymanie ogólnej wizji, architektury i integracji projektu.
    - **Co tu robisz:**
      - Wklejasz i zatwierdzasz finalną wersję "Konstytucji Projektu" (o tym za chwilę).
      - Dyskutujesz o ogólnej architekturze (np. "Czy `evolution_manager` powinien zapisywać pliki bezpośrednio, czy przez dedykowaną klasę `PersistenceManager`?").
      - Wklejasz tu **finalne, przetestowane** wersje kodu z poszczególnych modułów, mówiąc: "Oto ostateczna wersja `game_engine.py`. Dodajmy ją do naszego projektu."
    - **Dlaczego tak?** To tworzy jedno, spójne źródło prawdy o stanie całego projektu. Gdy będziesz chciał zintegrować moduły, wrócisz tutaj, a AI będzie miało wgląd w całość.

2.  **Wątki Funkcjonalne (Wiele, krótkich chatów):**
    - **Cel:** Skoncentrowana praca nad jednym, konkretnym modułem (np. `game_engine.py`).
    - **Co tu robisz:**
      - Rozpoczynasz nowy chat z jasnym celem: "Naszym zadaniem jest zbudowanie modułu `game_engine.py` do gry Splendor. Będziemy pracować tylko nad tym plikiem."
      - Iteracyjnie tworzysz kod, testujesz go, prosisz o poprawki, refaktoryzację – wszystko w ramach tego jednego modułu.
    - **Dlaczego tak?** AI ma ograniczony "bufor pamięci" (okno kontekstowe). Skupiając się na jednym module w dedykowanym chacie, zapewniasz, że AI ma maksymalny możliwy kontekst dotyczący _tego konkretnego zadania_. Unikasz sytuacji, w której AI zaczyna "mieszać" logikę z różnych części projektu.

### **Jak najefektywniej prosić AI o instrukcje dla... AI? Stwórz "Konstytucję Projektu"**

To jest najważniejsza koncepcja. Zanim napiszesz choćby jedną linię kodu, stwórz dokument (może być w markdown), który będzie twoim "meta-promptem" lub "Konstytucją Projektu". Będziesz go wklejać na początku każdej nowej sesji (wątku funkcjonalnego), aby natychmiastowo załadować cały kontekst.

Oto, co powinna zawierać "Konstytucja Projektu Splendor AI":

---

**Konstytucja Projektu: Splendor AI**

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

---

### **Praktyczny Przepływ Pracy (krok po kroku):**

**Krok 1: Budowa Silnika Gry (`game_engine.py`)**

1.  **Otwórz nowy chat.**
2.  **Wklej całą "Konstytucję Projektu"** jako pierwszą wiadomość.
3.  Twoje pierwsze polecenie:

    > "OK, bazując na naszej konstytucji, zaczynamy pracę nad modułem `game_engine.py`. Proszę, zaproponuj strukturę klasy `GameState` i `Card`. Na początek użyj dataclasses z Pythona. Klasa `Card` powinna zawierać koszt, kolor bonusu i punkty prestiżu. `GameState` powinna zawierać listę graczy, dostępne klejnoty i karty na stole."

4.  AI wygeneruje kod. Teraz twoja rola to recenzja i iteracja:

    > "Dobrze. Teraz dodaj do `GameState` metodę `get_valid_moves` dla aktywnego gracza. Na razie zaimplementujmy tylko jedną możliwą akcję: wzięcie trzech różnych klejnotów. Metoda powinna zwrócić listę możliwych do wzięcia kombinacji."

5.  Kontynuuj w ten sposób, budując i testując moduł kawałek po kawałku, aż będzie kompletny.

**Krok 2: Przejście do Następnego Modułu**

1.  Gdy `game_engine.py` jest gotowy i przetestowany, skopiuj jego finalny kod.
2.  Wróć do **Głównego Wątku Projektowego** i wklej go z komentarzem:
    > "Zakończyliśmy pracę nad `game_engine.py`. Oto jego finalna wersja: [wklej kod]. Teraz rozpoczniemy pracę nad `agents.py`."
3.  **Otwórz nowy chat** dedykowany dla `agents.py`.
4.  Znowu: **wklej "Konstytucję Projektu"**.
5.  A potem dodaj kluczowy kontekst:
    > "Rozpoczynamy pracę nad modułem `agents.py`. Będzie on korzystał z klas zdefiniowanych w `game_engine.py`. Oto kod `game_engine.py`, żebyś miał pełen kontekst: [wklej kod game_engine.py]. Teraz, zgodnie z konstytucją, stwórz abstrakcyjną klasę bazową `Player` z metodą `choose_move`, a następnie zaimplementuj `RandomAgent`, który losowo wybiera ruch z listy `valid_moves`."

Dzięki temu podejściu każdy "specjalistyczny" chat ma dokładnie ten kontekst, którego potrzebuje, a główny wątek służy jako spoiwo całego projektu.

### **Dodatkowe porady:**

- **Bądź precyzyjny:** Zamiast "napraw to", powiedz "W metodzie `apply_move` występuje błąd, gdy gracz próbuje kupić kartę, na którą go nie stać. Dodaj walidację sprawdzającą, czy `player.gems` są wystarczające do pokrycia `card.cost`."
- **Proś o wyjaśnienia:** Jeśli AI proponuje skomplikowane rozwiązanie, poproś: "Zanim napiszesz kod, wyjaśnij mi krok po kroku, jak zamierzasz zaimplementować funkcję ewolucji. Jakie kroki wykonasz w pętli treningowej?" To pozwala wyłapać błędy w logice, zanim powstaną.
- **Twoja rola to testowanie:** AI nie uruchomi kodu. Ty musisz być testerem. Kopiuj, wklejaj, uruchamiaj i raportuj błędy z powrotem do AI, dostarczając pełny traceback.

Podsumowując: twoja efektywność wzrośnie dramatycznie, jeśli przestaniesz traktować AI jak "czarną skrzynkę", a zaczniesz jak partnera w programowaniu, którego kontekstem i zadaniami musisz świadomie zarządzać. Stworzenie "Konstytucji Projektu" i stosowanie hybrydowej strategii chatów to najlepszy sposób, aby to osiągnąć.
