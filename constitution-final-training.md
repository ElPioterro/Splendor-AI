### **Konstytucja Projektu PROMETHEUS (Wersja 5.0 - Ostateczna)**

**Preambuła**

My, Architekci Projektu PROMETHEUS, w celu stworzenia stabilnej, rozszerzalnej i długowiecznej architektury oprogramowania, ustanawiamy niniejszą Konstytucję. Jej celem jest skodyfikowanie fundamentalnych zasad rozdziału odpowiedzialności (Separation of Concerns) pomiędzy kluczowymi modułami systemu. Niniejszy dokument stanowi najwyższe prawo projektu, a jego postanowienia są wiążące dla wszystkich obecnych i przyszłych deweloperów.

---

**Artykuł I: Zasada Jedynego Źródła Prawdy (The Single Source of Truth)**

1.  Moduł `game_engine.py` jest wyłącznym i ostatecznym autorytetem w kwestii stanu gry i jej zasad.
2.  Żaden inny moduł nie ma prawa modyfikować wewnętrznego stanu gry (`GameState`) w sposób bezpośredni. Zmiany mogą następować wyłącznie poprzez wywołanie dedykowanych metod silnika, takich jak `apply_move`.
3.  Silnik jest odpowiedzialny za walidację każdego ruchu i egzekwowanie reguł. Próba wykonania nielegalnego ruchu musi skutkować jednoznacznym błędem.

---

**Artykuł II: Zasada Czystej Intencji Strategicznej (The Pure Strategic Intent)**

1.  Moduł `agents.py` i jego klasy pochodne (np. `GeneticAgent`, `HumanAgent`) są odpowiedzialne wyłącznie za podejmowanie decyzji strategicznych.
2.  Zadaniem Agenta jest analiza stanu gry (`GameState`) i zwrócenie obiektu `Move`, reprezentującego wybraną akcję.
3.  Agent nie może posiadać wiedzy o sposobie prezentacji danych (GUI), ani o głównym cyklu aplikacji (`main.py`). Jego interakcja ze światem zewnętrznym jest ograniczona do otrzymania stanu gry i zwrócenia ruchu.

---

**Artykuł III: Zasada Integralności Definicji Gry (The Game Definition Integrity)**

1.  Moduł `game_loader.py` jest jedynym autorytetem odpowiedzialnym za wczytywanie, walidację i udostępnianie definicji komponentów gry (karty, arystokraci) z zewnętrznych źródeł danych (np. plik JSON).
2.  Silnik Gry (`game_engine.py`) oraz inne moduły muszą pozyskiwać definicje wyłącznie za pośrednictwem `GameLoader`.
3.  Format danych wejściowych jest nienaruszalny. `GameLoader` musi gwarantować, że obiekty dostarczane do systemu są spójne i kompletne.

---

**Artykuł IV: Zasada Centralnego Dyrygenta (The Central Conductor)**

1.  Moduł `main.py` pełni rolę centralnego dyrygenta aplikacji. Jest odpowiedzialny za inicjalizację wszystkich kluczowych komponentów (Silnik, Agenci, GUI) oraz za zarządzanie główną pętlą gry.
2.  `main.py` orkiestruje przepływ informacji pomiędzy modułami: pobiera ruchy od Agentów, przekazuje je do Silnika w celu walidacji i aplikacji, a następnie informuje GUI o nowym stanie gry do wyświetlenia.
3.  Logika biznesowa gry nie może znajdować się w `main.py`. Jego rola jest ograniczona do koordynacji.

---

**Artykuł V: Zasada Czystej Prezentacji (The Pure Presentation Layer)**

1.  Moduł `gui.py` jest odpowiedzialny wyłącznie za wizualną prezentację stanu gry (`GameState`) oraz za przechwytywanie surowych intencji użytkownika (np. kliknięcie myszą, zamknięcie okna).
2.  GUI nie może zawierać żadnej logiki biznesowej gry. Nie ma prawa decydować, czy ruch jest legalny, ani bezpośrednio modyfikować stanu gry.
3.  Wszelkie dane potrzebne do rysowania są przekazywane do `gui.py` przez `main.py`. Wszelkie akcje użytkownika są przez `gui.py` zwracane do `main.py` jako ustrukturyzowane, niezinterpretowane zdarzenia (intencje).

---

**Artykuł VI: Zasada Dekompozycji Intencji Użytkownika (The User Intent Deconstruction)**

1.  Wprowadza się dedykowany mechanizm lub klasę (np. `HumanMoveConstructor`) w `main.py`, którego wyłączną odpowiedzialnością jest tłumaczenie sekwencji surowych intencji z GUI na konkretne, gotowe do walidacji obiekty `Move`.
2.  Mechanizm ten zarządza stanem niekompletnych akcji (np. wybór pierwszego z trzech żetonów) bez obciążania logiki GUI.
3.  GUI informuje ten mechanizm o prostych zdarzeniach (`"select_gem"`, `"buy_card"`), a mechanizm decyduje, czy na ich podstawie można już zbudować kompletny ruch i zwrócić go do głównej pętli aplikacji.

---

**Artykuł VII: Architektura Intencji Użytkownika (The User Intent Architecture)**

1.  GUI jest odpowiedzialne za mapowanie surowych zdarzeń (np. kliknięcie na współrzędnych X,Y) na semantyczne intencje (np. `('buy_card', Card_Object)`).
2.  Intencje te są przekazywane do `main.py` w ustandaryzowanym formacie.
3.  `main.py`, przy użyciu konstruktora z Artykułu VI, przetwarza intencje na obiekty `Move`.
4.  **Zasada Wyzwalania Efektów Prezentacji.** `main.py`, po pomyślnym przetworzeniu intencji i walidacji ruchu przez silnik, jest odpowiedzialny za wywołanie w `gui.py` dedykowanych, nieblokujących funkcji (trigger\_...), które inicjują efekty wizualne (animacje). `gui.py` jest w pełni autonomiczny w zarządzaniu cyklem życia tych efektów.

---

**Artykuł VIII: Zasada Responsywnej Prezentacji (The Responsive Presentation Principle)**

Ustanawia się, że `gui.py` musi dynamicznie obliczać layout swoich komponentów (`_compute_layout`) w zależności od rozmiaru okna. Wszystkie operacje rysowania muszą bazować na tej dynamicznie wyliczonej metryce. Celem jest zapewnienie czytelności interfejsu niezależnie od rozdzieluczości.

---

**Artykuł IX: Architektura Inspektora Pomocy Kontekstowej (The Contextual Help Inspector Architecture)**

Ustanawia się w `gui.py` dedykowany panel "Inspektora", którego jedyną rolą jest dostarczanie graczowi pasywnych, kontekstowych informacji. Musi on:

1.  Wyświetlać szczegóły komponentu gry (np. karty), nad którym znajduje się kursor.
2.  Analizować i prezentować relację między najechanym komponentem a aktualnym stanem gracza (np. "brakuje ci 2 żetonów do zakupu").
3.  Nie może zawierać żadnych interaktywnych elementów (przycisków).
