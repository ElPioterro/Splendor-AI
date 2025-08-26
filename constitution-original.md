Konstytucja Odnogi „Neuroewolucja” (branch: nn-agent)

Preambuła
Niniejsza Konstytucja definiuje architekturę, kontrakty i zasady działania odnogi projektu PROMETHEUS opartej o agenta neuronowego trenowanego metodami ewolucyjnymi (ES). Celem jest zapewnienie:

- spójności z istniejącą filozofią projektu (SSOT, czyste warstwy, deterministyczność),
- wysokiej ekspresyjności decyzji dzięki nieliniowej funkcji oceny (MLP),
- powtarzalności, solidnych ewaluacji i łatwej orkiestracji treningu.

I. Słownik i założenia

- SSOT (Single Source of Truth): silnik gry (game_engine.py) jest jedynym autorytetem reguł, losowości i stanu.
- Agent NN (NnAgent): polityka wyboru ruchu oparta o niewielką sieć MLP, która ocenia parę (cechy stanu, cechy ruchu) i wybiera argmax.
- ES (Evolution Strategies, OpenES): algorytm optymalizacji parametrów sieci przez próbkowanie zaburzeń i oszacowanie gradientu bezpośrednio na wynikach gier, bez backprop w środowisku.
- Anchors (Kotwice): stały zestaw przeciwników testowych (baseline DNA, champion GA, top HoF) do oceny postępów i nadawania progu trudności.
- WR_LB: dolna granica Wilsona dla win rate (konserwatywna ocena skuteczności).
- PPT (points-per-turn): punkty na turę – lekki tie-break tempa.

II. Zasady nadrzędne (niezmienialne bez nowelizacji)

1. Zasada SSOT (Single Source of Truth)

- Tylko Game kontroluje stan, losowość rozdań i egzekwowanie reguł.
- Żaden agent, trener ani GUI nie może modyfikować GameState bezpośrednio.
- Każdy ruch jest walidowany przez Game; próba nielegalnego ruchu kończy się błędem lub odrzuceniem.

2. Zasada czystego agenta

- Agent (Genetic lub NnAgent) otrzymuje stan i listę dozwolonych ruchów, a zwraca (Move, Optional[Noble]).
- Agent nie manipuluje silnikiem; nie zna GUI ani pętli main.
- Agent posiada własny, lokalny RNG (deterministyczny przy rng_seed).

3. Zasada deterministyczności i powtarzalności

- Game ma własny RNG z seedem; seedy przekazywane są explicite; nie używamy globalnego random.seed.
- Evaluacje gier używają kontrolowanych seedów; odwrócenie ról może używać seed^XOR.
- Agent NN posiada lokalny RNG do tie-breaków.

4. Zasada rozdziału odpowiedzialności (Separation of Concerns)

- game_engine.py: logika gry; deterministyczny setup; finalize_turn waliduje nobles.
- feature_encoder.py: czysty kod cech; szybki (bez deepcopy), bez logiki gry.
- nn_agent.py: definicja MLP, unflatten parametrów, forward, choose_action.
- es_manager.py: trening ES, ewaluacje, anchors/HOF, logi, zapisy modeli.
- nn_config.py: wszystkie hiperparametry NN/ES i ewaluacji; brak logiki.
- train_nn.py: prosty driver uruchamiający trening (bootstrap środowiska).

III. Technologia

- Język i runtime: Python 3.10/3.11, bez ciężkich frameworków ML (numpy + własne MLP).
- Zależności minimalne: numpy, tqdm (opcjonalnie dev: pytest, pandas, matplotlib).
- Równoległość: multiprocessing (pool) na poziomie ewaluacji kandydatów/perturbacji.

IV. Struktura plików i kontrakty

1. nn_config.py (Konfiguracja NN/ES)

- Architektura: STATE_DIM, MOVE_DIM, HIDDEN1, HIDDEN2, ACT (relu/silu/tanh).
- ES: ES_POP, ES_SIGMA, ES_LR, ANNEAL_SIGMA, ES_ELITE_FRACTION (opcjonalne).
- Ewaluacja: ANCHORS_COUNT, EVAL_SEEDS_PER_ANCHOR, WILSON_Z, TURNS_BASELINE, PPT_WEIGHT.
- Losowość/MP: SEED_BASE, USE_MULTIPROCESSING, N_PROCESSES.

Kontrakt:

- Nie trzymamy logiki w configu; wartości są wyłącznie parametrami używanymi przez es_manager/nn_agent/encoder.

2. feature_encoder.py (Czyste kodowanie cech)

- encode_state_and_moves(state, valid_moves) -> (state_vec, [move_vecs]).
- Cechy stanu: prestiż, bonusy, żetony, rezerwy, agregaty przeciwników (max/średnia), dostępne żetony na stole, turn_number (znormalizowany).
- Cechy ruchu: one-hot typu (Buy/Reserve/Take), delta prestiżu/bonusu/żetonów w przybliżeniu, presja przeciwnika, postęp tury.
- Szybkość: bez deepcopy; delty aproksymujemy lokalnie; normalizacje proste i deterministyczne.

Kontrakt:

- Encoder jest czysty (stateless), niezależny od silnika i trenera; nie powoduje skutków ubocznych.

3. nn_agent.py (Agent NN)

- MLP: in_dim = STATE_DIM + MOVE_DIM; dwie warstwy ukryte; aktywacja ACT.
- forward(params, x) -> float; choose_action(state, valid_moves) -> (Move, None).
- Parametry sieci spakowane w wektor theta; unflatten(theta) rozpakowuje do (W,b).
- Tie-break dla remisów po wyniku: rng choice po indeksach najlepszych.

Kontrakt:

- NnAgent ma ten sam interfejs choose_action co agenci GA (kompatybilność z pętlą gry).
- Nie wybiera nobles – pozostawiamy to SSOT (Game po apply_move zwraca eligible; finalize_turn przyjmie tylko legalny).

4. es_manager.py (Trener ES)

- OpenES (mirrored sampling):
  - Losujemy epsilony e ~ N(0,I).
  - Tworzymy pary theta+σe oraz theta−σe (2\*ES_POP kandydatów).
  - Ewaluujemy je równolegle vs anchors/seed/role.
  - Budujemy estymatę gradientu z różnic (r_plus − r_minus) i aktualizujemy theta.
- Ewaluacja kandydatów:
  - Zestaw anchorów (baseline DNA + champion GA + top HoF GA – opcjonalnie).
  - Seedy i obie role.
  - Zliczamy wins (remis=0.5), games, WR, WR_LB (Wilson), avg_turns, avg_points, margin.
  - Reward ES: score = WR_LB + PPT_WEIGHT \* min(1, (avg_points/avg_turns)/0.30).
- Logi:
  - logs/es_log.csv: gen, score, wr_lb, wr, sigma, time_sec.
  - Model: nn_champion.npy (wektor theta).
- Kontrola zasobów:
  - USE_MULTIPROCESSING i N_PROCESSES; równoległa ewaluacja perturbacji.
  - Annealing sigmy (ANNEAL_SIGMA).

Kontrakt:

- es_manager nie łamie SSOT – wszystkie gry przez Game; nie dotyka GUI.
- es_manager nie wprowadza stanu do agentów poza seedem; każdy mecz tworzy świeże instancje.

5. train_nn.py (Driver)

- Ładuje dane gry (GameLoader, zgodnie z dotychczasowym kontraktem),
- Tworzy ESManager i wywołuje run(),
- Nie zawiera logiki treningowej poza bootstrapem.

V. Algorytmika i ewaluacja

1. OpenES (rdzeń)

- Pary lustrzane (mirrored sampling) poprawiają stosunek sygnał/szum.
- Standaryzacja różnic nagród przed obliczeniem gradientu.
- Krok aktualizacji: theta ← theta + (lr/sigma) _ E[ r _ epsilon ].

2. Reward vs selekcja

- Nie używamy arbitralnej sumy wag (jak w GA) – sygnałem jest WR_LB, a PPT to lekki tie-break tempa.
- Dodatkowo można raportować hold‑out WR_LB (vs zamrożony zestaw anchorów/seedów) dla stabilnego monitoringu.

3. Anchors i Hall of Fame

- Stabilny próg: baseline + champion GA + top HoF GA (unikalne DNA; brak losowych duplikatów).
- Liczba anchorów kontrolowana parametrem (NN.ANCHORS_COUNT); można użyć harmonogramów (schedule) analogicznie do GA.

4. Determinizm i seedy

- Game(seed=...), per‑rola opcjonalny XOR.
- NnAgent(rng_seed=...), powtarzalny tie-break.
- OpenES: SEED_BASE deterministycznie generuje seedy perturbacji i meczów; workerom przekazujemy pełny kontekst, unikając globalnych seedów.

VI. Interoperacyjność z gałęzią GA (main)

- Koegzystencja: NnAgent i GeneticAgent mają zgodny interfejs choose_action – można je zestawiać w meczach pokazowych/przeglądach.
- Anchors: na starcie NN odnogi kotwice mogą być wprost agentami GA (z baseline/HoF/champion). To skraca czas dowodzenia jakości NN.
- Adapter do main: flaga --play-nn ładuje nn_champion.npy i uruchamia mecz vs champion GA (dowód jakości).

VII. Przejrzystość i obserwowalność

- Logi CSV: es_log.csv w logs/ (niewielki stały zestaw kolumn; liczby jako liczby).
- Konsola: progres generacji i ewaluacji (tqdm), WR_LB w każdej generacji.
- Możliwość dodania „hold‑out WR_LB” (rekomendowane do ładnej, gładkiej krzywej porównawczej).
- Powtarzalność: zapis seeda bazowego i parametrów w pliku metadanych (opcjonalne).

VIII. Wydajność i budżet

- Minimalne zależności (numpy) → działa szybko na CPU.
- Równoległa ewaluacja perturbacji (Pool) daje niemal liniowe przyspieszenie do liczby rdzeni.
- Parametry do szybkiego startu:
  - ES_POP 64–128,
  - σ 0.1 (anneal do 0.03),
  - lr 0.05 (dostosować),
  - Anchors 4, Seedy 2 (dla top-perturbacji można przeliczyć z większą próbą).

IX. Jakość, testy i bezpieczeństwo

- Inwarianty:
  - żaden moduł nie modyfikuje GameState poza Game.apply_move/finalize_turn,
  - agent nie wybiera nielegalnego nobles; finalize_turn przyjmie wyłącznie kwalifikowalnego,
  - brak globalnych seedów wpływających na inne procesy.
- Testy (opcjonalnie na branch nn-agent):
  - deterministyczność NnAgent (tie-break po seed),
  - zgodność wybierania ruchu z interfejsem Agent,
  - powtarzalność wyników gier (Game RNG + seeds),
  - sanity check OpenES (prosta funkcja testowa – nie obowiązkowe).

X. Rozszerzalność i roadmap (otwarta ewolucja)

- Feature engineering:
  - lepsze delty dla TakeThree/TakeTwo (shortfall do wybranych kart),
  - block_value (rezerwa, by „ukraść” przeciwnikowi),
  - noble_distance (min/avg odległość),
  - scarcity_pressure (rzadkość kolorów w banku/na stole).
- Architektura:
  - HIDDEN1/2 32–128 (kontrolując param_count 10–50 tys.),
  - ACT silu/relu,
  - ew. dropout/warstwowe normy (tylko inferencyjnie, bez gradientu).
- Trener:
  - rank-based lub CMA‑ES (lepsza stabilność przy większej liczbie parametrów),
  - dystrybucja perturbacji antithetic + common random numbers (dalsza redukcja wariancji).
- Ewaluacja:
  - hold‑out kotwice/seed window (stały nadzór),
  - cross‑play vs agent GA i poprzednich NN championów.

XI. Migracja i zgodność

- Gałąź nn-agent rozwijana równolegle do main (GA).
- Modele NN zapisujemy do nn_champion.npy; nie kolidują z champion_agent.npy (GA).
- requirements.txt spójny dla obu gałęzi (numpy, tqdm).
- Nie naruszamy istniejących CSV/ścieżek GA; NN ma własny es_log.csv.

XII. Governance (zarządzanie zmianą)

- Każda zmiana łamiąca kontrakty (np. interfejs agenta, format logów) wymaga nowelizacji Konstytucji odnogi NN.
- Zmiany parametrów trenowania (nn_config.py) nie wymagają nowelizacji – to działania operacyjne.
- Skoordynowane wydania: tagujcie „nn-vX.Y” przy stabilnych milestone’ach (np. pierwsze zwycięstwo NN nad championem GA w hold‑out WR_LB > 0.7).

Załącznik A: Minimalne API i przepływ

- train_nn.py:

  - Ładuje all_cards/all_nobles,
  - Tworzy ESManager i run(generations=N),
  - Zapisuje nn_champion.npy i logs/es_log.csv.

- es_manager.run:

  - Dla generacji g:
    - Próbuje perturbacje (theta±σe),
    - Ewaluacja każdej perturbacji vs anchors (seeds, role),
    - Szacunek gradientu i update theta,
    - Ewaluacja aktualnej theta (WR, WR_LB, score),
    - Log do CSV + konsola, annealing σ.

- NnAgent.choose_action:
  - Buduje (state_vec, move_vecs),
  - Ocenia ruchy: y = MLP([state, move]),
  - Zwraca argmax + None (noble wybierze engine po apply_move).

Załącznik B: Zasady bezpieczeństwa i „ostrożność badawcza”

- WR_LB jest „konserwatywną” metryką – spodziewaj się zmian przy małej próbce; dla top‑modeli używaj większej liczby seedów (re-score).
- Zmieniaj jeden „wymiar” eksperymentu na raz (sigma albo lr, ale nie oba naraz).
- Zachowuj CSV i modele – porównania między biegami są kluczowe.

Koniec Konstytucji
Niniejsza Konstytucja obowiązuje w gałęzi nn-agent. Jest kompatybilna z wartościami i pryncypiami PROMETHEUSA (SSOT, czyste warstwy, deterministyczność) i otwiera nowy rozdział – agent neuronowy, trenowany metodami neuroewolucji, gotowy do rywalizacji z i przewyższania agenta genetycznego. Ruszamy.
