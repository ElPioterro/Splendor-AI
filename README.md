# Splendor AI: Projekt PROMETHEUS

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Samoucząca się sztuczna inteligencja do gry w strategiczną grę planszową Splendor. Projekt PROMETHEUS eksploruje dwie różne ścieżki tworzenia nadludzkiego agenta AI: klasyczne algorytmy genetyczne oraz nowoczesną neuroewolucję.

---

### Dwie Drogi do Inteligencji

Repozytorium to zawiera dwa fundamentalnie różne podejścia do stworzenia AI, rozwijane w osobnych gałęziach (branchach):

1.  **Gałąź `main`: Algorytm Genetyczny (GA)**

    - **Filozofia:** Agent oparty na liniowym modelu, którego "DNA" to wektor wag przypisanych do ręcznie stworzonych cech (np. "wartość punktu prestiżu", "wartość posiadania bonusu").
    - **Trening:** Klasyczna ewolucja darwinowska. Populacja agentów rozgrywa turnieje, a najlepsi są selekcjonowani, krzyżowani i mutowani, aby stworzyć doskonalsze pokolenia.
    - **Status:** W pełni ukończony, stabilny i grywalny. Stanowi solidny punkt odniesienia.

2.  **Gałąź `nn-agent`: Neuroewolucja (ES + MLP)**
    - **Filozofia:** Agent oparty na małej sieci neuronowej (MLP), która uczy się nieliniowych zależności między cechami stanu gry a oceną ruchu.
    - **Trening:** Ewolucja strategii (ES). Zamiast krzyżować agentów, optymalizujemy bezpośrednio wagi sieci neuronowej, traktując je jak wektor i szacując gradient "sukcesu" w przestrzeni parametrów.
    - **Status:** Gałąź eksperymentalna, mająca na celu sprawdzenie, czy nieliniowość sieci neuronowej może przełamać ograniczenia agenta genetycznego.

### Struktura Projektu

Projekt jest zbudowany w oparciu o radykalną separację odpowiedzialności, skodyfikowaną w wewnętrznej **Konstytucji Projektu**:

- `game_engine.py`: Bezwzględne źródło prawdy o zasadach gry.
- `agents.py` / `nn_agent.py`: Mózgi decyzyjne agentów.
- `evolution_manager.py` / `es_manager.py`: Fabryki Inteligencji, zarządzające procesem treningu offline.
- `gui.py`: "Głupi" terminal graficzny, odpowiedzialny wyłącznie za prezentację.
- `main.py` / `train_nn.py`: Główni dyrygenci, spajający wszystkie komponenty.

### Jak Zacząć

1.  **Klonuj repozytorium i przejdź do wybranej gałęzi:**

    ```bash
    # Dla agenta genetycznego
    git clone https://github.com/TWOJA_NAZWA/Splendor-AI.git
    cd Splendor-AI
    git checkout main

    # Dla agenta opartego na sieci neuronowej
    git clone https://github.com/TWOJA_NAZWA/Splendor-AI.git
    cd Splendor-AI
    git checkout nn-agent
    ```

2.  **Stwórz środowisko wirtualne i zainstaluj zależności:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # Na Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Trenuj AI:**

    ```bash
    # Dla wersji GA
    python main.py --train

    # Dla wersji NN
    python train_nn.py
    ```

    Proces treningu wygeneruje plik `champion_agent.npy` lub `nn_champion.npy`.

4.  **Graj przeciwko AI:**
    ```bash
    # Dla wersji GA
    python main.py --play
    ```
