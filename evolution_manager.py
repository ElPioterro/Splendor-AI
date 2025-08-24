# evolution_manager.py
#
# Wersja zintegrowana z patchem "Operacja GENIUSZ".
# Wprowadza architekturę opartą na kotwicach (anchors), Hall of Fame
# i annealingu mutacji w celu przełamania stagnacji treningu.

import time, random, math, os, csv, json
from datetime import datetime
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from multiprocessing import Pool, cpu_count

# Importy z innych modułów projektu
import config
from agents import GeneticAgent
from game_engine import Game, Card, Noble

# evolution_manager.py (u góry)
try:
    from tqdm.auto import tqdm
except Exception:
    # prosty fallback – brak paska, zachowuje API update/close
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, leave=False, position=None, dynamic_ncols=True):
            self.iterable = iterable if iterable is not None else range(total or 0)
        def __iter__(self): 
            return iter(self.iterable)
        def update(self, n=1): 
            pass
        def close(self): 
            pass

# Globalny kontekst dla workerów (żeby nie przerzucać all_cards/all_nobles w każdej pracy)
G_ALL_CARDS = None
G_ALL_NOBLES = None

def _wilson_lb(wins: float, games: int, z: float = 1.96) -> float:
    """Dolna granica przedziału Wilsona dla win-rate (obsługuje też remisy jako 0.5)."""
    if games <= 0:
        return 0.0
    p = wins / games
    denom = 1.0 + (z*z)/games
    centre = p + (z*z)/(2.0*games)
    margin = z * math.sqrt((p*(1.0-p))/games + (z*z)/(4.0*games*games))
    lb = (centre - margin) / denom
    return float(max(0.0, min(1.0, lb)))

def _mp_init_worker(all_cards, all_nobles):
    global G_ALL_CARDS, G_ALL_NOBLES
    G_ALL_CARDS = all_cards
    G_ALL_NOBLES = all_nobles

def _evaluate_candidate_worker_idx(args):
    # args: (idx, dna_list, anchors_dna_list, gen_idx, cfg)
    idx, dna_list, anchors_dna_list, gen_idx, cfg = args
    # wykorzystaj istniejącego workera bez-indexowego:
    metrics = _evaluate_candidate_worker((dna_list, anchors_dna_list, gen_idx, cfg))
    return idx, metrics

def _evaluate_candidate_worker(args):
    # Worker: ewaluacja jednego kandydata na wielu anchorach i seedach
    from agents import GeneticAgent
    from game_engine import Game

    dna_list, anchors_dna_list, gen_idx, cfg = args
    dna = np.array(dna_list, dtype=float)

    wins = 0.0; games = 0; pts_sum = 0.0; margin_sum = 0.0; turns_sum = 0.0
    seeds = [int(cfg["seed_base"] + 7919*gen_idx + 97*i) for i in range(cfg["seeds_per_anchor"])]

    for anchor_dna_list in anchors_dna_list:
        anchor_dna = np.array(anchor_dna_list, dtype=float)

        for s in seeds:
            for swap in (False, True):
                game_seed = s if not (cfg["different_seed_for_swap"]) else (s ^ cfg["swap_seed_xor"])
                game = Game(G_ALL_CARDS, G_ALL_NOBLES, seed=game_seed, target_points=cfg["target_points"])

                if not swap:
                    cand_seed = s + cfg["agent_seed_cand_offset"]
                    anch_seed = s + cfg["agent_seed_anch_offset"]
                    A = GeneticAgent(dna=dna.copy(), rng_seed=cand_seed); A.name = "Candidate"
                    B = GeneticAgent(dna=anchor_dna.copy(), rng_seed=anch_seed); B.name = "Anchor"
                    game.setup_new_game(["Candidate", "Anchor"])
                    players = [A, B]
                else:
                    s2 = game_seed
                    cand_seed = s2 + cfg["agent_seed_cand_offset"]
                    anch_seed = s2 + cfg["agent_seed_anch_offset"]
                    A = GeneticAgent(dna=anchor_dna.copy(), rng_seed=anch_seed); A.name = "Anchor"
                    B = GeneticAgent(dna=dna.copy(), rng_seed=cand_seed); B.name = "Candidate"
                    game.setup_new_game(["Anchor", "Candidate"])
                    players = [A, B]

                max_turns = int(cfg.get("max_turns", getattr(config, "MAX_TURNS", 150)))
                turns = 0
                while not game.is_game_over() and turns < max_turns:
                    state = game.game_state
                    agent = players[state.current_player_index]
                    valid_moves = game.get_valid_moves()
                    if not valid_moves:
                        game.finalize_turn(None)
                        turns += 1
                        continue
                    move, chosen_noble = agent.choose_action(state, valid_moves)
                    eligible_nobles = game.apply_move(move)
                    if chosen_noble not in eligible_nobles:
                        chosen_noble = None
                    game.finalize_turn(chosen_noble)
                    turns += 1

                p0 = game.game_state.players[0].prestige_points
                p1 = game.game_state.players[1].prestige_points
                pointsA = p0 if not swap else p1
                pointsB = p1 if not swap else p0

                winner_obj = game.get_winner()
                if winner_obj is None:
                    w_idx = -1
                else:
                    w_idx = 0 if winner_obj.name == "Candidate" else 1

                wins += 0.5 if w_idx == -1 else (1 if w_idx == 0 else 0)
                games += 1
                pts_sum += pointsA
                margin_sum += (pointsA - pointsB)
                turns_sum += turns

    if games == 0:
        return dict(win_rate=0.0, avg_points=0.0, avg_margin=0.0, avg_turns=0.0, fitness=0.0)

    win_rate = wins / games
    avg_points = pts_sum / games
    avg_margin = margin_sum / games
    avg_turns = turns_sum / games

    wr_lb = _wilson_lb(wins, games, z=cfg.get("wilson_z", getattr(config, "WILSON_Z", 1.96)))
    

    margin_norm = max(-1.0, min(1.0, avg_margin / 15.0))
    points_norm = max(0.0, min(1.0, avg_points / 15.0))
    turns_baseline = float(cfg.get("turns_baseline", getattr(config, "TURNS_BASELINE", 60.0)))
    turns_penalty = max(0.0, (avg_turns - turns_baseline) / turns_baseline)
    speed_bonus = max(0.0, (turns_baseline - avg_turns) / turns_baseline)

    fitness = (12.0 * win_rate) \
            + (3.0 * margin_norm) \
            + (1.5 * points_norm) \
            - (cfg.get("turns_penalty_weight", getattr(config, "TURNS_PENALTY_WEIGHT", 1.2)) * turns_penalty) \
            + (cfg.get("speed_bonus_weight", getattr(config, "SPEED_BONUS_WEIGHT", 0.8)) * points_norm * speed_bonus)

    return dict(
        win_rate=win_rate, avg_points=avg_points, avg_margin=avg_margin, avg_turns=avg_turns,
        wins=wins, games=games, wr_lb=wr_lb, fitness=fitness
    )

class EvolutionManager:
    """
    Zarządza całym cyklem ewolucji populacji agentów genetycznych,
    implementując zaawansowane techniki dla stabilnego i efektywnego treningu.
    """
    def __init__(self, all_cards: List[Card], all_nobles: List[Noble]):
        """
        Inicjalizuje menedżera ewolucji.
        """
        self.all_cards = all_cards
        self.all_nobles = all_nobles
        self.rng = np.random.default_rng(config.SEED_BASE)
        # Hall of Fame: lista par (dna, fitness_historyczny)
        self.hof: List[Tuple[np.ndarray, float]] = []
        self.champion: Optional[np.ndarray] = None  # Champion z ostatniej generacji
        self._load_hof()  # wczytaj, jeśli istnieje

        self.anchors_count = getattr(config, "ANCHORS_COUNT", 4)
        self._hi_winrate_streak = 0
        self._lo_winrate_streak = 0

        print("Zainicjowano EvolutionManager z nową, zaawansowaną architekturą.")
        print(f"Populacja: {config.POP_SIZE}, Generacje: {config.GENERATIONS}")


    def _adapt_dna_to_size(self, arr: np.ndarray) -> np.ndarray:
        """Dopasuj dowolny wektor do config.DNA_SIZE (pad 0 lub obetnij)."""
        arr = np.array(arr, dtype=float).ravel()
        target = int(getattr(config, "DNA_SIZE", 20))
        if arr.shape[0] == target:
            return arr.copy()
        if arr.shape[0] < target:
            pad = np.zeros(target - arr.shape[0], dtype=float)
            return np.concatenate([arr, pad], axis=0)
        return arr[:target].copy()

    # --- Hall of Fame (DNA + fitness historyczny) ---

    def _hof_find_similar(self, dna: np.ndarray, tol: float = 0.05) -> Optional[int]:
        """Zwraca indeks podobnego DNA w HoF (L2 < tol) lub None."""
        for i, (d, f) in enumerate(self.hof):
            try:
                if np.linalg.norm(d - dna) <= tol:
                    return i
            except Exception:
                pass
        return None

    def _hof_add(self, dna: np.ndarray, fitness: float, tol: float = 0.05) -> None:
        """Dodaje DNA do HoF lub aktualizuje wpis podobny, zachowując TOP-K po fitness."""
        dna = self._adapt_dna_to_size(dna)
        idx = self._hof_find_similar(dna, tol=tol)
        if idx is None:
            self.hof.append((dna, float(fitness)))
        else:
            # jeżeli nowy fitness jest lepszy – podmień
            if float(fitness) > float(self.hof[idx][1]):
                self.hof[idx] = (dna, float(fitness))
        # Posortuj malejąco po fitness i przytnij do K
        self.hof.sort(key=lambda t: t[1], reverse=True)
        self.hof = self.hof[:config.HOF_MAX]

    def _hof_get_dna_list(self) -> List[np.ndarray]:
        """Zwraca listę samych DNA z HoF (bez fitnessów)."""
        return [d for (d, f) in self.hof]

    def _save_hof(self, path: str = "hof.json") -> None:
        """Zapisuje HoF do JSON (trwałość między przebiegami)."""
        try:
            data = [{"fitness": float(f), "dna": d.tolist()} for (d, f) in self.hof]
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
        except Exception as e:
            print(f"[WARN] Nie udało się zapisać HoF: {e}")

    def _load_hof(self, path: str = "hof.json") -> None:
        """Wczytuje HoF z JSON, jeśli plik istnieje."""
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            loaded: List[Tuple[np.ndarray, float]] = []
            for item in data:
                dna_list = item.get("dna")
                fit_val = item.get("fitness")
                if isinstance(dna_list, list):
                    dna = self._adapt_dna_to_size(np.array(dna_list, dtype=float))
                    if dna.shape[0] == config.DNA_SIZE:
                        loaded.append((dna, float(fit_val)))
            # posortuj i przytnij do HOF_MAX, na wypadek starszego pliku
            loaded.sort(key=lambda t: t[1], reverse=True)
            self.hof = loaded[:config.HOF_MAX]
            if self.hof:
                print(f"Załadowano HoF ({len(self.hof)} wpisów) z {path}.")
        except Exception as e:
            print(f"[WARN] Nie udało się wczytać HoF: {e}")

    # --- Narzędzia Algorytmu Genetycznego (GA Utils) ---

    def _sigma_at(self, gen_idx: int, total: int) -> float:
        """Oblicza siłę mutacji (sigma) dla danej generacji (annealing)."""
        t = gen_idx / max(1, total - 1)
        return float((1 - t) * config.MUTATION_SIGMA_START + t * config.MUTATION_SIGMA_END)

    def _tournament_pick(self, population: List[np.ndarray], rank_keys: List[tuple]) -> np.ndarray:
        """Wybiera rodzica metodą turniejową po kluczu leksykograficznym rank_keys."""
        k = min(config.TOURNAMENT_K, len(population))
        idxs = self.rng.choice(len(population), size=k, replace=False)
        best_idx = max(idxs, key=lambda i: rank_keys[i])  # Python porównuje krotki leksykograficznie
        return population[best_idx]

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Wykonuje krzyżowanie jednorodne (uniform crossover)."""
        if self.rng.random() > config.CROSSOVER_RATE:
            return a.copy() if self.rng.random() < 0.5 else b.copy()
        mask = self.rng.random(a.shape) < 0.5
        child = np.where(mask, a, b)
        return child

    def _mutate(self, dna: np.ndarray, sigma: float) -> np.ndarray:
        """Aplikuje mutację z rozkładu normalnego i przycina wartości genów."""
        mask = self.rng.random(dna.shape) < config.MUTATION_RATE
        noise = self.rng.normal(0.0, sigma, size=dna.shape)
        dna = dna + mask * noise
        return np.clip(dna, -config.GENE_CLIP, config.GENE_CLIP)

    def _random_dna(self) -> np.ndarray:
        """Generuje losowe DNA w umiarkowanym zakresie."""
        return self.rng.uniform(-1.0, 1.0, size=config.DNA_SIZE)

    # --- Logika Przeciwników (Anchors) ---
    def _build_anchor_agents(self) -> List[GeneticAgent]:
        """Deterministyczny zestaw kotwic: baseline + champion + top HoF (unikalne)."""
        desired = max(1, int(getattr(self, "anchors_count", getattr(config, "ANCHORS_COUNT", 4))))
        pool: List[np.ndarray] = []

        # 1) wszystkie baseline’y (w kolejności), żeby mieć z czego uzupełniać
        for row in getattr(config, 'BASELINE_DNA_POOL', []):
            arr = self._adapt_dna_to_size(np.array(row, dtype=float))
            if arr.shape[0] == config.DNA_SIZE:
                pool.append(arr)

        # 2) champion
        if self.champion is not None:
            pool.append(self.champion.copy())

        # 3) cała HoF (posortowana po fitness – już jest)
        for dna, _fit in self.hof:
            pool.append(dna.copy())

        # 4) zbuduj listę UNIKALNYCH do osiągnięcia desired
        unique: List[np.ndarray] = []
        for d in pool:
            if not any(np.allclose(d, u, atol=1e-8) for u in unique):
                unique.append(d)
                if len(unique) == desired:
                    break

        # jeśli wciąż brakuje, dopychaj zerową kotwicą (awaryjnie)
        while len(unique) < desired:
            unique.append(np.zeros(config.DNA_SIZE))

        agents = [GeneticAgent(dna=d.copy()) for d in unique]
        for i, ag in enumerate(agents):
            ag.name = f"ANCHOR_{i+1}"
        return agents
    
    # --- Logika Ewaluacji (Serce Systemu) ---

    def _play_one_game(self, dnaA: np.ndarray, agentB_template: GeneticAgent, seed: int, swap_roles: bool) -> Tuple[int, int, int, int]:
        """
        Rozgrywa jedną grę 1v1. Zwraca: (winner_idx, pointsA, pointsB, turns)
        winner_idx: 0 = Candidate, 1 = Anchor, -1 = remis
        """
        target_points = getattr(config, "TARGET_PRESTIGE_POINTS", 15)
        if getattr(config, "DIFFERENT_SEED_FOR_SWAP", True) and swap_roles:
            game_seed = seed ^ getattr(config, "SWAP_ROLES_SEED_XOR", 0x09E3779B)
            agent_base_seed = game_seed
        else:
            game_seed = seed
            agent_base_seed = seed

        game = Game(all_cards=self.all_cards, all_nobles=self.all_nobles,
                    seed=game_seed, target_points=target_points)

        if not swap_roles:
            cand_seed = agent_base_seed + getattr(config, "AGENT_SEED_CAND_OFFSET", 101)
            anch_seed = agent_base_seed + getattr(config, "AGENT_SEED_ANCH_OFFSET", 313)
            agentA = GeneticAgent(dna=dnaA.copy(), rng_seed=cand_seed); agentA.name = "Candidate"
            agentB = GeneticAgent(dna=agentB_template.dna.copy(), rng_seed=anch_seed); agentB.name = "Anchor"
            game.setup_new_game(player_names=["Candidate", "Anchor"])
            players = [agentA, agentB]
        else:
            cand_seed = agent_base_seed + getattr(config, "AGENT_SEED_CAND_OFFSET", 101)
            anch_seed = agent_base_seed + getattr(config, "AGENT_SEED_ANCH_OFFSET", 313)
            agentA = GeneticAgent(dna=agentB_template.dna.copy(), rng_seed=anch_seed); agentA.name = "Anchor"
            agentB = GeneticAgent(dna=dnaA.copy(), rng_seed=cand_seed); agentB.name = "Candidate"
            game.setup_new_game(player_names=["Anchor", "Candidate"])
            players = [agentA, agentB]

        max_turns = int(getattr(config, "MAX_TURNS", 150))
        turns = 0
        while not game.is_game_over() and turns < max_turns:
            state = game.game_state
            current_player = players[state.current_player_index]

            valid_moves = game.get_valid_moves()
            if not valid_moves:
                game.finalize_turn(None)
                turns += 1
                continue

            move, chosen_noble = current_player.choose_action(state, valid_moves)
            eligible_nobles = game.apply_move(move)
            # SSOT: tylko kwalifikowalny arystokrata może być przekazany do finalize_turn
            if chosen_noble not in eligible_nobles:
                chosen_noble = None
            game.finalize_turn(chosen_noble)
            turns += 1

        winner_obj = game.get_winner()
        p0_score = game.game_state.players[0].prestige_points
        p1_score = game.game_state.players[1].prestige_points

        pointsA = p0_score if not swap_roles else p1_score  # Candidate
        pointsB = p1_score if not swap_roles else p0_score  # Anchor

        if winner_obj is None:
            winner_idx = -1
        else:
            winner_idx = 0 if winner_obj.name == "Candidate" else 1

        return winner_idx, pointsA, pointsB, turns

    def _evaluate_candidate_with_params(self, dna: np.ndarray, anchors: List[GeneticAgent],
                                    gen_idx: int, seeds_per_anchor: int) -> Dict[str, float]:
        """Sekwencyjna ewaluacja jednego kandydata z podaną liczbą seedów."""
        return self._evaluate_candidate(dna, anchors, gen_idx, seeds_per_anchor)

    def _evaluate_population_stream(self, population, anchors, gen_idx):
        from multiprocessing import Pool, cpu_count
        anchors_dna = [a.dna.copy() for a in anchors]
        cfg = dict(
            seed_base=config.SEED_BASE,
            seeds_per_anchor=config.EVAL_SEEDS_PER_ANCHOR,
            target_points=getattr(config, "TARGET_PRESTIGE_POINTS", 15),
            different_seed_for_swap=getattr(config, "DIFFERENT_SEED_FOR_SWAP", True),
            swap_seed_xor=getattr(config, "SWAP_ROLES_SEED_XOR", 0x9E3779B),
            agent_seed_cand_offset=getattr(config, "AGENT_SEED_CAND_OFFSET", 101),
            agent_seed_anch_offset=getattr(config, "AGENT_SEED_ANCH_OFFSET", 313),
            turns_baseline=getattr(config, "TURNS_BASELINE", 60.0),
            turns_penalty_weight=getattr(config, "TURNS_PENALTY_WEIGHT", 1.2),
            speed_bonus_weight=getattr(config, "SPEED_BONUS_WEIGHT", 0.8),
            max_turns=getattr(config, "MAX_TURNS", 150),
            wilson_z=getattr(config, "WILSON_Z", 1.96),
        )
        tasks = [(i, population[i].tolist(), [ad.tolist() for ad in anchors_dna], gen_idx, cfg)
                for i in range(len(population))]

        results = [None] * len(population)
        n_proc = config.N_PROCESSES or max(1, cpu_count()-1)
        chunksize = getattr(config, "CHUNK_SIZE", None) or 1

        with Pool(processes=n_proc, initializer=_mp_init_worker, initargs=(self.all_cards, self.all_nobles)) as pool:
            bar = tqdm(total=len(population), desc=f"Eval G{gen_idx+1}", leave=False, dynamic_ncols=True)
            for idx, met in pool.imap_unordered(_evaluate_candidate_worker_idx, tasks, chunksize=chunksize):
                results[idx] = met
                bar.update(1)
            bar.close()

        return results

    def _evaluate_candidate(self, dna: np.ndarray, anchors: List[GeneticAgent], gen_idx: int,
                        seeds_per_anchor: Optional[int] = None) -> Dict[str, float]:
        """Ewaluacja jednego kandydata vs anchory. Remisy liczone jako 0.5."""
        seeds_per_anchor = seeds_per_anchor or config.EVAL_SEEDS_PER_ANCHOR
        seeds = [int(config.SEED_BASE + 7919*gen_idx + 97*i) for i in range(seeds_per_anchor)]

        wins = 0.0; games = 0; pts_sum = 0.0; margin_sum = 0.0; turns_sum = 0.0
        for anchor in anchors:
            for s in seeds:
                for swap in (False, True):
                    w_idx, pA, pB, t = self._play_one_game(dna, anchor, seed=s, swap_roles=swap)
                    wins += (0.5 if w_idx == -1 else (1.0 if w_idx == 0 else 0.0))
                    games += 1
                    pts_sum += pA
                    margin_sum += (pA - pB)
                    turns_sum += t

        if games == 0:
            return dict(win_rate=0.0, avg_points=0.0, avg_margin=0.0, avg_turns=0.0, fitness=0.0)

        win_rate   = wins / games
        avg_points = pts_sum / games
        avg_margin = margin_sum / games
        avg_turns  = turns_sum / games

        margin_norm = max(-1.0, min(1.0, avg_margin / 15.0))
        points_norm = max(0.0, min(1.0, avg_points / 15.0))
        turns_baseline = float(getattr(config, "TURNS_BASELINE", 60.0))
        turns_penalty  = max(0.0, (avg_turns - turns_baseline) / turns_baseline)
        speed_bonus    = max(0.0, (turns_baseline - avg_turns) / turns_baseline)

        fitness = (12.0 * win_rate) \
                + (3.0 * margin_norm) \
                + (1.5 * points_norm) \
                - (getattr(config, "TURNS_PENALTY_WEIGHT", 1.2) * turns_penalty) \
                + (getattr(config, "SPEED_BONUS_WEIGHT", 0.8) * points_norm * speed_bonus)

        wr_lb = _wilson_lb(wins, games, z=getattr(config, "WILSON_Z", 1.96))
        return dict(
            win_rate=win_rate, avg_points=avg_points, avg_margin=avg_margin, avg_turns=avg_turns,
            wins=wins, games=games, wr_lb=wr_lb, fitness=fitness
        )
    
    def _evaluate_population(self, population: List[np.ndarray], anchors: List[GeneticAgent], gen_idx: int):
        if not getattr(config, "USE_MULTIPROCESSING", False):
            return [self._evaluate_candidate(d, anchors, gen_idx, config.EVAL_SEEDS_PER_ANCHOR) for d in population]

        anchors_dna = [a.dna.copy() for a in anchors]
        cfg = dict(
            seed_base=config.SEED_BASE,
            seeds_per_anchor=config.EVAL_SEEDS_PER_ANCHOR,
            target_points=getattr(config, "TARGET_PRESTIGE_POINTS", 15),
            different_seed_for_swap=getattr(config, "DIFFERENT_SEED_FOR_SWAP", True),
            swap_seed_xor=getattr(config, "SWAP_ROLES_SEED_XOR", 0x9E3779B),
            agent_seed_cand_offset=getattr(config, "AGENT_SEED_CAND_OFFSET", 101),
            agent_seed_anch_offset=getattr(config, "AGENT_SEED_ANCH_OFFSET", 313),
            turns_baseline=getattr(config, "TURNS_BASELINE", 60.0),
            turns_penalty_weight=getattr(config, "TURNS_PENALTY_WEIGHT", 1.2),
            speed_bonus_weight=getattr(config, "SPEED_BONUS_WEIGHT", 0.8),
            max_turns=getattr(config, "MAX_TURNS", 150),
            wilson_z=getattr(config, "WILSON_Z", 1.96),
        )
        tasks = [(d.tolist(), [ad.tolist() for ad in anchors_dna], gen_idx, cfg) for d in population]
        n_proc = config.N_PROCESSES or max(1, cpu_count() - 1)
        chunksize = getattr(config, "CHUNK_SIZE", None) or 1
        with Pool(processes=n_proc, initializer=_mp_init_worker, initargs=(self.all_cards, self.all_nobles)) as pool:
            return pool.map(_evaluate_candidate_worker, tasks, chunksize=chunksize)
        
    # --- Główna Pętla Ewolucji ---
    
    def run_evolution(self):
        """Główna pętla algorytmu genetycznego."""
        print("====== Rozpoczynanie Procesu Ewolucji (Architektura 'GENIUSZ') ======")

        # Inicjalizacja populacji
        population = [self._random_dna() for _ in range(config.POP_SIZE)]
        try:  # Spróbuj załadować istniejącego championa
            if os.path.exists("champion_agent.npy"):
                base_dna = np.load("champion_agent.npy")
                base_dna = self._adapt_dna_to_size(base_dna)
                if base_dna.shape[0] == config.DNA_SIZE:
                    population[0] = base_dna.copy()
                    self.champion = base_dna.copy()
                    print("Znaleziono i załadowano istniejącego championa jako punkt startowy.")
        except Exception as e:
            print(f"Nie udało się załadować championa: {e}")

        # Setup logowania
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evolution_log_{timestamp}.csv"
        logs_path = os.path.join(log_dir, filename)
        print(f"Statystyki ewolucji będą zapisywane do: {logs_path}")
        os.makedirs(os.path.dirname(logs_path), exist_ok=True)
        header = ["generation","best_fitness","avg_fitness","worst_fitness","std_dev_fitness",
                "best_win_rate","avg_win_rate","avg_game_score","avg_game_turns",
                "elite_dna_sample","time_elapsed_sec","sigma","games_per_candidate"]
        with open(logs_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

        # Pasek dla generacji
        gen_iter = tqdm(range(config.GENERATIONS), desc="Generations", dynamic_ncols=True) \
                if getattr(config, "SHOW_PROGRESS", False) else range(config.GENERATIONS)

        for g in gen_iter:
            t0 = time.time()
            sigma = self._sigma_at(g, config.GENERATIONS)
            msg = f">>> Generacja {g+1}/{config.GENERATIONS} | Sigma mutacji: {sigma:.3f}"
            if getattr(config, "SHOW_PROGRESS", False):
                tqdm.write(msg)
            else:
                print("\n" + msg)

            anchors = self._build_anchor_agents()
            info = f"Wybrano {len(anchors)} kotwic do ewaluacji tej generacji."
            if getattr(config, "SHOW_PROGRESS", False):
                tqdm.write(info)
            else:
                print(info)

            # EWALUACJA populacji z paskiem postępu
            if getattr(config, "SHOW_PROGRESS", False) and getattr(config, "USE_MULTIPROCESSING", False) and hasattr(self, "_evaluate_population_stream"):
                # wariant streamingowy dla MP (imap_unordered)
                metrics = self._evaluate_population_stream(population, anchors, gen_idx=g)
            elif getattr(config, "SHOW_PROGRESS", False) and not getattr(config, "USE_MULTIPROCESSING", False):
                metrics = []
                eval_bar = tqdm(total=len(population), desc=f"Eval G{g+1}", leave=False, dynamic_ncols=True)
                for dna in population:
                    met = self._evaluate_candidate(dna, anchors, gen_idx=g)
                    metrics.append(met)
                    eval_bar.update(1)
                eval_bar.close()
            else:
                # dotychczasowa ścieżka (także multiprocessing bez paska)
                metrics = self._evaluate_population(population, anchors, gen_idx=g)

            # 1) Wyciągnij metryki z ewaluacji
            fitness    = [m["fitness"]     for m in metrics]     # tylko do logów
            win_rates  = [m["win_rate"]    for m in metrics]
            avg_pts    = [m["avg_points"]  for m in metrics]
            avg_turns  = [m["avg_turns"]   for m in metrics]
            avg_margin = [m["avg_margin"]  for m in metrics]
            wr_lb_list = [m.get("wr_lb", m["win_rate"]) for m in metrics]  # fallback, gdyby wr_lb brak

            # 2) Klucz leksykograficzny: (WR_LB, −tury, margines, punkty)
            rank_keys = [(wr_lb_list[i], -avg_turns[i], avg_margin[i], avg_pts[i]) for i in range(len(population))]

            # --- Schedule trudności (zostawiasz jak masz) ---
            avg_wr = float(np.mean(win_rates))
            if getattr(config, "ANCHOR_SCHEDULE_ENABLED", True):
                base_count = getattr(config, "ANCHORS_COUNT", 4)
                max_count  = getattr(config, "ANCHORS_MAX", 5)
                up_thr     = getattr(config, "ANCHOR_SCHEDULE_UP_THRESHOLD", 0.65)
                down_thr   = getattr(config, "ANCHOR_SCHEDULE_DOWN_THRESHOLD", 0.50)
                streak     = getattr(config, "ANCHOR_SCHEDULE_STREAK", 3)

                if avg_wr > up_thr:
                    self._hi_winrate_streak += 1; self._lo_winrate_streak = 0
                elif avg_wr < down_thr:
                    self._lo_winrate_streak += 1; self._hi_winrate_streak = 0
                else:
                    self._hi_winrate_streak = 0; self._lo_winrate_streak = 0

                if self._hi_winrate_streak >= streak and self.anchors_count < max_count:
                    self.anchors_count = max_count
                    (tqdm.write if getattr(config, "SHOW_PROGRESS", False) else print)(
                        f"[Schedule] ANCHORS_COUNT -> {self.anchors_count} (avg_win_rate={avg_wr:.2f})"
                    )
                if self._lo_winrate_streak >= streak and self.anchors_count > base_count:
                    self.anchors_count = base_count
                    (tqdm.write if getattr(config, "SHOW_PROGRESS", False) else print)(
                        f"[Schedule] ANCHORS_COUNT -> {self.anchors_count} (avg_win_rate={avg_wr:.2f})"
                    )

            # 3) Re-score top‑N (jeśli włączony) – UŻYJ rank_keys do wyboru topki i po pętli przelicz rank_keys
            if getattr(config, "USE_TOP_RESCORE", False) and config.TOP_RESCORE_N > 0:
                pre_idx_sorted = sorted(range(len(population)), key=lambda i: rank_keys[i], reverse=True)
                topN = min(config.TOP_RESCORE_N, len(population))
                rescore_bar = None
                if getattr(config, "SHOW_PROGRESS", False):
                    rescore_bar = tqdm(total=topN, desc=f"Rescore G{g+1}", leave=False, dynamic_ncols=True)
                for k in range(topN):
                    i = pre_idx_sorted[k]
                    met = self._evaluate_candidate_with_params(population[i], anchors, gen_idx=g,
                                                            seeds_per_anchor=config.EVAL_SEEDS_PER_ANCHOR_TOP)
                    metrics[i]    = met
                    fitness[i]    = met["fitness"]
                    win_rates[i]  = met["win_rate"]
                    avg_pts[i]    = met["avg_points"]
                    avg_turns[i]  = met["avg_turns"]
                    avg_margin[i] = met["avg_margin"]
                    wr_lb_list[i] = met.get("wr_lb", met["win_rate"])
                    if rescore_bar: rescore_bar.update(1)
                if rescore_bar: rescore_bar.close()
                # przelicz klucz po re-score
                rank_keys = [(wr_lb_list[i], -avg_turns[i], avg_margin[i], avg_pts[i]) for i in range(len(population))]

            # 4) Sortowanie populacji po kluczu leksykograficznym
            idx_sorted = sorted(range(len(population)), key=lambda i: rank_keys[i], reverse=True)
            population = [population[i] for i in idx_sorted]
            fitness    = [fitness[i]    for i in idx_sorted]   # tylko do logów
            win_rates  = [win_rates[i]  for i in idx_sorted]
            avg_pts    = [avg_pts[i]    for i in idx_sorted]
            avg_turns  = [avg_turns[i]  for i in idx_sorted]
            avg_margin = [avg_margin[i] for i in idx_sorted]
            wr_lb_list = [wr_lb_list[i] for i in idx_sorted]
            rank_keys  = [rank_keys[i]  for i in idx_sorted]

            # USTAW best_dna/best_fit (skasowałeś to wcześniej komentarzem)
            best_dna = population[0].copy()
            best_fit = float(fitness[0])

            # (opcjonalnie) loguj WR_LB najlepszego
            best_wr_lb = wr_lb_list[0]
            if getattr(config, "SHOW_PROGRESS", False):
                tqdm.write(f"Best WR_LB: {best_wr_lb:.3f}")
            else:
                print(f"Best WR_LB: {best_wr_lb:.3f}")

            # 5) Aktualizacja Championa i HoF (jak było)
            self.champion = best_dna.copy()
            self._hof_add(best_dna, best_fit)

            if (g + 1) % 5 == 0:
                self._save_hof()

            # 6) Log CSV (bez zmian)
            games_per_candidate = len(anchors) * config.EVAL_SEEDS_PER_ANCHOR * 2
            with open(logs_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    g+1,
                    round(float(np.max(fitness)), 4),
                    round(float(np.mean(fitness)), 4),
                    round(float(np.min(fitness)), 4),
                    round(float(np.std(fitness)), 4),
                    round(float(np.max(win_rates)), 4),
                    round(float(np.mean(win_rates)), 4),
                    round(float(np.mean(avg_pts)), 4),
                    round(float(np.mean(avg_turns)), 4),
                    np.array2string(best_dna[:4], precision=2, separator=' '),
                    round(time.time() - t0, 2),
                    round(sigma, 4),
                    games_per_candidate
                ])

            # 7) Tworzenie nowej populacji – turniej po rank_keys
            new_pop: List[np.ndarray] = [dna.copy() for dna in population[:config.ELITE_COUNT]]
            while len(new_pop) < config.POP_SIZE - config.IMMIGRANTS:
                p1 = self._tournament_pick(population, rank_keys)
                p2 = self._tournament_pick(population, rank_keys)
                child = self._crossover(p1, p2)
                child = self._mutate(child, sigma)
                new_pop.append(child)
            while len(new_pop) < config.POP_SIZE:
                new_pop.append(self._random_dna())
            population = new_pop

        # Finalne zapisanie championa i HoF
        if self.champion is not None:
            np.save("champion_agent.npy", self.champion)
        self._save_hof()
        print("\n====== Proces Ewolucji Zakończony ======")
        print("Zapisano ostateczne DNA mistrza do champion_agent.npy oraz HoF do hof.json")