# es_manager.py (wersja z hold-out)
import os, csv, time, math
from typing import List
import numpy as np
from multiprocessing import Pool, cpu_count
import config
import nn_config as NN
from nn_agent import NnAgent, param_count
from game_engine import Game
from agents import GeneticAgent  # kotwice DNA GA
from nn_agent import NnAgent, param_count
# es_manager.py (dopisz przy importach)
import random  # deterministyczny tie-break dla GA, jeśli używa global RNG
import game_engine as ge  # do typów ruchów i GemColor

COLOR_ORDER = [ge.GemColor.WHITE, ge.GemColor.BLUE, ge.GemColor.GREEN, ge.GemColor.RED, ge.GemColor.BLACK]
COLOR_TO_IDX = {c: i for i, c in enumerate(COLOR_ORDER)}

def _move_sort_key(m: ge.Move):
    # Buy < ReserveVisible < ReserveFromDeck < TakeThree < TakeTwo
    if isinstance(m, ge.BuyCard):
        return (0, m.card.level, m.card.id)
    if isinstance(m, ge.ReserveVisibleCard):
        return (1, m.card.level, m.card.id)
    if isinstance(m, ge.ReserveFromDeck):
        return (2, m.tier, 0)
    if isinstance(m, ge.TakeThreeGems):
        cols = tuple(sorted((COLOR_TO_IDX[c] for c in m.colors)))
        return (3, cols)
    if isinstance(m, ge.TakeTwoGems):
        return (4, COLOR_TO_IDX[m.color])
    return (99, repr(m))

def _canonicalize_moves(moves: list[ge.Move]) -> list[ge.Move]:
    return sorted(moves, key=_move_sort_key)

def _choose_noble_deterministic(eligible: list[ge.Noble]):
    if not eligible:
        return None
    return sorted(eligible, key=lambda n: (-n.prestige_points, n.id))[0]

def _wilson_lb(wins, games, z=1.96):
    if games <= 0: return 0.0
    p = wins/games
    denom = 1.0 + (z*z)/games
    centre = p + (z*z)/(2.0*games)
    margin = z * math.sqrt((p*(1.0-p))/games + (z*z)/(4.0*games*games))
    return max(0.0, min(1.0, (centre - margin)/denom))

class ESManager:
    def __init__(self, all_cards, all_nobles):
        self.all_cards = all_cards
        self.all_nobles = all_nobles
        # zapisz architekturę z nn_config
        self.h1 = int(getattr(NN, "HIDDEN1", 64))
        self.h2 = int(getattr(NN, "HIDDEN2", 64))
        self.act = str(getattr(NN, "ACT", "silu"))

        self.dim = param_count(h1=self.h1, h2=self.h2)
        self.rng = np.random.default_rng(NN.SEED_BASE)
        self.theta = self.rng.normal(0, 0.05, size=self.dim)
        self.sigma = NN.ES_SIGMA
        self.anchors = self._build_anchors()
        print(f"[ES] dim={self.dim}, sigma={self.sigma}, pop={NN.ES_POP}")

    # --- Seedy treningowe i hold-out ---
    def _train_seeds(self) -> List[int]:
        # zachowujemy Twoją siatkę seedów (CRN przez role via swap)
        return [int(NN.SEED_BASE + 7919*i) for i in range(NN.EVAL_SEEDS_PER_ANCHOR)]

    def _holdout_seeds(self) -> List[int]:
        # stały, odseparowany od treningu strumień (inna baza + inny krok)
        base = int(getattr(NN, "SEED_BASE", 42)) ^ 0xC0FFEE
        count = int(getattr(NN, "HOLDOUT_SEEDS_PER_ANCHOR", 6))
        return [int((base + 104729*i) & 0x7FFFFFFF) for i in range(count)]

    def _eval_theta(self, theta: np.ndarray, seeds: List[int] = None, anchors: List[GeneticAgent] = None):
        """
        Ewaluacja vs kotwice:
          - jeśli seeds=None -> użyj seedy treningowe,
          - jeśli anchors=None -> zbuduj kotwice przez _build_anchors().
        Zwraca: dict(score, wr_lb, wr, avg_points, avg_turns)
        """
        #anchors = anchors if anchors is not None else self._build_anchors()
        anchors = anchors if anchors is not None else self.anchors
        seeds = seeds if seeds is not None else self._train_seeds()

        wins=0.0; games=0; turns_sum=0.0; pts_sum=0.0; margin_sum=0.0
        for a in anchors:
            for s in seeds:
                for swap in (False, True):
                    w, pa, pb, t = self._play(theta, a, seed=s, swap=swap)
                    # remis=0.5, wygrana NN=1, porażka=0
                    wins += (0.5 if w == -1 else (1.0 if w == 0 else 0.0))
                    games += 1
                    turns_sum += t
                    pts_sum += pa
                    margin_sum += (pa - pb)

        if games == 0:
            return dict(score=0.0, wr_lb=0.0, wr=0.0, avg_points=0.0, avg_turns=0.0)

        wr = wins/games
        wr_lb = _wilson_lb(wins, games, z=getattr(config, "WILSON_Z", 1.96))

        avg_turns = turns_sum/games
        avg_points = pts_sum/games
        ppt = avg_points/max(1.0, avg_turns)

        reward = wr_lb + getattr(NN, "PPT_WEIGHT", 0.05) * min(1.0, ppt/0.30)
        return dict(score=float(reward), wr_lb=float(wr_lb), wr=float(wr),
                    avg_points=float(avg_points), avg_turns=float(avg_turns))

    def _play(self, theta: np.ndarray, anchor: GeneticAgent, seed: int, swap: bool):
        # per‑game determinism (dla ewentualnych RNG używanych przez GA)
        tie_seed = (seed ^ 0x9E3779B1 ^ (1 if swap else 0)) & 0xFFFFFFFF
        random.seed(tie_seed)
        np.random.seed(tie_seed)

        game = Game(all_cards=self.all_cards, all_nobles=self.all_nobles, seed=seed,
                    target_points=getattr(config,"TARGET_PRESTIGE_POINTS",15))
        A = NnAgent(theta, hidden1=self.h1, hidden2=self.h2, act=self.act, rng_seed=seed+101, name="NN")
        B = anchor
        if not swap:
            game.setup_new_game(["NN","ANCH"])
            players = [A,B]
        else:
            game.setup_new_game(["ANCH","NN"])
            players = [B,A]

        turns=0
        max_turns = int(getattr(config,"MAX_TURNS",150))
        while not game.is_game_over() and turns<max_turns:
            st = game.game_state
            ag = players[st.current_player_index]
            valid = _canonicalize_moves(game.get_valid_moves())  # stabilny porządek
            if not valid:
                game.finalize_turn(None); turns+=1; continue
            mv, _ = ag.choose_action(st, valid)
            elig = game.apply_move(mv)
            chosen = _choose_noble_deterministic(elig)  # deterministyczny wybór
            game.finalize_turn(chosen); turns+=1

        winner = game.get_winner()
        p0 = game.game_state.players[0].prestige_points
        p1 = game.game_state.players[1].prestige_points
        if winner is None:
            w_idx = -1
        else:
            w_idx = 0 if winner.name=="NN" else 1
        pa = p0 if not swap else p1
        pb = p1 if not swap else p0
        return (w_idx, pa, pb, turns)

    def _build_anchors(self) -> List[GeneticAgent]:
        anchors=[]
        for row in getattr(config,"BASELINE_DNA_POOL",[]):
            try:
                anchors.append(GeneticAgent(dna=np.array(row,dtype=float)))
            except: pass
        try:
            if os.path.exists("champion_agent.npy"):
                dna = np.load("champion_agent.npy")
                anchors.append(GeneticAgent(dna=dna))
        except: pass
        if not anchors:
            anchors = [GeneticAgent(dna=np.zeros(getattr(config,"DNA_SIZE",20)))]
        return anchors[:max(1, getattr(NN, "ANCHORS_COUNT", 4))]

    def run(self, generations=200, theta_init=None):
        # ewentualne wznowienie od istniejącej thety
        if theta_init is not None:
            ti = np.asarray(theta_init)
            if ti.size != self.dim:
                print(f"[ES] Ostrzeżenie: theta_init ma rozmiar {ti.size}, oczekiwano {self.dim}. Ignoruję resume.")
            else:
                self.theta = ti.astype(np.float32, copy=True)
                print(f"[ES] Resume: załadowano theta_init (len={ti.size}).")

        logs_dir="logs"; os.makedirs(logs_dir, exist_ok=True)
        path=os.path.join(logs_dir,"es_log.csv")
        with open(path,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(["gen","score","wr_lb","wr","avg_points","avg_turns","sigma","time_sec"])        
        os.makedirs(getattr(NN, "SNAPSHOT_DIR", "snapshots"), exist_ok=True)

        # Snapshoty: przygotuj katalog
        snap_every = int(getattr(NN, "SNAPSHOT_EVERY", 0))
        snap_dir = str(getattr(NN, "SNAPSHOT_DIR", "snapshots"))
        if snap_every > 0:
            os.makedirs(snap_dir, exist_ok=True)
            # zapis stanu startowego (po init/resume)
            np.save(os.path.join(snap_dir, "theta_gen_0000.npy"), self.theta)

        # hold-out log (opcjonalnie)
        hold_enabled = bool(getattr(NN, "USE_HOLDOUT", False))
        hold_path = getattr(NN, "HOLDOUT_LOG_CSV", os.path.join(logs_dir, "es_holdout_log.csv"))
        if hold_enabled:
            os.makedirs(os.path.dirname(hold_path), exist_ok=True)
            with open(hold_path,"w",newline="",encoding="utf-8") as f:
                csv.writer(f).writerow(["gen","score","wr_lb","wr","avg_points","avg_turns"])
        
        best_score = -float('inf')
        best_hold_lb = -float('inf')  # Śledzimy najlepszy WR_LB na hold-out
        print(f"[ES] Hold-out set: {len(self._holdout_seeds())} seeds vs {len(self.anchors)} anchors (gry: {len(self._holdout_seeds()) * len(self.anchors) * 2})")
        
        for g in range(generations):
            t0=time.time()
            # mirrored sampling
            eps = self.rng.normal(0,1,size=(NN.ES_POP, self.dim))
            # ewaluacja thet ± sigma*e
            thetas_plus  = [self.theta + self.sigma*e for e in eps]
            thetas_minus = [self.theta - self.sigma*e for e in eps]
            evals_plus  = [self._eval_theta(th) for th in thetas_plus]
            evals_minus = [self._eval_theta(th) for th in thetas_minus]

            # różnice
            rewards_plus  = [e["score"] for e in evals_plus]
            rewards_minus = [e["score"] for e in evals_minus]
            r = np.array(rewards_plus) - np.array(rewards_minus)
            r = (r - r.mean()) / (r.std()+1e-8)

            grad = np.tensordot(r, eps, axes=1) / (NN.ES_POP)  # (dim,)
            self.theta = self.theta + (NN.ES_LR / self.sigma) * grad

            # ewaluacja aktualnej theta
            met = self._eval_theta(self.theta)
            # log do CSV (spójne kolumny)
            with open(path,"a",newline="",encoding="utf-8") as f:
                csv.writer(f).writerow([
                    g+1,
                    round(met["score"], 4),
                    round(met["wr_lb"], 4),
                    round(met["wr"], 4),
                    round(met.get("avg_points", 0.0), 3),
                    round(met.get("avg_turns", 0.0), 3),
                    round(self.sigma, 4),
                    round(time.time()-t0, 2),
                ])
            print(f"[ES] G{g+1}: WR_LB={met['wr_lb']:.3f} WR={met['wr']:.3f} score={met['score']:.3f} time={time.time()-t0:.2f}s")

            # Hold-out (opcjonalnie)
            hold_met = None
            if hold_enabled:
                hold_met = self._eval_theta(self.theta, seeds=self._holdout_seeds())
                with open(hold_path,"a",newline="",encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        g+1,
                        round(hold_met["score"],4),
                        round(hold_met["wr_lb"],4),
                        round(hold_met["wr"],4),
                        round(hold_met.get("avg_points",0.0),3),
                        round(hold_met.get("avg_turns",0.0),3),
                    ])

            # Anneal
            if getattr(NN,"ANNEAL_SIGMA",True):
                self.sigma = max(0.03, self.sigma*0.995)

            # Zapis championa — re-score na hold-out (jeśli włączony)
            save_champion = False
            if hold_enabled and hold_met is not None:
                if hold_met["wr_lb"] > best_hold_lb:
                    best_hold_lb = hold_met["wr_lb"]
                    save_champion = True
            elif met["score"] > best_score:
                save_champion = True

            if save_champion:
                best_score = max(best_score, met["score"])
                np.save("nn_champion.npy", self.theta)
                print(f"    -> Nowy champion zapisany (score={met['score']:.3f}"
                    f"{', hold_wr_lb='+str(round(hold_met['wr_lb'],3)) if hold_met else ''})")

            # Snapshot co N generacji (zachowaj to, co już masz)
            if snap_every > 0 and ((g + 1) % snap_every == 0):
                np.save(os.path.join(snap_dir, f"theta_gen_{g+1:04d}.npy"), self.theta)

        np.save("nn_champion.npy", self.theta)
        print("Trening zakończony.")
        print("Model: nn_champion.npy | Log: logs/es_log.csv")
        if hold_enabled:
            print(f"Hold-out log: {hold_path}")

        snap_every = int(getattr(NN, "SNAPSHOT_EVERY", 0))
        if snap_every > 0 and ((g % snap_every) == 0 or g == 1):
            snap_path = os.path.join(getattr(NN, "SNAPSHOT_DIR", "snapshots"), f"theta_gen_{g:04d}.npy")
            np.save(snap_path, self.theta)