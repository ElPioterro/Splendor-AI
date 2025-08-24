import os, csv, time, math
import numpy as np
from multiprocessing import Pool, cpu_count
import config
import nn_config as NN
from nn_agent import NnAgent, param_count
from game_engine import Game
from agents import GeneticAgent  # żeby użyć Twoich anchorów DNA jeśli chcesz; albo przygotuj baseline NN

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
        self.dim = param_count()
        self.rng = np.random.default_rng(NN.SEED_BASE)
        self.theta = self.rng.normal(0, 0.05, size=self.dim)  # start
        self.sigma = NN.ES_SIGMA
        print(f"[ES] dim={self.dim}, sigma={self.sigma}, pop={NN.ES_POP}")

    def _eval_theta(self, theta: np.ndarray):
        # ocena vs kotwice – 4 anchory, 2 seedy, 2 role => 16 gier
        anchors = self._build_anchors()
        wins=0.0; games=0; turns_sum=0.0; pts_sum=0.0; margin_sum=0.0
        seeds = [int(NN.SEED_BASE + 7919*i) for i in range(NN.EVAL_SEEDS_PER_ANCHOR)]
        for a in anchors:
            for s in seeds:
                for swap in (False, True):
                    wr = self._play(theta, a, seed=s, swap=swap)
                    w, pa, pb, t = wr
                    wins += (0.5 if w==-1 else (1 if w==0 else 0))
                    games += 1; turns_sum += t; pts_sum += pa; margin_sum += (pa-pb)
        if games==0: return dict(score=0.0, wr_lb=0.0, wr=0.0)
        wr = wins/games
        wr_lb = _wilson_lb(wins, games, z=getattr(config, "WILSON_Z", 1.96))
        # scalar reward dla ES: głównie WR_LB + lekki PPT
        avg_turns = turns_sum/games
        avg_points = pts_sum/games
        ppt = avg_points/max(1.0, avg_turns)
        reward = wr_lb + getattr(NN, "PPT_WEIGHT", 2.0) * min(1.0, ppt/0.30)
        return dict(score=float(reward), wr_lb=float(wr_lb), wr=float(wr))

    def _play(self, theta: np.ndarray, anchor: GeneticAgent, seed: int, swap: bool):
        game = Game(all_cards=self.all_cards, all_nobles=self.all_nobles, seed=seed,
                    target_points=getattr(config,"TARGET_PRESTIGE_POINTS",15))
        A = NnAgent(theta, rng_seed=seed+101, name="NN")
        B = anchor  # przeciwnik kotwica (możesz też zrobić NnAgent baseline)
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
            valid = game.get_valid_moves()
            if not valid:
                game.finalize_turn(None); turns+=1; continue
            mv, nob = ag.choose_action(st, valid)
            elig = game.apply_move(mv)
            if nob not in elig: nob=None
            game.finalize_turn(nob); turns+=1
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
        # na start kotwice z Waszego BASELINE_DNA_POOL + champion z dysku (jeśli jest)
        anchors=[]
        for row in getattr(config,"BASELINE_DNA_POOL",[]):
            try:
                anchors.append(GeneticAgent(dna=np.array(row,dtype=float)))
            except: pass
        # możesz też wczytać waszego championa GA:
        try:
            if os.path.exists("champion_agent.npy"):
                dna = np.load("champion_agent.npy")
                anchors.append(GeneticAgent(dna=dna))
        except: pass
        if not anchors:
            anchors = [GeneticAgent(dna=np.zeros(getattr(config,"DNA_SIZE",20)))]
        return anchors[:max(1, getattr(NN, "ANCHORS_COUNT", 4))]

    def run(self, generations=200):
        logs_dir="logs"; os.makedirs(logs_dir, exist_ok=True)
        path=os.path.join(logs_dir,"es_log.csv")
        with open(path,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(["gen","score","wr_lb","wr","sigma","time_sec"])
        for g in range(generations):
            t0=time.time()
            # mirrored sampling
            eps = self.rng.normal(0,1,size=(NN.ES_POP, self.dim))
            rewards_plus=[]; rewards_minus=[]
            # ewaluacja równoległa
            thetas_plus  = [self.theta + self.sigma*e for e in eps]
            thetas_minus = [self.theta - self.sigma*e for e in eps]
            evals = [self._eval_theta(th) for th in thetas_plus] + [self._eval_theta(th) for th in thetas_minus]
            rewards_plus  = [e["score"] for e in evals[:NN.ES_POP]]
            rewards_minus = [e["score"] for e in evals[NN.ES_POP:]]

            # standaryzacja nagród (rank-based można dodać później)
            r = np.array(rewards_plus) - np.array(rewards_minus)
            r = (r - r.mean()) / (r.std()+1e-8)

            grad = np.tensordot(r, eps, axes=1) / (NN.ES_POP)  # (dim,)
            self.theta = self.theta + (NN.ES_LR / self.sigma) * grad

            # ewaluacja aktualnej theta
            met = self._eval_theta(self.theta)
            sigma = self.sigma
            with open(path,"a",newline="",encoding="utf-8") as f:
                csv.writer(f).writerow([g+1, round(met["score"],4), round(met["wr_lb"],4),
                                        round(met["wr"],4), round(sigma,4), round(time.time()-t0,2)])
            print(f"[ES] G{g+1}: WR_LB={met['wr_lb']:.3f} WR={met['wr']:.3f} score={met['score']:.3f} time={time.time()-t0:.2f}s")

            # annealing sigmy
            if getattr(NN,"ANNEAL_SIGMA",True):
                self.sigma = max(0.03, self.sigma*0.995)

        np.save("nn_champion.npy", self.theta)
        print("Zapisano nn_champion.npy oraz logi w logs/es_log.csv")