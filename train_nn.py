# train_nn.py
# Driver odnogi NN: bootstrap środowiska i uruchomienie OpenES.
# Zgodnie z Konstytucją: brak logiki treningowej – wszystko w es_manager.py.

from __future__ import annotations

import argparse
import sys
from typing import List, Optional
import numpy as np

import config  # SSOT dla DATA_FILE itp.
from game_loader import GameLoader
from es_manager import ESManager
import nn_config as cfg
from nn_agent import NnAgent


def run_training_nn(
    generations: int,
    anchors_files: Optional[List[str]] = None,
    resume_path: Optional[str] = None,
):
    print("=" * 60)
    print(" PROMETHEUS NN: OpenES Training ".center(60, " "))
    print("=" * 60)
    print(f"[Info] Generations: {generations}")
    print(f"[Info] SEED_BASE: {cfg.SEED_BASE}")
    print(f"[Info] MP: {'ON' if cfg.USE_MULTIPROCESSING else 'OFF'} (N_PROCESSES={cfg.N_PROCESSES})")
    print(f"[Info] NN: H1={cfg.HIDDEN1}, H2={cfg.HIDDEN2}, ACT={cfg.ACT}")
    print(f"[Info] Logs: {cfg.ES_LOG_CSV}")
    print(f"[Info] Model out: {cfg.MODEL_OUT_FILE}")
    if anchors_files:
        print(f"[Info] Anchors (custom): {anchors_files}")
    print("-" * 60)

    # 1) Wczytaj definicje gry (SSOT)
    try:
        loader = GameLoader()
        all_cards, all_nobles = loader.load_definitions_from_json(config.DATA_FILE)
    except Exception as e:
        print(f"[BŁĄD] Nie udało się wczytać definicji gry z '{config.DATA_FILE}': {e}")
        sys.exit(1)

    # 2) ES Manager
    try:
        manager = ESManager(
            all_cards=all_cards,
            all_nobles=all_nobles,
        )
    except Exception as e:
        print(f"[BŁĄD] Nie udało się zainicjalizować ESManager: {e}")
        print("Upewnij się, że istnieje champion_agent.npy lub podaj kotwice w configu.")
        sys.exit(1)

    # 3) Opcjonalne wznowienie od poprzedniego champion’a NN
    theta_init = None
    if resume_path:
        try:
            theta_init = np.load(resume_path).astype(np.float32)
            print(f"[Resume] Załadowano theta z: {resume_path} (len={theta_init.size})")
        except Exception as e:
            print(f"[Resume] Nie udało się wczytać '{resume_path}': {e} (pomiń wznowienie)")
            theta_init = None
    else:
        # Spróbuj domyślnego pliku, jeśli istnieje
        try:
            theta_init = np.load(cfg.MODEL_OUT_FILE).astype(np.float32)
            print(f"[Resume] Załadowano istniejący model: {cfg.MODEL_OUT_FILE} (len={theta_init.size})")
        except Exception:
            theta_init = None

    # Weryfikacja rozmiaru (jeśli theta_init jest dostępna)
    if theta_init is not None:
        expected = NnAgent.param_count(cfg.HIDDEN1, cfg.HIDDEN2)
        if theta_init.size != expected:
            print(f"[Resume] Ostrzeżenie: rozmiar theta ({theta_init.size}) != oczekiwany ({expected}). Ignoruję resume.")
            theta_init = None

    # 4) Odpal trening
    best_theta = manager.run(generations=generations, theta_init=theta_init)

    # 5) Podsumowanie
    print("=" * 60)
    print(" Trening zakończony ")
    print(f" Najlepszy model zapisany w: {cfg.MODEL_OUT_FILE}")
    print(f" Log: {cfg.ES_LOG_CSV}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PROMETHEUS NN: OpenES trainer (numpy, mirrored sampling)")
    p.add_argument("--gens", type=int, default=200, help="Liczba generacji ES (domyślnie 200)")
    p.add_argument(
        "--anchors", nargs="*", default=None,
        help="Ścieżki do plików *.npy z DNA kotwic GA (opcjonalnie). Jeśli brak, użyje champion_agent.npy / anchors/ / hof/"
    )
    p.add_argument(
        "--resume", type=str, default=None,
        help="Ścieżka do wcześniejszego nn_champion.npy (wznowienie). Jeśli brak, spróbuje cfg.MODEL_OUT_FILE."
    )
    p.add_argument(
        "--no-mp", action="store_true",
        help="Wyłącz multiprocessing (debug)."
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Opcjonalnie wyłącz MP przez CLI
    if args.no_mp:
        cfg.USE_MULTIPROCESSING = False

    run_training_nn(
        generations=int(args.gens),
        anchors_files=args.anchors if args.anchors else None,
        resume_path=args.resume,
    )