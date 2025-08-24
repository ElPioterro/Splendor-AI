# tests/test_e2e_generation.py
import csv
import os
import numpy as np
import config
from agents import GeneticAgent
from evolution_manager import EvolutionManager
from pathlib import Path

def test_e2e_one_generation_and_csv_header(all_cards, all_nobles, tmp_path, monkeypatch):
    """
    End-to-end: 1 generacja, mała populacja, anchors=1, seeds=1.
    - Uruchamia pełne run_evolution
    - Weryfikuje nagłówki CSV i typy danych
    - Sprawdza istnienie champion_agent.npy i hof.json
    - Sprawdza games_per_candidate = anchors * seeds * 2
    """

    # 1) Pracuj w katalogu tymczasowym, żeby nie ruszać prawdziwych plików projektu
    monkeypatch.chdir(tmp_path)

    # 2) Minimalna i szybka konfiguracja
    monkeypatch.setattr(config, "POP_SIZE", 6, raising=False)
    monkeypatch.setattr(config, "GENERATIONS", 1, raising=False)
    monkeypatch.setattr(config, "ELITE_COUNT", 2, raising=False)
    monkeypatch.setattr(config, "IMMIGRANTS", 1, raising=False)
    monkeypatch.setattr(config, "TOURNAMENT_K", 3, raising=False)
    monkeypatch.setattr(config, "CROSSOVER_RATE", 0.8, raising=False)
    monkeypatch.setattr(config, "MUTATION_RATE", 0.2, raising=False)
    monkeypatch.setattr(config, "MUTATION_SIGMA_START", 0.2, raising=False)
    monkeypatch.setattr(config, "MUTATION_SIGMA_END", 0.1, raising=False)
    monkeypatch.setattr(config, "GENE_CLIP", 1.5, raising=False)

    monkeypatch.setattr(config, "ANCHORS_COUNT", 1, raising=False)
    monkeypatch.setattr(config, "HOF_MAX", 3, raising=False)
    monkeypatch.setattr(config, "EVAL_SEEDS_PER_ANCHOR", 1, raising=False)
    monkeypatch.setattr(config, "USE_MULTIPROCESSING", False, raising=False)
    monkeypatch.setattr(config, "USE_TOP_RESCORE", False, raising=False)

    # Szybsze gry (nie użyjemy ich realnie, ale trzymamy spójność)
    monkeypatch.setattr(config, "TARGET_PRESTIGE_POINTS", 5, raising=False)

    # Zapewnij 1 baseline anchor (spójność z ANCHORS_COUNT)
    monkeypatch.setattr(config, "BASELINE_DNA_POOL", [[0.0] * config.DNA_SIZE], raising=False)

    # 3) Podmień _play_one_game na szybki, deterministyczny stub
    #    - generuje mieszankę wygranych i remisów w zależności od seeda i roli
    def fake_play_one_game(self, dnaA, anchor, seed, swap_roles):
        win = 0 if ((seed & 1) ^ (1 if swap_roles else 0)) else -1  # 0 = Candidate, -1 = remis
        pa = 12 if win == 0 else 10
        pb = 10
        turns = 24 if win == 0 else 28
        return (win, pa, pb, turns)

    monkeypatch.setattr(EvolutionManager, "_play_one_game", fake_play_one_game, raising=True)

    # 4) Odpal E2E
    em = EvolutionManager(all_cards, all_nobles)
    em.run_evolution()

    # 5) Walidacja plików
    paths = sorted((tmp_path/"logs").glob("evolution_log*.csv"))
    assert paths, "Nie znaleziono pliku CSV"; log_path = paths[0]
    assert log_path.exists(), "Log CSV powinien zostać utworzony"
    assert (tmp_path / "champion_agent.npy").exists(), "Powinien zostać zapisany champion_agent.npy"
    assert (tmp_path / "hof.json").exists(), "Powinien zostać zapisany hof.json"

    # 6) Walidacja nagłówków i typów danych w CSV
    with open(log_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    assert len(rows) >= 2, "CSV powinien zawierać nagłówek i co najmniej jeden wiersz danych"

    header = rows[0]
    expected_header = [
        "generation","best_fitness","avg_fitness","worst_fitness","std_dev_fitness",
        "best_win_rate","avg_win_rate","avg_game_score","avg_game_turns",
        "elite_dna_sample","time_elapsed_sec","sigma","games_per_candidate"
    ]
    assert header == expected_header, f"Błędny nagłówek CSV.\nOtrzymano: {header}\nOczekiwano: {expected_header}"

    data = rows[1]
    assert len(data) == len(expected_header), "Wiersz danych powinien mieć tyle pól, co nagłówek"

    # Pola numeryczne: dają się zrzutować
    float_idx = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]  # fitnessy, win_rate, avg, time_elapsed, sigma
    int_idx = [0, 12]  # generation, games_per_candidate

    for i in float_idx:
        _ = float(data[i])  # nie rzuci wyjątku

    for i in int_idx:
        _ = int(float(data[i]))  # dopuszczamy zapis "1.0" -> int(1.0)

    # 7) Sprawdź poprawność games_per_candidate
    expected_games = min(config.ANCHORS_COUNT, len(config.BASELINE_DNA_POOL)) * config.EVAL_SEEDS_PER_ANCHOR * 2
    assert int(float(data[12])) == expected_games, "games_per_candidate powinno odzwierciedlać liczbę gier per kandydat"