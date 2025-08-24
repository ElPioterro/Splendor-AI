# tests/test_hof.py
import os
import json
import numpy as np
from config import DNA_SIZE
from evolution_manager import EvolutionManager

def test_hof_add_dedupe_and_topk(all_cards, all_nobles, tmp_path, monkeypatch):
    em = EvolutionManager(all_cards, all_nobles)
    em.hof.clear()  # zaczynamy od pustego

    # Dwie bardzo podobne próbki (L2 < 0.05), druga ma wyższy fitness
    d1 = np.zeros(DNA_SIZE)
    d2 = np.ones(DNA_SIZE) * 0.01  # norm ~ 0.04
    em._hof_add(d1, 1.0)
    em._hof_add(d2, 2.0)

    assert len(em.hof) == 1, "Podobne DNA powinny zostać zdeduplikowane"
    assert np.allclose(em.hof[0][0], d2), "Lepszy fitness powinien zastąpić starszy wpis"
    assert em.hof[0][1] == 2.0

    # Dodaj więcej niż HOF_MAX unikalnych wpisów i sprawdź przycięcie
    for i in range(20):
        em._hof_add(np.ones(16) * (i + 1) * 0.2, float(i))  # daleko od siebie

    assert len(em.hof) <= getattr(__import__("config"), "HOF_MAX"), "HoF powinien być przycięty do HOF_MAX"
    # Lista powinna być posortowana malejąco po fitness
    fits = [f for (_, f) in em.hof]
    assert fits == sorted(fits, reverse=True)

    # Trwałość: zapisz i wczytaj z pliku
    hof_path = tmp_path / "hof.json"
    em._save_hof(str(hof_path))

    # Nowy EM i wczytanie z pliku
    em2 = EvolutionManager(all_cards, all_nobles)
    em2.hof.clear()
    em2._load_hof(str(hof_path))

    assert len(em2.hof) == len(em.hof)
    for (d_a, f_a), (d_b, f_b) in zip(em.hof, em2.hof):
        assert np.allclose(d_a, d_b)
        assert f_a == f_b