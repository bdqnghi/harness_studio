"""Difficulty-stratified splits + the blind random fallback."""

from studio.components.profiler import Profile
from studio.components.splitter import random_split, stratified_split


def _disjoint(sp):
    sets = [set(sp.held_in), set(sp.regression), set(sp.held_out)]
    for i in range(3):
        for j in range(i + 1, 3):
            assert sets[i].isdisjoint(sets[j])


def test_stratified_routes_by_difficulty():
    pr = {**{f"s{i}": 1.0 for i in range(6)},   # solved
          **{f"f{i}": 0.0 for i in range(6)},   # failing
          **{f"m{i}": 0.5 for i in range(3)}}   # mixed
    sp = stratified_split(Profile(pass_rate=pr), held_in=6, reg=4, held_out_cap=4, seed=0)
    # held_in is learnable (failing/mixed), regression is reliably-solved
    assert all(t[0] in ("f", "m") for t in sp.held_in)
    assert all(t.startswith("s") for t in sp.regression)
    assert len(sp.held_out) == 4
    _disjoint(sp)


def test_stratified_tops_up_held_in_when_few_failures():
    # only 2 failing tasks but held_in asks for 5 -> top up from mixed/solved
    pr = {**{f"s{i}": 1.0 for i in range(10)}, **{f"f{i}": 0.0 for i in range(2)}}
    sp = stratified_split(Profile(pass_rate=pr), held_in=5, reg=2, held_out_cap=3, seed=0)
    assert len(sp.held_in) == 5  # filled despite few failures
    _disjoint(sp)


def test_stratified_deterministic_and_seed_sensitive():
    prof = Profile(pass_rate={f"t{i}": (i % 3) / 2 for i in range(18)})
    a = stratified_split(prof, held_in=5, reg=3, held_out_cap=6, seed=7)
    b = stratified_split(prof, held_in=5, reg=3, held_out_cap=6, seed=7)
    assert (a.held_in, a.regression, a.held_out) == (b.held_in, b.regression, b.held_out)
    c = stratified_split(prof, held_in=5, reg=3, held_out_cap=6, seed=8)
    assert a.held_out != c.held_out  # a different seed reshuffles the locked test


def test_random_split_sizes_and_disjoint():
    sp = random_split([f"t{i}" for i in range(30)], seed=0, held_in=10, reg=6, held_out_cap=8)
    assert len(sp.held_in) == 10 and len(sp.regression) == 6 and len(sp.held_out) == 8
    _disjoint(sp)
