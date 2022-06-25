import pytest

import alogos as al


# Crossover

def test_crossover_two_points():
    for length, expected_num_variants in [(2, 2), (3, 5), (4, 9)]:
        observed_variants = set()
        for _ in range(1000):
            p1, p2 = al.systems._shared.crossover._get_two_different_points(length)
            observed_variants.add((p1, p2))
        assert len(observed_variants) == expected_num_variants


# Neighborhood

def test_generate_combinations_1():
    no_choices = [
        [],
        [0],
        [0, 0, 0],
    ]
    for choices_per_position in no_choices:
        for distance in (1, 2, 3, 4):
            # Full
            comb = al.systems._shared.neighborhood.generate_combinations(
                choices_per_position, distance)
            assert comb == []
            # Subset
            for max_size in (1, 2, 3, 4):
                comb = al.systems._shared.neighborhood.generate_combinations(
                    choices_per_position, distance, max_size=2)
                assert comb == []


def test_generate_combinations_2():
    choices_per_position = [2, 2]
    d1_comb = [(0, 1), (0, 2), (1, 0), (2, 0)]
    d2_comb = [(1, 1), (1, 2), (2, 1), (2, 2)]
    d3_comb = []

    # Full
    distance = 1
    comb = al.systems._shared.neighborhood.generate_combinations(choices_per_position, distance)
    assert sorted(comb) == d1_comb

    distance = 2
    comb = al.systems._shared.neighborhood.generate_combinations(choices_per_position, distance)
    assert sorted(comb) == d2_comb

    distance = 3
    comb = al.systems._shared.neighborhood.generate_combinations(choices_per_position, distance)
    assert sorted(comb) == d3_comb

    # Subset
    n = 1000
    for max_size in (1, 2, 3, 4):
        distance = 1
        full = set()
        for _ in range(n):
            new = al.systems._shared.neighborhood.generate_combinations(
                choices_per_position, distance, max_size=max_size)
            assert len(new) == max_size
            full = full.union(new)
        assert sorted(full) == d1_comb

        distance = 2
        full = set()
        for _ in range(n):
            new = al.systems._shared.neighborhood.generate_combinations(
                choices_per_position, distance, max_size=max_size)
            assert len(new) == max_size
            full = full.union(new)
        assert sorted(full) == d2_comb

        distance = 3
        full = set()
        for _ in range(n):
            new = al.systems._shared.neighborhood.generate_combinations(
                choices_per_position, distance, max_size=max_size)
            assert len(new) == 0
            full = full.union(new)
        assert sorted(full) == d3_comb


def test_generate_combinations_3():
    choices_per_position = [0, 3, 1, 0, 2]
    d1_comb = set([
        (0, 1, 0, 0, 0),
        (0, 2, 0, 0, 0),
        (0, 3, 0, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 0, 0, 1),
        (0, 0, 0, 0, 2)])
    d2_comb = set([
        (0, 1, 1, 0, 0),
        (0, 2, 1, 0, 0),
        (0, 3, 1, 0, 0),
        (0, 1, 0, 0, 1),
        (0, 1, 0, 0, 2),
        (0, 2, 0, 0, 1),
        (0, 2, 0, 0, 2),
        (0, 3, 0, 0, 1),
        (0, 3, 0, 0, 2),
        (0, 0, 1, 0, 1),
        (0, 0, 1, 0, 2)])
    d3_comb = set([
        (0, 1, 1, 0, 1),
        (0, 1, 1, 0, 2),
        (0, 2, 1, 0, 1),
        (0, 2, 1, 0, 2),
        (0, 3, 1, 0, 1),
        (0, 3, 1, 0, 2)])

    # Full
    distance = 1
    comb = al.systems._shared.neighborhood.generate_combinations(choices_per_position, distance)
    assert set(comb) == d1_comb

    distance = 2
    comb = al.systems._shared.neighborhood.generate_combinations(choices_per_position, distance)
    assert set(comb) == d2_comb

    distance = 3
    comb = al.systems._shared.neighborhood.generate_combinations(choices_per_position, distance)
    assert set(comb) == d3_comb

    # Subset
    n = 1000
    for max_size in (1, 2, 3, 4):
        distance = 1
        full = set()
        for _ in range(n):
            new = al.systems._shared.neighborhood.generate_combinations(
                choices_per_position, distance, max_size=max_size)
            assert len(new) == max_size
            full = full.union(new)
        assert full == d1_comb

        distance = 2
        full = set()
        for _ in range(n):
            new = al.systems._shared.neighborhood.generate_combinations(
                choices_per_position, distance, max_size=max_size)
            assert len(new) == max_size
            full = full.union(new)
        assert full == d2_comb

        distance = 3
        full = set()
        for _ in range(n):
            new = al.systems._shared.neighborhood.generate_combinations(
                choices_per_position, distance, max_size=max_size)
            assert len(new) == max_size
            full = full.union(new)
        assert full == d3_comb
