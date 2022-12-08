"""Shared neighborhood functions for several systems."""

import functools as _functools
import itertools as _itertools
import operator as _operator
import random as _random


def generate_combinations(num_choices_per_pos, distance, max_size=None):
    """Generate all combinations of choices available at each position.

    In each returned combination, the value 0 means that the original choice
    shall be kept, while values > 0 mean that the alternative choice at
    position `value-1` in the list of all alternatives shall be used.

    Example
    -------
    - Input

        - num_choices_per_pos = [4, 1, 0, 7, 3]
        - distance = 1

    - Output

        - combinations = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 0, 0, 0, 3], [0, 0, 0, 1, 0], ...]

    """
    # Special cases
    # - No positions
    if not num_choices_per_pos:
        return []
    # - Distance outside the number of positions
    num_pos = len(num_choices_per_pos)
    if distance < 1 or distance > num_pos:
        return []

    # Count the number of all neighbors in the chosen distance
    pos_combinations = [
        comb
        for comb in _itertools.combinations(range(num_pos), distance)
        if all(num_choices_per_pos[pos] > 0 for pos in comb)
    ]
    count_per_pos_comb = list(
        _product(num_choices_per_pos[pos] for pos in positions)
        for positions in pos_combinations
    )
    count = sum(count_per_pos_comb)

    # Check if the number of possible neighbors is over the max desired number of neighbors
    choices_per_pos = [
        [0] if n == 0 else list(range(1, n + 1)) for n in num_choices_per_pos
    ]
    if max_size is not None and count > max_size:
        # If yes, generate a random subset
        choice_combinations = _generate_some_combinations(
            choices_per_pos,
            pos_combinations,
            num_pos,
            max_size,
            count,
            count_per_pos_comb,
        )
    else:
        # If no, generate the entire set
        choice_combinations = _generate_all_combinations(
            choices_per_pos, pos_combinations, num_pos
        )
    return choice_combinations


def _product(iterable):
    """Multiply all items in an iterable in order to form their product."""
    # Available as built-in math.prod since Python 3.8
    return _functools.reduce(_operator.mul, iterable, 1)


def _generate_some_combinations(
    choices_per_pos, pos_combinations, num_pos, max_size, count, count_per_pos_comb
):
    """Generate a random subset of all possible combinations."""
    # Choose random neighbors by selecting indices from the enumerated possibilities
    chosen = sorted(_random.sample(range(count), max_size))

    # Construct the chosen neighbors
    if chosen:
        combinations = []
        i = 0
        finished = False
        current = chosen.pop(0)
        template = [[0] for _ in range(num_pos)]
        for cnt, positions in zip(count_per_pos_comb, pos_combinations):
            if finished:
                break
            if i + cnt < current:
                i += cnt
                continue
            selected_choices = template[:]  # fast shallow copy
            for pos in positions:
                selected_choices[pos] = choices_per_pos[pos]
            for comb in _itertools.product(*selected_choices):
                if i == current:
                    combinations.append(comb)
                    if chosen:
                        current = chosen.pop(0)
                    else:
                        finished = True
                i += 1
    return combinations


def _generate_all_combinations(choices_per_pos, pos_combinations, num_pos):
    """Generate the full set of all possible combinations."""
    combinations = []
    template = [[0] for _ in range(num_pos)]
    for positions in pos_combinations:
        selected_choices = template[:]  # fast shallow copy
        for pos in positions:
            selected_choices[pos] = choices_per_pos[pos]
        new_combinations = list(_itertools.product(*selected_choices))
        combinations.extend(new_combinations)
    return combinations
