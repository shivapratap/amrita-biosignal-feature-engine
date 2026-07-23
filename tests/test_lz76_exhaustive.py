from __future__ import annotations

from itertools import product

import numpy as np

from amrita_biosignal_feature_engine.complexity import _lz76_phrase_count


def _independent_exhaustive_history_oracle(sequence: tuple[int, ...]) -> int:
    """Parse by finding each shortest phrase absent from prior start positions."""
    phrase_count = 1
    cursor = 1
    while cursor < len(sequence):
        remaining = len(sequence) - cursor
        phrase_length = remaining
        for candidate_length in range(1, remaining + 1):
            candidate = sequence[cursor : cursor + candidate_length]
            reproducible = any(
                sequence[start : start + candidate_length] == candidate
                for start in range(cursor)
            )
            if not reproducible:
                phrase_length = candidate_length
                break
        phrase_count += 1
        cursor += phrase_length
    return phrase_count


def test_all_binary_sequences_through_length_twelve_match_independent_oracle() -> None:
    for length in range(2, 13):
        for sequence in product((0, 1), repeat=length):
            encoded = np.asarray(sequence, dtype=np.uint8)
            assert _lz76_phrase_count(encoded) == _independent_exhaustive_history_oracle(
                sequence
            )
