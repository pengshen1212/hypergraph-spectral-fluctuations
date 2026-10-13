"""Exact finite-sample tolerance ranks for upper-tail calibration."""

from __future__ import annotations

from fractions import Fraction
from math import comb


def _fraction(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    raise TypeError("probabilities must be exact Fraction or int values")


def binomial_upper_tail_exact(n, rank, success_probability):
    """Return P(Binomial(n,p) >= rank) as an exact rational number."""
    n, rank = int(n), int(rank)
    p = _fraction(success_probability)
    if n < 1 or not 1 <= rank <= n:
        raise ValueError("require n>=1 and 1<=rank<=n")
    if not 0 <= p <= 1:
        raise ValueError("success probability must lie in [0,1]")
    a, b, denominator = p.numerator, p.denominator - p.numerator, p.denominator
    numerator = sum(
        comb(n, j) * (a ** j) * (b ** (n - j))
        for j in range(rank, n + 1)
    )
    return Fraction(numerator, denominator ** n)


def tolerance_order_statistic_rank(n, alpha, delta):
    """Smallest k with P(Bin(n,1-alpha)>=k)<=delta, using exact arithmetic."""
    n = int(n)
    alpha, delta = _fraction(alpha), _fraction(delta)
    if n < 1:
        raise ValueError("n must be positive")
    if not 0 < alpha < 1 or not 0 < delta < 1:
        raise ValueError("alpha and delta must lie strictly between zero and one")
    p = 1 - alpha
    a, b, denominator = p.numerator, p.denominator - p.numerator, p.denominator
    total_denominator = denominator ** n
    tail_numerator = 0
    best = None
    for rank in range(n, 0, -1):
        tail_numerator += comb(n, rank) * (a ** rank) * (b ** (n - rank))
        if tail_numerator * delta.denominator <= delta.numerator * total_denominator:
            best = rank
        else:
            break
    if best is None:
        raise ValueError("sample size cannot attain the requested tolerance")
    return best
