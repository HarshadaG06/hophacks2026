"""
Kleinberg burst detection — vendored MIT port of pybursts.

Attribution: adapted from https://github.com/romain-fontugne/pybursts
(pybursts 0.1.1). The PyPI package's __init__ breaks `from pybursts import kleinberg`
on modern Python, so we vendor the algorithm here.

Original package computes n = len(gaps) and T = sum(gaps) internally from each
offset series; there are no optional n/T constructor args in 0.1.1. We keep
s and gamma identical across audios for comparability.
"""
from __future__ import annotations

import math

import numpy as np


def kleinberg(offsets, s: float = 2.0, gamma: float = 1.0):
    if s <= 1:
        raise ValueError("s must be greater than 1!")
    if gamma <= 0:
        raise ValueError("gamma must be positive!")
    if len(offsets) < 1:
        raise ValueError("offsets must be non-empty!")

    offsets = np.asarray(offsets, dtype=float)
    if offsets.size == 1:
        return np.array([[0, offsets[0], offsets[0]]], dtype=float)

    offsets = np.sort(offsets)
    gaps = np.diff(offsets)
    if np.any(gaps <= 0):
        raise ValueError("Input cannot contain events with zero time between!")

    T = float(np.sum(gaps))
    n = int(gaps.size)
    g_hat = T / n

    k = int(math.ceil(1 + math.log(T, s) + math.log(1.0 / float(np.min(gaps)), s)))
    gamma_log_n = gamma * math.log(n)

    def tau(i, j):
        if i >= j:
            return 0.0
        return (j - i) * gamma_log_n

    alpha = np.array([s ** x / g_hat for x in range(k)], dtype=float)

    def f(j, x):
        return alpha[j] * math.exp(-alpha[j] * x)

    C = np.full(k, np.inf)
    C[0] = 0.0
    q = np.empty((k, 0))

    for t in range(n):
        C_prime = np.full(k, np.inf)
        q_prime = np.full((k, t + 1), np.nan)
        for j in range(k):
            cost = C + np.array([tau(x, j) for x in range(k)])
            el = int(np.argmin(cost))
            fj = f(j, gaps[t])
            if fj > 0:
                C_prime[j] = cost[el] - math.log(fj)
            if t > 0:
                q_prime[j, :t] = q[el, :]
            q_prime[j, t] = j + 1
        C = C_prime
        q = q_prime

    j = int(np.argmin(C))
    q = q[j, :]

    prev_q = 0
    N = 0
    for t in range(n):
        if q[t] > prev_q:
            N += int(q[t] - prev_q)
        prev_q = q[t]

    bursts = np.zeros((N, 3), dtype=float)
    bursts[:, 1] = offsets[0]
    bursts[:, 2] = offsets[0]

    burst_counter = -1
    prev_q = 0
    stack = np.full(N, np.nan)
    stack_counter = -1
    for t in range(n):
        if q[t] > prev_q:
            for i in range(int(q[t] - prev_q)):
                burst_counter += 1
                bursts[burst_counter, 0] = prev_q + i
                bursts[burst_counter, 1] = offsets[t]
                stack_counter += 1
                stack[stack_counter] = burst_counter
        elif q[t] < prev_q:
            for _ in range(int(prev_q - q[t])):
                bursts[int(stack[stack_counter]), 2] = offsets[t]
                stack_counter -= 1
        prev_q = q[t]

    while stack_counter >= 0:
        bursts[int(stack[stack_counter]), 2] = offsets[n]
        stack_counter -= 1

    return bursts
