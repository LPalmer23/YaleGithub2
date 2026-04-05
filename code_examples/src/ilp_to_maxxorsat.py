"""Transformation from 0-1 ILP to max-XORSAT instance.

Implements the pipeline from arXiv:2509.08328 Section 5:
  ILP constraints -> pseudo-Boolean encoding -> max-XORSAT -> parity check matrix B

For the DQI algorithm, we need:
  - B: parity check matrix (binary, m x n)
  - v: constraint vector (binary, m)
  - weights: clause weights (float, m)
  - var_map: which columns of B correspond to original ILP decision variables

This module provides a simplified encoding suitable for small toy problems,
focusing on equality constraints (after slack variable introduction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class MaxXORSATInstance:
    """A weighted max-XORSAT instance: maximize sum of satisfied B_i x = v_i (mod 2).

    Attributes:
        B: Parity check matrix (m x n), binary.
        v: Constraint vector (m,), binary.
        weights: Clause weights (m,), float. Higher weight = more important.
        num_original_vars: Number of original ILP decision variables.
        var_map: Indices in B columns corresponding to original decision variables.
    """
    B: np.ndarray
    v: np.ndarray
    weights: np.ndarray
    num_original_vars: int
    var_map: list[int]

    @property
    def num_equations(self) -> int:
        return self.B.shape[0]

    @property
    def num_variables(self) -> int:
        return self.B.shape[1]

    def check_solution(self, x: np.ndarray) -> tuple[int, int]:
        """Check how many equations a solution satisfies.

        Args:
            x: Binary vector of length num_variables.

        Returns:
            Tuple of (satisfied, total) equation counts.
        """
        syndrome = (self.B @ x) % 2
        satisfied = int(np.sum(syndrome == self.v))
        return satisfied, self.num_equations

    def extract_original_vars(self, x: np.ndarray) -> np.ndarray:
        """Extract original ILP decision variables from full variable vector."""
        return x[self.var_map]


def and_gadget() -> tuple[np.ndarray, np.ndarray, int]:
    """AND gadget: z = x AND y encoded as 4 XOR equations over 3 variables.

    Variables: [x, y, z]
    Equations (Eq. 36 in paper):
        x + y + z = 1  (mod 2)
        x + z = 0      (mod 2)
        y + z = 0      (mod 2)
        z = 0           (mod 2)

    Maximally satisfied (3 out of 4) iff z = x * y.

    Returns:
        Tuple of (B_local, v_local, num_vars=3) where B_local is (4,3), v_local is (4,).
    """
    B = np.array([
        [1, 1, 1],  # x + y + z = 1
        [1, 0, 1],  # x + z = 0
        [0, 1, 1],  # y + z = 0
        [0, 0, 1],  # z = 0
    ], dtype=int)
    v = np.array([1, 0, 0, 0], dtype=int)
    return B, v, 3


def carry1_gadget() -> tuple[np.ndarray, np.ndarray, int]:
    """CARRY1 gadget: first step of binary addition (k=0).

    Variables: [s_0, u_0, v_0, c_0]
    Encodes: s_0 = u_0 + v_0 mod 2, c_0 = u_0 * v_0

    Equations (Eq. 39 in paper):
        c_0 + u_0 + v_0 = 1  (mod 2)
        c_0 + v_0 = 0        (mod 2)
        u_0 + v_0 = 0        (mod 2)
        v_0 = 0              (mod 2)
        s_0 + v_0 + u_0 = 0  (mod 2)

    Returns:
        Tuple of (B_local, v_local, num_vars=4).
        Variable order: [s_0, u_0, v_0, c_0]
    """
    # Variables indexed as: 0=s_0, 1=u_0, 2=v_0, 3=c_0
    B = np.array([
        [0, 1, 1, 1],  # c_0 + u_0 + v_0 = 1
        [0, 0, 1, 1],  # c_0 + v_0 = 0
        [0, 1, 1, 0],  # u_0 + v_0 = 0
        [0, 0, 1, 0],  # v_0 = 0
        [1, 1, 1, 0],  # s_0 + v_0 + u_0 = 0
    ], dtype=int)
    v = np.array([1, 0, 0, 0, 0], dtype=int)
    return B, v, 4


def ilp_to_maxxorsat(
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    senses: list[int],
    objective_weight: float = 1.0,
) -> MaxXORSATInstance:
    """Convert a 0-1 ILP to a max-XORSAT instance.

    This is a simplified encoding for small toy problems. Instead of the full
    gadget-based encoding from the paper (which produces very large instances),
    we use a direct approach:

    1. For each constraint, add slack variables to convert inequalities to equalities
    2. Encode each equality constraint bit-by-bit as XOR equations
    3. Encode the objective function as soft XOR clauses

    For the toy insurance problem (5 vars, 4 constraints), this produces a
    manageable max-XORSAT instance suitable for DQI simulation.

    Args:
        c: Cost vector (n,).
        A: Constraint matrix (m, n).
        b: RHS vector (m,).
        senses: Constraint senses (-1=<=, 0===, 1=>=). PuLP convention.
        objective_weight: Weight for objective function clauses.

    Returns:
        MaxXORSATInstance with parity check matrix B, constraint vector v, etc.
    """
    n_orig = len(c)
    m_constraints = len(b)

    # Collect all XOR equations as rows of B and entries of v
    B_rows = []
    v_entries = []
    weight_entries = []
    next_var = n_orig  # Index for auxiliary/slack variables

    # Track variable mapping: first n_orig columns are original decision variables
    var_map = list(range(n_orig))

    # ---- Step 1: Encode constraints as XOR equations ----
    for row_idx in range(m_constraints):
        a_row = A[row_idx]
        b_val = b[row_idx]
        sense = senses[row_idx]

        # Convert to equality: a^T x + slack = b (for <=) or a^T x - slack = b (for >=)
        if sense == -1:  # <=
            # a^T x <= b  =>  a^T x + s = b where s >= 0
            # For binary: we need enough slack bits to cover the range
            max_lhs = sum(max(0, a_row[j]) for j in range(n_orig))
            slack_range = max(0, int(max_lhs - b_val))
            a_eq = a_row.copy()
            b_eq = b_val
        elif sense == 1:  # >=
            # a^T x >= b  =>  a^T x - s = b  =>  -a^T x + s = -b => a^T x = b + s
            max_lhs = sum(max(0, a_row[j]) for j in range(n_orig))
            slack_range = max(0, int(max_lhs - b_val))
            a_eq = a_row.copy()
            b_eq = b_val
        else:  # == (sense == 0)
            a_eq = a_row.copy()
            b_eq = b_val
            slack_range = 0

        # Encode the integer equality a_eq^T x (+ slack) = b_eq as XOR equations
        # For binary coefficients: a^T x = b mod 2 is a single XOR equation
        # For integer equality, we need bit-level encoding

        # Simplification: for binary-coefficient constraints (all coeffs are -1, 0, or 1),
        # encode directly as XOR on the LSB
        int_coeffs = np.round(a_eq).astype(int)
        int_b = int(round(b_eq))

        if slack_range == 0:
            # Equality constraint: encode each bit position
            # For the simplest case (small coefficients), just do mod-2
            row = np.zeros(next_var + slack_range, dtype=int)
            for j in range(n_orig):
                if int_coeffs[j] % 2 == 1:
                    row[j] = 1
            B_rows.append(row)
            v_entries.append(int_b % 2)
            weight_entries.append(10.0)  # Hard constraint
        else:
            # Inequality: add slack bits and encode
            num_slack_bits = max(1, int(np.ceil(np.log2(slack_range + 1))))
            # XOR equation for the LSB
            row = np.zeros(next_var + num_slack_bits, dtype=int)
            for j in range(n_orig):
                if int_coeffs[j] % 2 == 1:
                    row[j] = 1
            # Add first slack bit
            row[next_var] = 1
            B_rows.append(row)
            v_entries.append(int_b % 2)
            weight_entries.append(10.0)

            next_var += num_slack_bits

    # ---- Step 2: Encode objective as soft XOR clauses ----
    # For each non-zero c_i, add a soft clause: x_i = 1 (mod 2)
    # with weight proportional to c_i
    # This encourages the solver to set x_i = 1 for positive-cost variables
    for j in range(n_orig):
        if c[j] > 0:
            row = np.zeros(next_var, dtype=int)
            row[j] = 1
            B_rows.append(row)
            v_entries.append(1)  # want x_j = 1
            weight_entries.append(objective_weight * c[j])

    # Pad all rows to same width
    max_cols = next_var
    for i in range(len(B_rows)):
        if len(B_rows[i]) < max_cols:
            B_rows[i] = np.concatenate([B_rows[i], np.zeros(max_cols - len(B_rows[i]), dtype=int)])

    B = np.array(B_rows, dtype=int)
    v = np.array(v_entries, dtype=int)
    weights = np.array(weight_entries, dtype=float)

    return MaxXORSATInstance(
        B=B,
        v=v,
        weights=weights,
        num_original_vars=n_orig,
        var_map=var_map,
    )
