#!/usr/bin/env python3
"""Convention microtests — settle algebraic-convention questions with
numpy, once, so they are never re-derived. Run: python3 microtests.py
(needs the proto venv)."""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from proto import sym_orbit_min

def test_permute_direction():
    """einsum(inputs=[id], output=[sigma(p)]) computes M[o] = R[o_sigma(.)].
    The canon relation (ren_t[p] = ren_rep[pi[p]]) requires sigma = pi^-1."""
    R = np.random.randn(2, 2, 2)
    Mtrue = np.transpose(R, (2, 0, 1))  # M[o0,o1,o2] = R[o2,o0,o1]
    pi = [1, 2, 0]
    pi_inv = [pi.index(q) for q in range(3)]
    MA = np.einsum(R, [0, 1, 2], [pi[p] for p in range(3)])
    MB = np.einsum(R, [0, 1, 2], [pi_inv[p] for p in range(3)])
    assert not np.allclose(MA, Mtrue)
    assert np.allclose(MB, Mtrue)
    print("ok: permute direction (output = [pi_inv(p)])")

def test_orbit_sign_double_antisym():
    """factor(instance) = orbit-sign x factor(orbit-min), verified against a
    double-antisymmetric rank-4 tensor."""
    np.random.seed(0)
    V = np.random.randn(3, 3, 3, 3)
    V = V - V.transpose(1, 0, 2, 3)
    V = V - V.transpose(0, 1, 3, 2)
    syms = {"0": [{"permutation": [1, 0, 2, 3], "sign": -1},
                  {"permutation": [0, 1, 3, 2], "sign": -1}]}
    pt = (1, 2, 0)
    for ls in [(2, 0, 1, 2), (1, 2, 0, 2), (2, 1, 2, 0), (0, 2, 1, 2)]:
        best, s = sym_orbit_min(0, ls, syms)
        lhs = V[tuple(pt[l] for l in ls)]
        rhs = V[tuple(pt[l] for l in best)]
        assert np.isclose(lhs, s * rhs), (ls, best, s)
    print("ok: orbit sign convention (double-antisymmetric)")

if __name__ == "__main__":
    np.random.seed(0)
    test_permute_direction()
    test_orbit_sign_double_antisym()
    print("all microtests pass")
