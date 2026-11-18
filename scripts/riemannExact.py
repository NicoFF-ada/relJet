"""Exact solution of the 1D special-relativistic Riemann problem.

Ideal gas, velocities normal to the discontinuity only (no tangential flow),
left rarefaction + contact + right shock, the wave pattern of a blast wave
such as Problem 1 of Marti & Mueller (2003), Living Rev. Relativ. 6, 7.

    rarefaction : the Riemann invariant A(v) + B(c_s) is constant across the
                  fan, with A = artanh(v) and
                  B = ln((sqrt(g-1) + c_s) / (sqrt(g-1) - c_s)) / sqrt(g-1)
    shock       : Rankine-Hugoniot conditions F(U_b) - F(U_a) = V_s (U_b - U_a)
                  for D, S and tau, solved numerically for (rho_b, v_b, V_s)
    contact     : pressure and velocity continuous; p* found by bisection

Only numpy is needed. Used by plotRiemann.py; run directly to print the
star state of Problem 1.
"""
import math

import numpy as np


def _bisect(f, lo, hi, tol=1e-14, maxit=200):
    flo = f(lo)
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, abs(hi)):
            break
    return 0.5 * (lo + hi)


class Gas:
    def __init__(self, gamma):
        self.g = gamma

    def h(self, rho, p):
        return 1 + self.g / (self.g - 1) * p / rho

    def cs(self, rho, p):
        return math.sqrt(self.g * p / (rho * self.h(rho, p)))

    def B(self, cs):
        s = math.sqrt(self.g - 1)
        return math.log((s + cs) / (s - cs)) / s

    def cons(self, rho, p, v):
        W = 1 / math.sqrt(1 - v * v)
        h = self.h(rho, p)
        D = rho * W
        S = rho * h * W * W * v
        tau = rho * h * W * W - p - D
        return np.array([D, S, tau]), np.array([D * v, S * v + p, S - D * v])


def rarefaction_left(gas, rho_a, p_a, v_a, p):
    """State behind a left-facing rarefaction at pressure p < p_a."""
    K = p_a / rho_a ** gas.g
    rho = (p / K) ** (1 / gas.g)
    J = math.atanh(v_a) + gas.B(gas.cs(rho_a, p_a))
    v = math.tanh(J - gas.B(gas.cs(rho, p)))
    return rho, v


def shock_right(gas, rho_a, p_a, v_a, p_b):
    """State behind a right-facing shock with post-shock pressure p_b > p_a.

    Newton iteration on the Rankine-Hugoniot conditions for (rho_b, v_b, V_s),
    started from the strong-shock guess.
    """
    U_a, F_a = gas.cons(rho_a, p_a, v_a)
    x = np.array([rho_a * 4.0, 0.5 * (1 + v_a), 0.9])

    def residual(x):
        rho_b, v_b, Vs = x
        U_b, F_b = gas.cons(rho_b, p_b, v_b)
        return (F_b - F_a) - Vs * (U_b - U_a)

    # Newton with a finite-difference Jacobian, damped to stay physical.
    for _ in range(100):
        r = residual(x)
        J = np.empty((3, 3))
        for k in range(3):
            dx = np.zeros(3)
            dx[k] = 1e-7 * max(1.0, abs(x[k]))
            J[:, k] = (residual(x + dx) - r) / dx[k]
        step = np.linalg.solve(J, -r)
        lam = 1.0
        while True:
            y = x + lam * step
            if y[0] > 0 and abs(y[1]) < 1 and abs(y[2]) < 1:
                break
            lam *= 0.5
        x = y
        if np.max(np.abs(step)) < 1e-15:
            break
    return x


class RiemannSolution:
    """Exact solution for left rarefaction + right shock."""

    def __init__(self, left, right, gamma):
        self.gas = Gas(gamma)
        self.L, self.R = left, right  # (rho, p, v)
        rhoL, pL, vL = left
        rhoR, pR, vR = right

        def mismatch(p):
            _, v_l = rarefaction_left(self.gas, rhoL, pL, vL, p)
            _, v_r, _ = shock_right(self.gas, rhoR, pR, vR, p)
            return v_l - v_r

        self.p = _bisect(mismatch, pR * (1 + 1e-9), pL * (1 - 1e-12))
        self.rhoLs, self.v = rarefaction_left(self.gas, rhoL, pL, vL, self.p)
        self.rhoRs, _, self.Vs = shock_right(self.gas, rhoR, pR, vR, self.p)

        def lam_minus(rho, p, v):
            c = self.gas.cs(rho, p)
            return (v - c) / (1 - v * c)

        self.head = lam_minus(rhoL, pL, vL)
        self.tail = lam_minus(self.rhoLs, self.p, self.v)

    def _fan(self, xi):
        """State inside the rarefaction at x/t = xi."""
        rhoL, pL, vL = self.L
        K = pL / rhoL ** self.gas.g

        def f(p):
            rho = (p / K) ** (1 / self.gas.g)
            _, v = rarefaction_left(self.gas, rhoL, pL, vL, p)
            c = self.gas.cs(rho, p)
            return (v - c) / (1 - v * c) - xi

        p = _bisect(f, self.p, pL)
        rho = (p / K) ** (1 / self.gas.g)
        _, v = rarefaction_left(self.gas, rhoL, pL, vL, p)
        return rho, p, v

    def sample(self, x, t, x0=0.5):
        """Arrays rho, p, v at positions x and time t."""
        out = np.empty((3, len(x)))
        for n, xn in enumerate(x):
            xi = (xn - x0) / t
            if xi < self.head:
                s = self.L
            elif xi < self.tail:
                s = self._fan(xi)
            elif xi < self.v:
                s = (self.rhoLs, self.p, self.v)
            elif xi < self.Vs:
                s = (self.rhoRs, self.p, self.v)
            else:
                s = self.R
            out[:, n] = s
        return out


if __name__ == "__main__":
    sol = RiemannSolution((10.0, 40.0 / 3.0, 0.0), (1.0, 1e-6, 0.0), 5.0 / 3.0)
    print("Problem 1 (Marti & Mueller 2003):")
    print("  p*         = %.4f" % sol.p)
    print("  v*         = %.4f" % sol.v)
    print("  rho*_L     = %.4f" % sol.rhoLs)
    print("  rho*_R     = %.4f   (shell)" % sol.rhoRs)
    print("  shock speed= %.4f" % sol.Vs)
    print("  fan        = [%.4f, %.4f]" % (sol.head, sol.tail))
