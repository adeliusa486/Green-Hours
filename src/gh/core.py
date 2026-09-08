"""
Core model for the carbon-aware deferral game.

Parameterisation follows REVIEW.md B1: the model is stated in terms of three
*measured* primitives per slot rather than a fitted no-intercept quadratic.

    a_t   published average emission factor at the forecast operating point,
          i.e. AEF_t(D_t).                                    [gCO2/kWh]
    m_t   true marginal emission factor at that point,
          i.e. MEF_t(D_t).                                    [gCO2/kWh]
    b_t   response curvature: d(MEF)/d(load), the rate at which the marginal
          factor rises with additional flexible load.         [gCO2/kWh/MWh]

The old parameterisation E_t(L) = alpha*L + beta*L^2 is the special case
m_t - a_t = beta_t * D_t, and it forces alpha_t >= 0, hence m_t/a_t <= 2 and
eta_t <= 0.5.  The (a, m, b) form has no such cap.  See tests/test_b1_cap.py.

Costs
-----
    y_t = sum_i x_it                       aggregate flexible load in slot t
    J_i(x) = sum_t x_it (a_t + b_t y_t) + lam_i sum_t psi_it x_it     (agent)
    C(x)   = sum_t [ m_t y_t + b_t y_t^2 ] + sum_i lam_i sum_t psi_it x_it
    Phi(x) = sum_t [ a_t y_t + (b_t/2)(y_t^2 + sum_i x_it^2) ]
             + sum_i lam_i sum_t psi_it x_it                      (potential)

Every equilibrium concept in this file reduces to repeated solution of one
primitive: a separable convex quadratic over the feasible polytope X_i.  The
three concepts differ only in the linear coefficient handed to that primitive.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

TOL = 1e-10


# --------------------------------------------------------------------------
# feasible set and the separable-QP primitive
# --------------------------------------------------------------------------

@dataclass
class Feasible:
    """X_i = { x >= 0 : sum_t x_t = E, x_t <= u_t, sum_{tau<=t} x_tau >= R_t }.

    R is the cumulative deadline staircase (non-decreasing, R_T <= E).  When R
    is None the set is the parallel-links / common-deadline case, which is what
    the analytic oracles in tests/ use.
    """
    E: float
    u: np.ndarray                      # per-slot power envelope
    R: np.ndarray | None = None        # cumulative staircase, or None

    def __post_init__(self):
        self.u = np.asarray(self.u, dtype=float)
        if self.R is not None:
            self.R = np.asarray(self.R, dtype=float)
            if np.any(np.diff(self.R) < -1e-12):
                raise ValueError("staircase R must be non-decreasing")
            if self.R[-1] > self.E + 1e-9:
                raise ValueError("staircase demands more than E")

    @property
    def T(self) -> int:
        return len(self.u)

    def check(self, x, tol=1e-6) -> bool:
        if x.min() < -tol or np.any(x > self.u + tol):
            return False
        if abs(x.sum() - self.E) > tol * max(1.0, self.E):
            return False
        if self.R is not None and np.any(np.cumsum(x) < self.R - tol):
            return False
        return True


def _waterfill(q, c, E, u):
    """min sum_t (q_t x_t + c_t x_t^2)  s.t.  sum x = E, 0 <= x_t <= u_t.

    c_t > 0 strictly.  Exact and non-iterative.

    Stationarity gives x_t(nu) = clip((nu - q_t)/(2 c_t), 0, u_t), so
    S(nu) = sum_t x_t(nu) is a non-decreasing piecewise-linear function of the
    equality multiplier nu, with breakpoints where a slot leaves zero
    (nu = q_t) or reaches its envelope (nu = q_t + 2 c_t u_t).  We sort the 2T
    breakpoints, accumulate the slope and intercept of S across them, locate
    the segment containing E, and solve one linear equation.  O(T log T), and
    exact to machine precision rather than to a bisection tolerance.
    """
    q = np.asarray(q, float)
    c = np.asarray(c, float)
    u = np.asarray(u, float)

    tot_u = u.sum()
    if E < -1e-9 or E > tot_u + 1e-9:
        raise ValueError(f"infeasible: E={E}, capacity={tot_u}")
    E = min(max(E, 0.0), tot_u)
    if E >= tot_u - 1e-15:
        return u.copy()
    if E <= 1e-15:
        return np.zeros_like(u)

    inv = 1.0 / (2.0 * c)
    enter = q                      # x_t leaves 0
    leave = q + 2.0 * c * u        # x_t reaches u_t

    # event stream: (nu, dA, dB)
    nu = np.concatenate([enter, leave])
    dA = np.concatenate([inv, -inv])
    dB = np.concatenate([-q * inv, q * inv + u])
    order = np.argsort(nu, kind="stable")
    nu, dA, dB = nu[order], dA[order], dB[order]

    A = np.cumsum(dA)              # slope of S on segment ending at nu[k+1]
    B = np.cumsum(dB)              # intercept
    # S evaluated at the right end of each segment
    nxt = np.append(nu[1:], nu[-1])
    S_right = A * nxt + B
    k = int(np.searchsorted(S_right, E, side="left"))
    k = min(k, len(A) - 1)
    Ak, Bk = A[k], B[k]
    if Ak <= 1e-300:                        # flat segment: fall back one step
        j = int(np.max(np.nonzero(A > 1e-300)[0])) if np.any(A > 1e-300) else k
        Ak, Bk = A[j], B[j]
    nu_star = (E - Bk) / Ak
    x = np.clip((nu_star - q) * inv, 0.0, u)

    resid = E - x.sum()
    if abs(resid) > 1e-12 * max(1.0, E):     # only at exact breakpoint ties
        room = (u - x) if resid > 0 else x.copy()
        tot = room.sum()
        if tot > 1e-15:
            x = x + np.sign(resid) * room * (abs(resid) / tot)
    return x


def _solve_on_active_set(q, c, u, E, R, A):
    """Minimise over the box, the total-energy equality, AND the prefix
    constraints in `A` forced to EQUALITY.  The forced equalities cut the
    horizon into independent segments, each a plain water-filling problem.

    Raises ValueError if a segment cannot carry the energy the equalities
    demand of it, which is how an infeasible guess at the active set is
    detected.
    """
    idx = sorted(A)
    x = np.empty(len(q))
    lo, carried = 0, 0.0
    for k in idx:
        hi = k + 1
        seg_E = R[k] - carried
        x[lo:hi] = _waterfill(q[lo:hi], c[lo:hi], seg_E, u[lo:hi])
        carried = R[k]
        lo = hi
    if lo < len(q):
        x[lo:] = _waterfill(q[lo:], c[lo:], E - carried, u[lo:])
    return x


def solve_sep_qp(q, c, X: Feasible, _tol=1e-11):
    """min sum_t (q_t x_t + c_t x_t^2) over X, exactly.

    The deadline staircase is handled by a primal ACTIVE-SET method over the
    prefix constraints.  Starting from none of them forced, we repeatedly force
    the most-violated prefix to equality; when the iterate is feasible we test
    each forced constraint for release and drop any whose removal both keeps
    feasibility and lowers the objective.  The forced equalities partition the
    horizon, so every subproblem is one water-filling call.

    The release step is the part that matters, and it is here because leaving
    it out is wrong.  An earlier version forced the most-violated prefix and
    recursed without ever reconsidering, on the reasoning that a constraint
    violated by the relaxation must be active at the optimum.  That is true of
    a single constraint and false of a chain of them: forcing prefix k to carry
    EXACTLY R_k also forbids it from carrying more, and when a shorter prefix
    is separately forced, or when the curvature c_t varies enough across slots
    to move where the relaxation puts its mass, the optimum wants more than
    R_k there.  E11's trace-derived feasible sets -- envelopes read off
    measured peak concurrency, staircases from a short deferral horizon
    against a bursty arrival profile -- hit exactly that case, and the old
    routine returned a point 1.5% above the block optimum, which showed up as a
    social cost BELOW the planner's, an impossibility that is what exposed it.
    Random instances with a flat envelope and a smooth staircase, which is what
    the synthetic generator produces and what tests/test_solver.py sampled,
    never triggered it.  `tests/test_solver.py` now carries that instance and
    samples the family it came from.
    """
    q = np.asarray(q, float)
    c = np.asarray(c, float)
    if np.any(c <= 0):
        raise ValueError("solve_sep_qp needs strictly positive curvature; "
                         "use solve_sep_lp for linear objectives")
    R = X.R
    if R is not None:
        R = np.maximum(R, 0.0)
        if R.max() <= 1e-12:
            R = None
    if R is None:
        return _waterfill(q, c, X.E, X.u)

    u, E = X.u, X.E
    obj = lambda z: float(q @ z + c @ (z * z))          # noqa: E731
    # Seed with the greedy set the old recursion would have chosen.  It is
    # usually the right active set and always a feasible one, so the loop below
    # normally has only release steps left to do; seeding cuts the number of
    # water-filling calls by about half against starting from the empty set.
    A = set()
    x = _waterfill(q, c, E, u)
    for _ in range(len(q)):
        viol = R - np.cumsum(x)
        k = int(np.argmax(viol))
        if viol[k] <= _tol * max(1.0, E) or k in A:
            break
        A.add(k)
        try:
            x = _solve_on_active_set(q, c, u, E, R, A)
        except ValueError:
            A.discard(k)
            x = _solve_on_active_set(q, c, u, E, R, A)
            break
    for _ in range(4 * len(q) + 8):
        viol = R - np.cumsum(x)
        k = int(np.argmax(viol))
        if viol[k] > _tol * max(1.0, E):
            if k in A:                                   # cannot make progress
                break
            A.add(k)
            try:
                x = _solve_on_active_set(q, c, u, E, R, A)
            except ValueError:
                A.discard(k)
                break
            continue
        # feasible: try to release each forced constraint
        improved = False
        for k in sorted(A):
            try:
                z = _solve_on_active_set(q, c, u, E, R, A - {k})
            except ValueError:
                continue
            if np.all(np.cumsum(z) >= R - _tol * max(1.0, E)) \
                    and obj(z) < obj(x) - 1e-13 * max(1.0, abs(obj(x))):
                A.discard(k)
                x = z
                improved = True
                break
        if not improved:
            return x
    return x


def solve_sep_lp(q, X: Feasible):
    """min sum_t q_t x_t over X.  Used by the signal-taking baselines, which
    have no congestion term and so face a linear objective."""
    from scipy.optimize import linprog
    T = X.T
    A_eq = np.ones((1, T))
    b_eq = [X.E]
    if X.R is not None and X.R.max() > 0:
        # -cumsum(x) <= -R
        A_ub = -np.tril(np.ones((T, T)))
        b_ub = -X.R
    else:
        A_ub, b_ub = None, None
    res = linprog(q, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0.0, ui) for ui in X.u], method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    return res.x


# --------------------------------------------------------------------------
# instance
# --------------------------------------------------------------------------

@dataclass
class Instance:
    """One (region, day) problem with n operators over T slots."""
    a: np.ndarray                     # published AEF, per slot
    m: np.ndarray                     # true MEF, per slot
    b: np.ndarray                     # response curvature, per slot
    X: list                           # list of Feasible, one per operator
    psi: np.ndarray                   # (n, T) deferral penalty shape
    lam: np.ndarray                   # (n,) penalty weight
    arrival: np.ndarray | None = None # (n, T) carbon-agnostic run-on-arrival
    label: str = ""

    def __post_init__(self):
        self.a = np.asarray(self.a, float)
        self.m = np.asarray(self.m, float)
        self.b = np.asarray(self.b, float)
        self.psi = np.asarray(self.psi, float)
        self.lam = np.asarray(self.lam, float)
        if np.any(self.m < self.a - 1e-9):
            raise ValueError("MEF must be >= AEF at the operating point")

    @property
    def n(self) -> int:
        return len(self.X)

    @property
    def T(self) -> int:
        return len(self.a)

    @property
    def eta(self) -> np.ndarray:
        """Unpriced share per slot, eta_t = 1 - a_t/m_t."""
        return 1.0 - self.a / self.m

    def poa_bound(self, active=None) -> float:
        """Theorem 1 bound, 3/(2(1-eta)) with eta = max over active slots."""
        e = self.eta if active is None else self.eta[active]
        return 1.5 / (1.0 - float(np.max(e)))

    def pen(self, x) -> float:
        return float(np.sum(self.lam[:, None] * self.psi * x))

    def social(self, x) -> float:
        """C(x), Eq. (8) in the (a, m, b) parameterisation."""
        y = x.sum(axis=0)
        return float(np.sum(self.m * y + self.b * y ** 2)) + self.pen(x)

    def potential(self, x) -> float:
        """Phi(x), Eq. (9)."""
        y = x.sum(axis=0)
        return float(np.sum(self.a * y + 0.5 * self.b * (y ** 2 + (x ** 2).sum(axis=0)))) \
            + self.pen(x)

    def agent_cost(self, x, i) -> float:
        """J_i(x), Eq. (5)."""
        y = x.sum(axis=0)
        return float(np.sum(x[i] * (self.a + self.b * y))
                     + self.lam[i] * np.sum(self.psi[i] * x[i]))

    def zeros(self):
        return np.zeros((self.n, self.T))

    def feasible_start(self):
        """A feasible profile: spread each operator's energy as evenly as the
        envelope and staircase allow."""
        x = np.zeros((self.n, self.T))
        for i, Xi in enumerate(self.X):
            x[i] = solve_sep_qp(np.zeros(self.T), np.ones(self.T), Xi)
        return x


# --------------------------------------------------------------------------
# equilibrium concepts -- all three are best response with a different
# linear coefficient, and the same curvature b_t.
# --------------------------------------------------------------------------

def _br_loop(inst: Instance, lin_coef, x0=None, damp=1.0, iters=4000,
             tol=1e-10, order="cyclic", rng=None, obj=None, ftol=1e-11,
             patience=2):
    """Gauss-Seidel best response.  `lin_coef(i, s)` returns the linear
    coefficient vector for operator i given the others' aggregate s.

    Two stopping rules, and which one is right depends on the objective.

      * On the ITERATE (`tol`), the historical rule.  Correct when the
        objective is strictly convex in the full profile.
      * On the OBJECTIVE VALUE (`obj`, `ftol`), used when it is not.

    The distinction is not cosmetic.  The measured curvatures span three orders
    of magnitude -- CAISO's beta runs from 5e-6 to 8e-3 across the day, because
    the estimated marginal factor barely moves with load in half the hours --
    so the potential is nearly flat in those coordinates and the iterate crawls
    while the value has already converged to twelve digits.  Waiting on the
    iterate turned a four-second solve into one that did not finish.  Where a
    caller passes `obj`, convergence is declared when the objective stops
    improving for `patience` consecutive sweeps, and the caller is expected to
    check the residual it actually cares about (for an equilibrium: the largest
    profitable unilateral deviation, which `max_gain` below reports).
    """
    x = inst.feasible_start() if x0 is None else x0.copy()
    y = x.sum(axis=0)
    idx = np.arange(inst.n)
    prev = obj(x) if obj is not None else None
    flat = 0
    for it in range(iters):
        delta = 0.0
        if order == "random":
            rng.shuffle(idx)
        for i in idx:
            s = y - x[i]
            xi = solve_sep_qp(lin_coef(i, s), inst.b, inst.X[i])
            if damp < 1.0:
                xi = (1 - damp) * x[i] + damp * xi
            delta = max(delta, float(np.max(np.abs(xi - x[i]))))
            y = s + xi
            x[i] = xi
        if obj is not None:
            cur = obj(x)
            if prev - cur <= ftol * max(1.0, abs(prev)):
                flat += 1
                if flat >= patience:
                    return x, it + 1
            else:
                flat = 0
            prev = cur
        elif delta <= tol * max(1.0, float(np.max(np.abs(x)))):
            return x, it + 1
    return x, iters


def max_gain(inst: Instance, x, lin_coef, curv=None):
    """Largest improvement any single operator can make, as a fraction of the
    objective the loop was minimising.  This is the residual that matters for an
    equilibrium claim, and it is what the tests check."""
    c = inst.b if curv is None else curv
    y = x.sum(axis=0)
    best = 0.0
    for i in range(inst.n):
        s = y - x[i]
        xi = solve_sep_qp(lin_coef(i, s), c, inst.X[i])
        xa = x.copy(); xa[i] = xi
        best = max(best, inst.agent_cost(x, i) - inst.agent_cost(xa, i))
    return best


def nash(inst: Instance, signal=None, **kw):
    """Unique pure Nash equilibrium = argmin Phi (Proposition 1).

    `signal` optionally replaces the common a_t with a per-operator perceived
    signal (n, T), which is how the precision experiments inject dispersion.
    """
    A = np.broadcast_to(inst.a, (inst.n, inst.T)) if signal is None else signal

    def lin(i, s):
        return A[i] + inst.b * s + inst.lam[i] * inst.psi[i]

    kw.setdefault("obj", inst.potential)
    return _br_loop(inst, lin, **kw)


def planner(inst: Instance, **kw):
    """Social optimum, argmin C."""
    def lin(i, s):
        return inst.m + 2.0 * inst.b * s + inst.lam[i] * inst.psi[i]

    kw.setdefault("obj", inst.social)
    return _br_loop(inst, lin, **kw)


def wedge_fixed(inst: Instance, **kw):
    """The intermediate regime that makes the gap decomposition exact
    (REVIEW.md Move 1).

    Agents are charged the marginal factor level m_t but still see only their
    own congestion, i.e. the perceived per-unit cost is m_t + b_t y_t.  The
    perceived marginal is then m + b*y + b*x_i against a social marginal of
    m + 2*b*y, so the residual gap is exactly b*(y - x_i): the congestion
    externality, with the accounting wedge (m - a) removed.

    Comparing Nash -> wedge_fixed -> planner therefore splits the equilibrium
    gap into an accounting component and a strategic component, which is the
    split a reviewer will demand once they notice that eta > 0 holds even at
    n = 1 (REVIEW.md P2).
    """
    def lin(i, s):
        return inst.m + inst.b * s + inst.lam[i] * inst.psi[i]

    kw.setdefault("obj", lambda x: inst.social(x))
    return _br_loop(inst, lin, **kw)


def mef_signal_equilibrium(inst: Instance, **kw):
    """Baseline B8: agents charged the *responsive* marginal factor
    m_t + 2 b_t y_t instead of the average factor.  This is the fix the paper
    itself recommends in its discussion, and it over-corrects: the resulting
    perceived marginal is m + 2 b y + 2 b x_i."""
    def lin(i, s):
        return inst.m + 2.0 * inst.b * s + inst.lam[i] * inst.psi[i]

    # objective x(m + 2b(s+x)) = (m + 2bs) x + 2b x^2  -> curvature 2b
    x = inst.feasible_start()
    y = x.sum(axis=0)
    for it in range(4000):
        delta = 0.0
        for i in range(inst.n):
            s = y - x[i]
            xi = solve_sep_qp(lin(i, s), 2.0 * inst.b, inst.X[i])
            delta = max(delta, float(np.max(np.abs(xi - x[i]))))
            y = s + xi
            x[i] = xi
        if delta <= 1e-10 * max(1.0, float(np.max(np.abs(x)))):
            return x, it + 1
    return x, 4000


def nash_two_slot(inst: Instance, signal=None, iters=200, tol=1e-14):
    """Exact equilibrium solver for the two-slot family with a common curvature.

    Best response reduces to a function of the single scalar y_1.  Equalising
    the two perceived marginals subject to x_i1 + x_i2 = E_i gives

        x_i1(y_1) = clip( [ dA_i + b*(S - 2*y_1) + b*E_i ] / (2*b), 0, E_i )

    with dA_i = A_i2 - A_i1 and S = sum_j E_j.  So the equilibrium is the root
    of F(y_1) = sum_i x_i1(y_1) = y_1.  Each clipped term is non-increasing in
    y_1, hence F(y_1) - y_1 is strictly decreasing and the root is unique --
    which is Proposition 1 specialised to two slots, and lets us bisect instead
    of iterating best response.

    (A damped Jacobi iteration does NOT work here: the aggregate map has gain
    -n/2, so it is expansive for every n > 1.  That is precisely why the
    general solver uses Gauss-Seidel.)
    """
    if inst.T != 2:
        raise ValueError("nash_two_slot requires T == 2")
    b1, b2 = float(inst.b[0]), float(inst.b[1])
    if abs(b1 - b2) > 1e-12 * max(b1, b2, 1.0):
        raise ValueError("nash_two_slot requires b_1 == b_2")
    b = b1
    n = inst.n
    E = np.array([Xi.E for Xi in inst.X])
    S = E.sum()
    A = np.broadcast_to(inst.a, (n, 2)).astype(float) if signal is None         else np.asarray(signal, float).copy()
    A = A + inst.lam[:, None] * inst.psi
    dA = A[:, 1] - A[:, 0]

    def F(y1):
        return np.clip((dA + b * (S - 2.0 * y1) + b * E) / (2.0 * b), 0.0, E).sum()

    lo, hi = 0.0, S
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if F(mid) - mid > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, S):
            break
    y1 = 0.5 * (lo + hi)
    x1 = np.clip((dA + b * (S - 2.0 * y1) + b * E) / (2.0 * b), 0.0, E)
    return np.stack([x1, E - x1], axis=1), 1
