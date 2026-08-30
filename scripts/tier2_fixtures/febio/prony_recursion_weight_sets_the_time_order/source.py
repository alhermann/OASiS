"""febio::viscoelasticity#5 — the Prony recursion weight sets the time order.

Every code integrates h(t) = int_0^t exp(-(t-s)/tau) d(sigma_e)/ds ds by the
same exponential recursion; only the weight W on the driving increment differs.
The claim is that W = 1 is first order, while the midpoint and trapezoid weights
are second order, and that none of them announces itself.

Intentional-failure arm: the endpoint weight, whose measured slope is 1 when the
study asks for 2. It converges and raises nothing, which is the pathology.

Also pins the reconciliation with pitfall #4: on a HELD load the three weights
agree, so a step-insensitive plateau is not evidence of second order.
"""
from math import exp, sin

TAU = 0.3
WEIGHTS = {
    "endpoint":  lambda dt: 1.0,
    "midpoint":  lambda dt: exp(-dt / (2.0 * TAU)),
    "trapezoid": lambda dt: 0.5 * (1.0 + exp(-dt / TAU)),
}


def integrate(nstep, T, drive, kind):
    dt = T / nstep
    decay, w, h = exp(-dt / TAU), WEIGHTS[kind](dt), 0.0
    for n in range(nstep):
        h = decay * h + w * (drive((n + 1) * dt) - drive(n * dt))
    return h


def exact_sin(T):
    a = 1.0 / TAU
    return (a * __import__("math").cos(T) + sin(T)
            - exp(-a * T) * a) / (a * a + 1.0)


def slope(kind, T=1.0):
    """Least-squares convergence order over a halving sequence."""
    ref = exact_sin(T)
    errs, steps = [], [8, 16, 32, 64, 128, 256]
    for n in steps:
        errs.append(abs(integrate(n, T, sin, kind) - ref))
    orders = [__import__("math").log2(errs[i] / errs[i + 1])
              for i in range(len(errs) - 1)]
    return orders[-1], errs


def main():
    for kind, want in (("endpoint", 1), ("midpoint", 2), ("trapezoid", 2)):
        p, errs = slope(kind)
        print(f"ORDER_{kind.upper()}={p:.2f} "
              f"coarsest_err={errs[0]:.3e} finest_err={errs[-1]:.3e}")

    # midpoint must be the most accurate of the two second-order forms
    _, e_mid = slope("midpoint")
    _, e_trap = slope("trapezoid")
    print(f"TRAPEZOID_OVER_MIDPOINT={e_trap[0] / e_mid[0]:.1f}")

    # HELD load: the weights become indistinguishable, which is why a flat
    # plateau proves nothing about the order.
    held = lambda s: min(s / 0.05, 1.0)
    for nstep in (40, 160, 640):
        v = [integrate(nstep, 5.0, held, k) for k in WEIGHTS]
        print(f"HELD_SPREAD_{nstep}={max(v) - min(v):.2e}")


if __name__ == "__main__":
    main()
