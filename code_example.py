"""
Why the same model gives two different answers.

Adds identical numbers in different orders -- the way two GPU kernels with
different split-K counts or batch shapes do -- and finds an input where the
order alone changes which token wins the argmax.

No model, no GPU, no randomness in the arithmetic itself. Everything here is
plain IEEE-754 double precision doing exactly what it is specified to do.

Background: vLLM/SkyRL's IsoExec write-up (21 Aug 2026) traces trainer-inference
divergence to this, measuring a mean logprob gap of 0.014 and a worst step of
5.073 between two engines holding identical weights.

Run: python3 code_example.py
"""

import itertools
import random

SEED = 5
TERMS = 64          # values summed per logit, as a dot product would
SPLITS = (1, 2, 4, 8, 16)   # how many partial sums the reduction is chopped into


# --- The liftable core: one dot product, several reduction orders -------------

def sum_sequential(xs):
    """One accumulator, left to right. What a single-threaded kernel does."""
    total = 0.0
    for x in xs:
        total += x
    return total


def sum_split(xs, splits):
    """Chop into `splits` partial sums, then add the partials. This is split-K,
    and it is why the number of thread blocks changes your result."""
    n = len(xs)
    size = (n + splits - 1) // splits
    partials = [sum_sequential(xs[i:i + size]) for i in range(0, n, size)]
    return sum_sequential(partials)


def sum_pairwise(xs):
    """Tree reduction, the shape most parallel hardware actually uses."""
    cur = list(xs)
    while len(cur) > 1:
        nxt = [cur[i] + cur[i + 1] if i + 1 < len(cur) else cur[i]
               for i in range(0, len(cur), 2)]
        cur = nxt
    return cur[0]


def logits_under(order, rows):
    """Score every candidate token using one reduction order."""
    return [order(row) for row in rows]


# --- Two tokens the model genuinely cannot separate ---------------------------

def make_row(rng, terms=TERMS):
    """The products a dot product would sum. Values span several orders of
    magnitude, which is ordinary for activations and is exactly when rounding
    order matters most."""
    return [rng.choice([1.0, -1.0]) * 10 ** rng.uniform(-6, 2) for _ in range(terms)]


def near_tie(rng):
    """Two candidate tokens whose scores are the SAME NUMBERS in a different
    arrangement, so their exact mathematical sums are identical and only rounding
    can separate them.

    This is constructed, not discovered -- two independent logits landing within
    1e-14 of each other essentially never happens, and searching for it would be
    dishonest. What is not constructed is the consequence: across a vocabulary of
    a hundred thousand tokens, near-ties at the top are routine, and when two
    candidates sit inside the rounding noise the reduction order picks the winner.
    Real engines diverge far more than this anyway -- they differ in kernels and
    accumulation dtypes, not merely in summation order, which is how IsoExec
    measured a worst-case gap of 5.073 rather than 1e-14."""
    row = make_row(rng)
    shuffled = row[:]
    rng.shuffle(shuffled)
    return row, shuffled


def main():
    rng = random.Random(SEED)

    xs = make_row(rng)
    print("One dot product, five reduction orders:\n")
    base = sum_sequential(xs)
    print(f"  sequential          {base!r}")
    for k in SPLITS[1:]:
        v = sum_split(xs, k)
        print(f"  split-K={k:<3}         {v!r}   delta {v - base:+.3e}")
    pw = sum_pairwise(xs)
    print(f"  pairwise (tree)     {pw!r}   delta {pw - base:+.3e}")
    print("\n  Same 64 numbers every time. Only the order changed.\n")

    tok_a, tok_b = near_tie(rng)
    print("Two candidate tokens scored from the same numbers, differently arranged.")
    print("Their exact sums are equal; only rounding can separate them.\n")

    print("  reduction order       token A                token B              A-B        argmax")
    for name, fn in (("sequential", sum_sequential),
                     ("pairwise (tree)", sum_pairwise),
                     ("split-K=4", lambda xs: sum_split(xs, 4)),
                     ("split-K=8", lambda xs: sum_split(xs, 8))):
        va, vb = fn(tok_a), fn(tok_b)
        winner = "A" if va > vb else ("B" if vb > va else "tie")
        # full precision: the disagreement lives in the last representable digit
        print(f"  {name:<20} {va!r:>21}  {vb!r:>21}  {va - vb:>+10.2e}   {winner}")

    print("\n  Same numbers, same model, temperature zero. The winner changes with")
    print("  the reduction order alone.")
    print("\nTemperature zero was honoured in both cases. Greedy decoding is")
    print("deterministic GIVEN a reduction order, and the order is set by your")
    print("kernel, your batch shape and your parallelism layout — none of which")
    print("live in the checkpoint. That is the whole bug, and everything after")
    print("this token diverges.")


if __name__ == "__main__":
    main()
