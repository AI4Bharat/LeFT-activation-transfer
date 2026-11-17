"""The statistics the paper reports, as functions.

Every headline number in the manuscript is a mean over five seeds with a
standard deviation, and every claim of the form "FAct beats X" is a *paired*
t-test across the shared seeds, Holm-corrected over the four datasets. The
pairing matters: seed 3 of the transferred arm and seed 3 of the GELU arm saw
the same data order and the same initialisation stream, so the difference
between them is a much tighter quantity than the difference of two means. The
Holm correction matters because four datasets means four tests, and an
uncorrected "significant on one of four" is not evidence.
"""
import math

try:
    from scipy.stats import ttest_rel as _ttest_rel
except Exception:          # scipy is optional for the stats path
    _ttest_rel = None


def top1(correct, total):
    return correct / max(1, total)


def mean_sd(xs):
    """Sample mean and standard deviation (ddof=1, the seed-spread convention
    used throughout the paper)."""
    xs = list(xs)
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def paired_t(a, b):
    """Two-sided paired t-test of `a` against `b` (same seeds, same order).
    Returns `(t, p)`. Falls back to a normal approximation without scipy."""
    if len(a) != len(b):
        raise ValueError(f"paired test needs equal lengths, got {len(a)}, {len(b)}")
    if _ttest_rel is not None:
        r = _ttest_rel(a, b)
        return float(r.statistic), float(r.pvalue)
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    m = sum(diffs) / n
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1)) if n > 1 else 0.0
    if sd == 0.0:
        return (float("inf") if m else 0.0), (0.0 if m else 1.0)
    t = m / (sd / math.sqrt(n))
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return t, p


def holm(pvalues, alpha=0.05):
    """Holm-Bonferroni step-down correction.

    Returns a list of `(adjusted_p, reject)` in the input order. Holm rather
    than Bonferroni because it is uniformly more powerful at the same
    family-wise error rate, and the family here is small (four datasets).
    """
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adj[i] = min(1.0, running)
    return [(adj[i], adj[i] < alpha) for i in range(m)]


def compare_arms(results, arm, baseline, alpha=0.05):
    """The paper's comparison, end to end.

    `results` maps dataset -> arm name -> {seed: accuracy}. Returns a dict per
    dataset with both arms' mean+-sd, the margin in percentage points, the
    paired t/p over the shared seeds, and the Holm-adjusted p over the datasets
    in `results`.
    """
    datasets = list(results)
    rows, praw = {}, []
    for ds in datasets:
        A, B = results[ds][arm], results[ds][baseline]
        seeds = sorted(set(A) & set(B))
        a = [A[s] for s in seeds]
        b = [B[s] for s in seeds]
        am, asd = mean_sd(a)
        bm, bsd = mean_sd(b)
        t, p = paired_t(a, b)
        rows[ds] = {"seeds": seeds, "n": len(seeds),
                    arm: (am, asd), baseline: (bm, bsd),
                    "margin_pp": 100.0 * (am - bm), "t": t, "p_raw": p}
        # A single shared seed (or two identical arms) gives an undefined
        # t-test; carry the nan through rather than letting it sort as the
        # smallest p and come back "significant".
        praw.append(p if math.isfinite(p) else 1.0)
    for ds, (p_adj, rej) in zip(datasets, holm(praw, alpha=alpha)):
        rows[ds]["p_holm"] = p_adj
        rows[ds]["significant"] = bool(rej) and math.isfinite(rows[ds]["p_raw"])
    return rows
