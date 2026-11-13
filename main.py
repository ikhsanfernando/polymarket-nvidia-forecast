"""Empirical change-quantile forecast for Polymarket's Next-CEO event: quiet minutes stay cheap."""
import json
import math
import os

IN, OUT = os.environ.get("MAKEFUN_INPUT_DIR", "/input"), os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
Q = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
HS = [1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 180, 240]
SHRINK, TAIL, BLEND, FLOOR, HORIZON = 0.2, 0.35, 0.2, 0.001, 240
OWN, HOUSE = 0.9, 0.0025  # weight on this model, house random-walk sigma blended back in


def qtl(v, q):
    i = q * (len(v) - 1)
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return v[lo] if lo == hi else v[lo] + (v[hi] - v[lo]) * (i - lo)


def table(prices):
    tab = {}
    for h in HS:
        if len(prices) > h + 5:
            d = sorted(prices[i + h] - prices[i] for i in range(len(prices) - h))
            tab[h] = [qtl(d, q) for q in Q]
    return tab or {1: [0.0] * 5}


def clean(prices):
    """Keep finite numeric prices and clamp each to [0, 1]."""
    # NaNs break ordering and can contaminate quantiles, so remove them before forecasting.
    return [min(1.0, max(0.0, p)) for p in prices if isinstance(p, (int, float)) and math.isfinite(p)]


def forecast(prices):
    prices = clean(prices)
    if not prices:
        return [[0.45, 0.475, 0.5, 0.525, 0.55] for _ in range(HORIZON)]
    x, tab, keys = prices[-1], table(prices), sorted(table(prices))
    floor = FLOOR * math.sqrt(max(1e-6, x * (1 - x)))
    shrink = [TAIL, TAIL, SHRINK, TAIL, TAIL]
    rows = []
    for h in range(1, HORIZON + 1):
        lo = max([k for k in keys if k <= h], default=keys[0])
        hi = min([k for k in keys if k >= h], default=keys[-1])
        w = 0 if lo == hi else (h - lo) / (hi - lo)
        emp = [tab[lo][i] if lo == hi else tab[lo][i] + (tab[hi][i] - tab[lo][i]) * w for i in range(5)]
        band = [z * floor * math.sqrt(h) for z in Z]
        mix = [emp[i] * shrink[i] * (1 - BLEND) + band[i] * BLEND for i in range(5)]
        house = [z * HOUSE * math.sqrt(h) for z in Z]
        row = [x + OWN * min(mix[0], band[0]) + (1 - OWN) * house[0],
               x + OWN * min(mix[1], band[1]) + (1 - OWN) * house[1],
               x + OWN * mix[2] + (1 - OWN) * house[2],
               x + OWN * max(mix[3], band[3]) + (1 - OWN) * house[3],
               x + OWN * max(mix[4], band[4]) + (1 - OWN) * house[4]]
        rows.append(sorted(min(1.0, max(0.0, round(v, 6))) for v in row))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in sorted(n for n in os.listdir(IN) if n.endswith(".in")):
        try:
            with open(os.path.join(IN, name)) as f:
                case = json.load(f)
            out = [{"id": m["id"], "q": forecast(m.get("prices") or [])} for m in case.get("markets") or []]
        except Exception:
            out = []
        with open(os.path.join(OUT, name[:-3] + ".out"), "w") as f:
            f.write(json.dumps({"forecasts": out}, separators=(",", ":")))


if __name__ == "__main__":
    main()
