#!/usr/bin/env python3
"""GTA VI trophy earn-rate forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads every *.in case in $MAKEFUN_INPUT_DIR (else /input/) and writes one
forecast per case to $MAKEFUN_OUTPUT_DIR (else /output/) as
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.

Model (written from the competition brief, standard library only,
deterministic, no network, no randomness):

For each trophy, work on the logit scale, logit(p) = ln(p / (1 - p)) with
rates clamped to [0.01, 99.99] first.  Fit, by least squares, a straight line
of logit(rate) against x = ln(1 + hours since the first read of the case),
and read the line at the target instant.  The fitted slope is damped and
the extrapolated move is capped, because earn rates saturate: a trophy
cannot pass 100%, and growth slows as a trophy gets very common or stays
very rare.  A trophy whose history is flat keeps its last rate as its
median (persistence), which is the honest centre when the page has not
moved -- but the spread is then kept deliberately WIDE, because Exophase
refreshes its counts in discrete jumps: the practice cases prove a refresh
can move every rate at once by more than a logit, and a narrow band pays
for every trophy outside it.

The spread (logit sigma) combines three honest sources of doubt:
  * the fit's residual scatter (how noisy the trophy's own history is),
  * a random-walk widening with horizon, 0.5 * sqrt(days), and
  * a refresh-jump floor: Exophase may refresh between the last read and
    the target, so sigma never drops below a floor that is larger when the
    history is flat/uninformative (the data then says nothing about the
    direction of the next jump) and smaller once a stable trend is visible.

Trophies missing from some reads (null rates) simply contribute fewer
points; a trophy with one point, or none, falls back to persistence (or
1% with a wide spread when never seen).  Quantiles use the standard
normal levels for 0.05/0.25/0.5/0.75/0.95 and are rounded to 4 decimals,
forced non-decreasing, and clamped to [0, 100].
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

LO = 0.01
HI = 99.99
LOGIT_LO = math.log(LO / (100.0 - LO))
LOGIT_HI = math.log(HI / (100.0 - HI))

# Random-walk widening per sqrt(day), on the logit scale.
RW_PER_SQRT_DAY = 0.5
# Refresh-jump floor on sigma: flat/uninformative history vs stable trend.
SIGMA_FLOOR_FLAT = 0.95
SIGMA_FLOOR_TREND = 0.55
# How far the fitted line may move the median, in logit units.
MAX_EXTRAP_MOVE = 2.5
# Slope damping: shrink the fitted slope a little so a short noisy
# history does not extrapolate wildly over a week.
SLOPE_SHRINK = 0.85


def clamp_rate(r):
    if r is None:
        return None
    if not isinstance(r, (int, float)) or not math.isfinite(r):
        return None
    return min(HI, max(LO, float(r)))


def logit_pct(r):
    p = r / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(l):
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def parse_time(s):
    """Parse an ISO-8601 UTC timestamp; return epoch seconds or None."""
    if not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def fit_line(xs, ys):
    """Least-squares y = a + b*x. Returns (a, b, resid_std, n)."""
    n = len(xs)
    if n == 0:
        return None
    if n == 1:
        return (ys[0], 0.0, 0.0, 1)
    # A history that never moved is exactly flat: do not let floating-point
    # noise in the mean invent a phantom slope from identical values.
    if max(ys) - min(ys) < 1e-9:
        return (ys[0], 0.0, 0.0, n)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return (my, 0.0, 0.0, n)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        resid = math.sqrt(max(0.0, var))
    else:
        resid = 0.0
    return (a, b, resid, n)


def forecast_trophy(points, x_target, days):
    """points: list of (x, logit_rate) sorted by x. Returns 5 quantiles."""
    if not points:
        # Never seen: centre on 1% with a very wide band.
        med = logit_pct(1.0)
        sigma = 1.6
    else:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        a, b, resid, n = fit_line(xs, ys)
        last_logit = ys[-1]
        if n < 2 or b == 0.0:
            med = last_logit
            informative = False
        else:
            b_damped = b * SLOPE_SHRINK
            move = b_damped * (x_target - xs[-1])
            move = max(-MAX_EXTRAP_MOVE, min(MAX_EXTRAP_MOVE, move))
            med = last_logit + move
            informative = True
        med = max(LOGIT_LO, min(LOGIT_HI, med))
        # Spread: residual scatter + random walk + refresh floor.
        rw = RW_PER_SQRT_DAY * math.sqrt(max(days, 0.0))
        floor = SIGMA_FLOOR_TREND if informative else SIGMA_FLOOR_FLAT
        sigma = math.sqrt(resid * resid + rw * rw + floor * floor)
        # A flat history with a long horizon deserves extra width: the
        # longer the page sits unrefreshed, the bigger the next jump can be.
        if not informative:
            sigma = max(sigma, SIGMA_FLOOR_FLAT + 0.25 * math.sqrt(max(days, 0.0)))
    qs = []
    for z in Z:
        q = inv_logit_pct(med + z * sigma)
        q = round(min(100.0, max(0.0, q)), 4)
        qs.append(q)
    # Enforce non-decreasing after rounding.
    for i in range(1, 5):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_min = case.get("horizon_minutes") or 0
    try:
        horizon_min = float(horizon_min)
    except (TypeError, ValueError):
        horizon_min = 0.0
    days = horizon_min / 1440.0

    # Time axis: hours since the first read, log-compressed.
    read_times = [parse_time(h.get("at")) for h in history]
    t0 = None
    for t in read_times:
        if t is not None:
            t0 = t
            break
    if t0 is None:
        t0 = parse_time(case.get("at"))
    target_t = parse_time(case.get("target"))
    if t0 is not None and target_t is not None:
        x_target = math.log1p(max(0.0, (target_t - t0) / 3600.0))
    else:
        x_target = math.log1p(max(0.0, horizon_min / 60.0))

    rows = []
    for idx, trophy in enumerate(trophies):
        tid = trophy.get("id", idx + 1)
        points = []
        for h, t in zip(history, read_times):
            rates = h.get("rates") or []
            if idx >= len(rates):
                continue
            r = clamp_rate(rates[idx])
            if r is None or t is None:
                continue
            x = math.log1p(max(0.0, (t - t0) / 3600.0))
            points.append((x, logit_pct(r)))
        points.sort()
        rows.append({"id": tid, "q": forecast_trophy(points, x_target, days)})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    if not names:
        print("no .in case files found in %s" % in_dir, file=sys.stderr)
        return 1
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        out = forecast_case(case)
        out_name = name[:-3] + ".out"
        data = json.dumps(out, separators=(",", ":"), ensure_ascii=False)
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as f:
            f.write(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
