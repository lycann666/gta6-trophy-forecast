# GTA VI Trophy Rarity Forecast

[![Competition](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/lycann666/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/lycann666/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for the [make.fun bounty "Forecast how many GTA VI players earn each trophy in week one"](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): forecast, for every GTA VI trophy on PlayStation 5, its earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles (5/25/50/75/95%) per trophy.

## How the forecaster works

`main.py` (Python 3.13, standard library only, no network, deterministic) reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input/`) and writes `/output/<id>.out` per case.

For each trophy it works on the logit scale: it fits a least-squares line of logit(earn rate) against log-compressed time since the first read, extrapolates that line to the target with the slope damped and the total move capped (rates saturate — they cannot pass 100%), and centres a flat history on its last rate (persistence). The spread combines the fit's residual scatter, a random-walk widening of 0.5·√days, and a refresh-jump floor that is wider when the history is flat and uninformative — Exophase refreshes its counts in discrete jumps, so a narrow band pays for every trophy a refresh moves. Edge cases (null rates, 0% / 100% rates, trophies appearing mid-history) are handled; output is always 55 rows of five non-decreasing quantiles in [0, 100].

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 -P -s main.py
```

## License

MIT — see [LICENSE](LICENSE).
