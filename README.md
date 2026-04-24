# ProsperityRepo
Link to wiki: https://imc-prosperity.notion.site/prosperity-4-wiki

## Backtester

`PYTHONPATH` must include:

1. **`imc-prosperity-4-backtester`** (backtester root) so `python -m prosperity4bt` finds the `prosperity4bt` package.
2. **`imc-prosperity-4-backtester/prosperity4bt`** so `from datamodel import …` in your trader resolves.

Do **not** use a literal `...` path. Do **not** use `../Prosperity4Data` unless your shell’s current directory is **`imc-prosperity-4-backtester`** (otherwise `..` is wrong).

**If you skip `PYTHONPATH`, you always get `No module named 'datamodel'`.** The `PYTHONPATH=...` line below is required — copy it with the `python3` line.

### Copy-paste from repo root (`ProsperityRepo`)

Buy-hold smoke test:

```bash
PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \
python3 -m prosperity4bt \
  "$PWD/Round1/pepper_buy_hold_t0_80.py" 1 \
  --data "$PWD/Prosperity4Data" \
  --match-trades all \
  --no-vis
```

Rolling-slope pepper trader (same `PYTHONPATH` — do not omit):

```bash
PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \
python3 -m prosperity4bt \
  "$PWD/Round1/pepper_rolling_slope_trader.py" 1 \
  --data "$PWD/Prosperity4Data" \
  --match-trades all \
  --no-vis
```

Drift MM (emerald-style) for pepper:

```bash
PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \
python3 -m prosperity4bt \
  "$PWD/Round1/pepper_mm_emeraldstyle_drift.py" 1 \
  --data "$PWD/Prosperity4Data" \
  --match-trades all \
  --no-vis
```

### Alternative: `cd` first, then relative paths

```bash
cd imc-prosperity-4-backtester

PYTHONPATH="$PWD/prosperity4bt" python3 -m prosperity4bt \
  ../Round1/pepper_buy_hold_t0_80.py 1 \
  --data ../Prosperity4Data \
  --match-trades all \
  --no-vis
```

Run **`cd imc-prosperity-4-backtester`** before the second block; if you stay in `ProsperityRepo`, `../Prosperity4Data` does not exist.

### Flags

- **`<DAYS>`** — e.g. `0`, `1`, `1--2` (round 1, day -2).
- **`--match-trades`** — `all` (default if omitted), `worse`, or `none`.
- **`--no-vis`** — do not open the web visualizer after the run (omit if your `prosperity4bt` build does not support this flag).
