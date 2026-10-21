# webagg-sports — WebAggr in the NBA domain (Experiment 2-S)

A from-scratch instantiation of the WebAggr pipeline (webaggr.pdf) for the
query **"What is [player]'s total career regular-season points in the NBA?"**
— f_Q = SUM over (player, season) records — with **official NBA stats as the
withheld oracle** (the EDGAR role: defines truth, denied at run time).

Purpose: Exp 2-S, a control experiment for the finance Exp 2 failure. See
`experiments/exp2s/PREREGISTRATION_EXP2S.md` — read it first; it is the
contract for every run.

## Layout

```
webaggs/                 the pipeline package
  config.py              parameters (impl-guide Table 4 names)
  canon.py               typed validators: names, seasons, points (§7.1)
  frontier.py            Û, ψ, the two-conjunct stop test (§3.2–3.3)
  claims.py              COUNT/SUM checksums, gap direction, brake (§5.2)
  pipeline.py            FrontierDiscovery loop + end_to_end wiring (§14.1)
  replay.py              offline stop replay from frozen ledgers (Exp 2)
  extract.py, conformal.py   certified reading + δ_E gate (§7.1)
  er.py, corroborate.py  deterministic ER + echo-collapse corroboration
  aggregate.py           SUM with the honest (ε_C + ε_F) interval (§7.3)
  oracle_nba.py, truth.py   the withheld oracle, freeze, exact grading
  storage.py             per-run SQLite ledger (the replay substrate)
  fetch.py               deny list + fetch-outcome classifier (paywall instrument)
  llm.py, search.py      single-provider LLM, Serper
notebooks/
  exp2s_stopping_calibration.ipynb   Exp 2-S, end to end: gated live cell,
                         S1/S2 self-tests, C1 calibration plot, C2 coverage
                         table, shortfall decomposition, grading
scripts/
  build_cohort.py        freeze the truth table + cohort report
  run_exp2s.py           LIVE phase (gated by RUN_LIVE=1)
  replay_exp2s.py        OFFLINE arms + self-tests S1/S2 + C1/C2 tables
  grade_exp2s.py         exact grading vs the frozen key
tests/                   87-test suite, 9 layers (see TESTING.md) — the gate;
                         it also EXECUTES the notebook headlessly against a
                         synthetic cohort, so the notebook itself is gated
```

The notebook and the scripts are two entry points over the SAME ledgers and
run index (identical run_id scheme), so you can run live from either and
replay from either. Knobs: `WEBAGG_DATA_DIR` relocates all experiment data
(used by the tests for isolation); `WEBAGG_COHORT` overrides the notebook's
cohort for pilots without editing the pre-registered constant.

## Setup (PyCharm / Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env     # fill in OPENAI_API_KEY, SERPER_API_KEY
pytest                     # the gate: green before any live spend
```

Load `.env` into the shell before live runs (PyCharm run configs can set env
vars directly; nothing in the code reads `.env` implicitly).

## Calibration (before the first live run)

1. **Conformal gate** (`data/conformal/calibration.csv`, columns
   `confidence,correct`): run the extractor on a pilot player **not in the
   cohort**, label ≥ 20 correct rows by hand against the oracle, save. The
   gate refuses to run unfitted — by design.
2. **ε_F is NOT calibrated** in this domain. Reports carry
   `eps_f = None` and print no two-term interval. Do not invent a number.

## Online sanity preflight (once, from the machine that will run live)

```powershell
$env:RUN_ONLINE="1"; pytest -m online -v
```

Five checks, skipped on normal `pytest` runs: the oracle answers and its
rows parse (a dry run of the truth builder), a public page fetches as
`ok`, the deny list blocks the real oracle, and — when the keys are set —
one tiny Serper search (~$0.001) and one tiny LLM call (<$0.01). Run it
before `build_cohort.py` and again before `RUN_LIVE=1`; a red here means a
network/key problem, caught before any spend.

## Running Exp 2-S (order is part of the pre-registration)

```powershell
pytest                                                   # 1. gate
python scripts/build_cohort.py --cohort nba_v1 --candidates cohort/candidates.txt
#   -> apply the pre-registered selection rule to cohort_report.csv,
#      finalize candidates, rebuild under the final cohort name, FREEZE
# 2. supervisor signs PREREGISTRATION_EXP2S.md §4; commit it
$env:RUN_LIVE="1"
python scripts/run_exp2s.py    --cohort nba_v1           # 3. live (ε = 0.05)
python scripts/replay_exp2s.py --cohort nba_v1           # 4. offline arms + C1/C2
python scripts/grade_exp2s.py  --cohort nba_v1           # 5. exact grading
```

Idempotent: `run_exp2s.py` skips finished players (`runs_index.json`); no
re-rolling after seeing results.

## Design notes carried over from the finance build (why the code looks like this)

- **Occasions, not sources** — `covered[record_key]` holds *formulation*
  ids; counting sources collapses f₁ and certifies early (impl-guide
  pitfall #1). There is a regression test.
- **ε is read only in the stop test** — that is what makes the offline
  looser-ε replay arms valid from one live ledger.
- **`certified` is `""`, never `None`** — pandas coerces None→NaN and NaN is
  truthy; the replay script avoids pandas entirely.
- **Raw LLM replies are always persisted** (`llm_calls` table) and
  `call_llm` returns the parsed payload directly — no `{"payload": ...}`
  envelope for a consumer to misread.
- **Prompts anchor to the package dir**, never the process CWD.
- **Loud failure everywhere**: unfitted conformal gate, missing keys, oracle
  rate-limit blocks, non-JSON LLM replies and a tampered frozen truth table
  all raise; nothing degrades silently.
- **Paywall/denial outcomes are first-class data** (`fetch_outcomes` table):
  a censored fetch is logged as censored, never recorded as evidence of
  absence — the instrument the finance diagnosis needed.
- **Crash-resume hygiene**: a ledger left by a run that crashed mid-player is
  moved aside (`<run_id>.crashed-N.sqlite`, kept for diagnosis) before the
  rerun — a second trajectory appended under one run_id would interleave
  steps and corrupt every offline replay. Both entry points do this.
