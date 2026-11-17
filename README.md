# webagg-citations — WebAggr in the citation domain (Experiment 2-C)

**This branch** runs the WebAggr pipeline (webaggr.pdf) on the query
*"which DOI-registered papers cite [target paper], up to a cutoff
year?"* (f_Q = COUNT). It is the experiment that tests the
**statistical stopping guarantee** (Theorem 1) at a size where the
certificate can actually fire. The NHL instantiation (Exp 2-S, complete)
lives on the `sports` branch.

Read `experiments/exp2c/PREREGISTRATION_EXP2C.md` first — it is the
contract for every run.

## The design in one screen

| Role | Source | Notes |
|---|---|---|
| **Oracle** (defines truth, DENIED to the agent) | OpenCitations | free, no key; returns the whole citing list at once |
| **Enumerator 1** | OpenAlex | cursor paging; free key (`OPENALEX_API_KEY`) |
| **Enumerator 2** | Europe PMC | `CITES:` search, cursor paging; no key; biomedicine only |
| Optional enumerator 3 | Semantic Scholar | needs `S2_API_KEY`; off unless pre-registered |
| Everything else | the open web via Serper | prose pages; read by the LLM reader + conformal gate |

How a run works:

1. The agent walks each enumerator **one page per capture occasion**
   (Def. 5). The page size it requests (50) is the per-occasion novelty
   bound **Y_CAP** by construction (Assumption 1b).
2. Pages are emitted **cursor-style**: page k+1 becomes a formulation only
   after page k is read, so the frontier stays "hot" until a walk ends.
3. A record seen by only one enumerator is a **singleton** (f₁). After
   both walks, Û ≈ the singleton share u.
4. The unchanged stop test fires when **Û + ψ < ε** and the frontier is
   cold. ψ shrinks like 1/N, so the paper needs enough citers — and the
   walks must fit inside τ_max. `webaggs/design.py` computes that band.
5. Recall at the stop is graded against the frozen OpenCitations truth.

## Layout

```
webaggs/                 the pipeline package
  config.py              parameters (impl-guide Table 4 names)
  canon.py               typed validators: DOIs, titles (§7.1)
  frontier.py            Û, ψ, the two-conjunct stop test (§3.2–3.3)
  indexes.py             paginated citation indexes as capture occasions
  design.py              feasibility arithmetic for the statistical stop
  pipeline.py            FrontierDiscovery loop + end_to_end wiring (§14.1)
  replay.py              offline stop replay from frozen ledgers
  extract.py, conformal.py   LLM reader + δ_E gate; structured index reader
  er.py, corroborate.py  ER guard + echo-collapse corroboration
  aggregate.py           COUNT with the honest ε_C interval (§7.3)
  oracle_opencitations.py, truth.py   the withheld oracle, freeze, grading
  storage.py             per-run SQLite ledger (the replay substrate)
  fetch.py               deny list + fetch-outcome classifier
  llm.py, search.py      single-provider LLM, Serper
notebooks/
  exp2c_stopping_guarantee.ipynb   the experiment end to end
scripts/
  probe_enumerators.py   $0: oracle size, reach, overlap u, feasible band
  pilot_walk.py          short live walk + automatic gate calibration
  build_citation_truth.py   freeze truth + targets from the oracle
  run_exp2c.py           LIVE phase (gated by RUN_LIVE=1)
  replay_exp2c.py        OFFLINE arms + self-tests S1/S2 + C1/C2 tables
  grade_exp2c.py         exact grading vs the frozen key
  pilot_diagnose.py      explain a run's discovery rate from its ledger
tests/                   112 tests (see TESTING.md) — the gate
```

## Setup (PyCharm / Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env     # OPENAI_API_KEY, SERPER_API_KEY, OPENALEX_API_KEY
pytest                     # the gate: green before any live spend
```

`.env` is auto-loaded by `webaggs.config` (shell values win). `RUN_LIVE`
and `RUN_ONLINE` are never read from it: arming spend or the network is a
deliberate per-session `$env:` in the shell.

## Running Exp 2-C (order is part of the pre-registration)

```powershell
pytest                                             # 1. gate
$env:RUN_ONLINE="1"; pytest -m online -v           # 2. preflight ($0)
python scripts/probe_enumerators.py                # 3. probe ($0): u, band
#    -> write eps, tau_max and the cohort band into the prereg (section 6.3)
$env:RUN_LIVE="1"
python scripts/pilot_walk.py --title "..." --doi ... --pmid ... --cutoff 2023-12-31
#    -> 4. pilot: plumbing check + conformal gate fitted automatically
#    5. choose cohort papers by the prereg rule; cohort/papers.txt lines:
#       Title | DOI | PMID
python scripts/build_citation_truth.py --cohort cite_v1 --papers cohort/papers.txt --cutoff 2023-12-31
#    6. supervisor signs the prereg; commit it
python scripts/run_exp2c.py    --cohort cite_v1 --eps <eps> --max-steps <tau_max>
python scripts/replay_exp2c.py --cohort cite_v1 --live-eps <eps> --arms 0.9,0.8,0.7,0.6,0.5
python scripts/grade_exp2c.py  --cohort cite_v1
```

Idempotent: `run_exp2c.py` skips finished papers (`runs_index.json`); no
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
