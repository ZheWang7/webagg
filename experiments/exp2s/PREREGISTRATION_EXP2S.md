# PREREGISTRATION — Experiment 2-S: Certified Stopping in an Accessible Domain (NBA)

**Status: DRAFT — not in force until (a) the supervisor signs off on the cohort
selection rule and (b) this file is committed BEFORE the first live run.**
This is a NEW pre-registration (new domain, new oracle, new grading key), not
an addendum to any formd document. No analytical decision below may be changed
after the first live run; changes require a dated addendum committed before
any run they affect.

Date drafted: 2026-10-20
Supervisor sign-off: ______________________  (name / date)

## 1. Purpose and claim scope

Exp 2-S is a control experiment for the finance Exp 2 failure. The finance
run produced certificates claiming >= 90% mass at ~41% realized recall; the
working hypothesis (supervisor meeting, 2026-10-20) is access censoring —
paywalled sources made the oracle's truth exceed the accessible web and let
denied fetches masquerade as informative samples. The NBA domain removes the
access ceiling: every official stat is freely mirrored.

Claims tested: **C1** (Û + ψ stays above the true unseen mass at every step;
Assumption 1(a)) and **C2** (at a certified stop, realized recall ≥ 1 − ε at
rate ≥ 1 − δ_M). Theorem 1 / §3.2–3.3 of webaggr.pdf.

Explicitly OUT of scope: corroboration (Theorem 2) — every source in this
domain mirrors one origin (the official scorer), so independent confirmation
does not exist here; nothing in this experiment validates the corroboration
layer. Fidelity (ε_F) is NOT calibrated for this domain; all reports carry
`eps_f = None` and no two-term interval is claimed.

Pre-registered interpretations:
- certifies here + failed in finance → the completeness guarantee holds over
  the *accessible* web; the finance oracle exceeds it (scope limitation).
- fails here too → the calibration itself is broken. Reported as a finding.

## 2. Query and aggregation

Query template: *"What is [player]'s total career regular-season points in
the NBA?"* — f_Q = SUM over (player, season) strata records of integer
`points`. One stratum per run (the player's career); records are seasons.

## 3. Oracle and deny list

- Oracle: official NBA statistics (stats.nba.com `playercareerstats`,
  PerMode=Totals, regular season). Truth is complete by construction
  (two-sided: every played season is scored; no unfiled-record analogue).
- Deny list at run time: `nba.com`, `stats.nba.com` and subdomains, enforced
  inside the fetcher; every denial is logged. **Basketball-Reference and all
  other sites count as "the web"** — a deliberate design choice (denying
  aggregators too would make this an artificial-scarcity study).

## 4. Cohort selection rule  *(supervisor signs this section)*

Target ~25–30 players, drawn BEFORE any live run from the frozen
`cohort_report.csv` (official career-games numbers):

- **Anchor stars (~5):** players with ≥ 500 career games (e.g. Jalen
  Brunson). Expected to certify fast; sanity check, not the test.
- **Fringe cohort (~20–25):** players with **≤ 150 career regular-season
  games**, preferring careers with gap seasons, 10-day/two-way stints, or a
  pre-2010 start — where no single page is complete and stopping is actually
  exercised.

Names are listed in `cohort/candidates.txt` and frozen with the truth table
(`data/truth/<cohort>/manifest.json`, SHA-256). Append-only after any entity
has been run; additions go under a new cohort name.

## 5. Scope rules (grading key, locked now)

1. **Regular season only.** Playoff, preseason, G-League, international and
   college points are out of scope; the query is "NBA career regular-season
   points".
2. **Traded seasons:** one record per (player, season), league-wide total
   (the oracle's `TOT` row when present).
3. **Grading: EXACT.** Record key = (player_norm, canonical season
   "YYYY-YY"); value = integer points. `record_recall` = seasons found /
   true seasons (the discovery quantity the stopping rule certifies);
   `exact_recall` additionally requires the points value to match. No
   tolerance machinery.
4. Season canonicalization: "2018–2019", "2018/19", "2018-19" → "2018-19";
   multi-year spans are rejected by the typed validator, not guessed.

## 6. Run design and parameters

- LIVE: one run per cohort player at **ε = 0.05, δ_M = 0.05**, budget $5
  search per player + LLM; `RUN_LIVE=1` gate; idempotent per-player runs
  index (no re-rolling — a finished player is never rerun).
- OFFLINE: arms ε ∈ {0.30, 0.20, 0.10} replayed from the recorded ledgers
  (valid: ε is read only in the stop test). Self-tests, which must pass
  before any table is reported: S1 replay at 0.05 reproduces the live stop;
  S2 stop steps are monotone in ε.
- Stopping machinery exactly as the implementation guide: Û = f1/N + β·Σŷ/N
  (β = 1), ψ = √(2·V_max·log(1/(δ_M·w_g))) with c = Y_CAP(1+β)/n, Y_CAP = 12,
  η = 0.20, max 200 occasions; cardinality hard brake from corroborated
  COUNT claims; economic stop is never reported as a certificate.

## 7. Reporting split  *(prevents a repeat of the exp3 p3 vacuous pass)*

Statistical-path and checksum-path certifications are reported SEPARATELY.
The headline C1/C2 numbers rest on **statistical stops only**; checksum
closures (corroborated "played N seasons" / "career total X" claims) are a
secondary Theorem-4-style result. A checksum certificate granted pre-ER is
revalidated post-ER and revocations are logged.

## 8. Pre-registered simplifications (documented deviations from the finance build)

- **ER is deterministic** (name-variant folding onto the queried player +
  a contamination guard), not a fitted matcher: the season is a join key
  here, so Theorem 7's probabilistic machinery is unnecessary. Any
  cross-player contamination that survives shows up as spurious records in
  grading and is reported, not hidden.
- **Corroboration is echo-collapse only** (one vote per domain, class-prior
  weighted). No provenance graph; no corroboration claims made (§1).
- **ε_F uncalibrated** in this domain; no fidelity certificate is claimed.

## 9. Pre-registered bad outcomes (reported as findings, no re-rolls)

1. C1 points above the diagonal (claimed unseen < realized unseen) —
   falsifies Assumption 1(a) in an access-uncensored domain.
2. Statistical-path violation rate > δ_M at the live ε.
3. Zero statistical stops on the fringe cohort (everything ends economic /
   frontier_exhausted / checksum) — stopping untestable even here.
4. Anchor stars certify only via checksum — the statistical path never fires
   even in the easy regime.

## 10. Execution order

pytest suite green → supervisor signs §4 → candidates screened →
`build_cohort.py` freezes truth → conformal gate calibrated (labeled
extraction set, ≥ 20 correct examples; calibration labels come from a
separate pilot player NOT in the cohort) → commit this file →
`run_exp2s.py` (live) → `replay_exp2s.py` (offline) → grade → write up.
