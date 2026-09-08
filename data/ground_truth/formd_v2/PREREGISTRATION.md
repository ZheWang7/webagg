# A3 Pre-registration — formd_v2 fidelity certification (eps_F)

Locked 2026-09-07, before any reference run. The cohort, truth tables, and
split were frozen at commit `b846882` ("formd_v2 truth build: 52 entities
frozen"). Nothing in this document may change after the first reference run;
any later instrument fix requires its own dated addendum committed before the
runs it affects.

## 1. Certification target

- eps_F = **0.25**, delta_F = **0.05** (`config.DELTA_F`).
- Single target, no fallback ladder: if the fixed-sequence test cannot
  certify 0.25, A3 reports that failure as a finding. Zero-loss floor at
  n_cal = 34 is 0.210, leaving ~0.04 of headroom for realized losses.

## 2. Cohort and split

- Cohort: `formd_v2`, 52 entities (see `manifest.json` — the sole authority).
- Split: 34 calibration / 18 validation, cal_frac = 0.6667, seed = 0,
  assigned by `assign_split` and **append-only**: assignments never move;
  departures are recorded in `split_dropped`, never reshuffled.
- The validation half is untouched until certification on the calibration
  half is complete; realized loss on it is the honest check of eps_F.

## 3. Grading key (decided from the frozen answer key only)

- **Primary alignment: amount-only** — greedy nearest-amount matching within
  `GRADING_AMOUNT_TOL` = 0.05; ties broken by nearest date. An explicit
  `registry_key` (recovered accession number) still takes priority when
  present. Records matching nothing remain SPURIOUS at full value
  (conservative direction), as before.
- Evidence, computed on the frozen tables before any run: 5 of 672
  within-entity round pairs collide at the 5% tolerance (10 of 267 rounds,
  3.7%, 5 of 52 entities); 4 of the 5 colliding pairs are >= 2 years apart,
  so the date tiebreak resolves all observed collisions.
- Exact `(kind|date)` keying is rejected because pipeline dates are press
  announcement dates while truth keys carry filing/firstSale dates; the
  verdict pass documented announcement-to-filing lags of days to months as
  the norm, which would misalign matches systematically.
- Collisions remain visible via `ambiguous_truth_pairs` / the build-time
  `close_amounts` flag; a grade over an ambiguous entity is never silently
  trusted.

## 4. Loss, tolerance, and universe

- Loss: `fidelity_loss` as implemented at the frozen commit.
- Amount tolerance: 5% (`GRADING_AMOUNT_TOL`).
- Universe: primary equity only — debt facilities, pure secondaries, and
  IPO/public offerings are out of universe by design.
- Alignment uses base (unqualified) kinds; instance qualifiers are stripped.

## 5. Procedure (Learn-Then-Test)

- `learn_then_test` fixed-sequence testing over the pre-committed
  cheapest-first config list in `scripts/certify_fidelity.py` at the frozen
  commit; changing that list before runs requires an addendum.
- Stop at the first failing config; no re-ordering or re-running the grid
  after seeing losses; p-values via `hoeffding_p`.

## 6. Truth-table integrity rules (as applied at freeze)

- Late filers adjudicated by `totalOfferingAmount` + `dateOfFirstSale`,
  never date proximity alone.
- Still-selling chains captured as of freeze (Stoke D-ext, Apptronik,
  Castelion Series C — flagged in `cohort_screen.csv` notes).
- Cityblock Health excluded by the pre-registered truth-build-day rule
  (Series E Form D absent from EDGAR on 2026-09-07).

## 7. No re-rolling

- Verdicts in `cohort_screen.csv` are closed; the split is append-only
  history; results are reported regardless of outcome — a failed
  certification is a publishable finding, not a reason to re-run.

---

# Addendum 1 — Attempt #2 (locked 2026-09-08, before any attempt-#2 replay)

## A. Outcome of attempt #1 (recorded)

- NOT CERTIFIED: the first grid config failed E[L] <= 0.25 at delta_F = 0.05
  (mean L = 0.6834, p = 1.0, n = 34 calibration entities).
- Diagnosis (calibration half ONLY; the validation half was not consulted):
  loss was ~entirely the spurious channel (1.709 vs 0.002 amount-error);
  record dumps showed duplicate copies of true rounds under distinct
  resolved-entity ids, i.e. ER FALSE SPLITS. Mention-level audit: the
  fitted matcher carried ~zero weight on name features (no variance in the
  A2 training pairs, which were same-page aggregator-heavy) and gated
  merges on same_domain + temporal; cross-domain same-name pairs scored
  theta ~= 0.09 <= tau_minus and were split without adjudication.

## B. Instrument changes for attempt #2 (all committed before replays)

1. Adjudicator robustness: malformed LLM payloads retry once, then fall
   back loudly to theta = 0.5 (band); confidences clipped to [0, 1].
   Crash fix; behavior on well-formed payloads unchanged (pinned by test).
2. Matcher refit on a deployment-representative pair set: match_pairs.csv
   grown 320 -> 460 rows (148 same / 312 different) via harvest_er_pairs.py
   over the frozen calibration pools -- cross-domain same-surface
   positives, cross-entity look-alike hard negatives, band pairs -- all
   human-labeled. Refit coefficients are name/embedding-dominated;
   same_domain is mildly negative. Cheap-matcher out-of-fold error
   alpha = 0.3739. Consequence, measured offline: tau_plus is unreachable
   by cheap features, so ALL 5,946 blocked pairs across the cohort
   escalate to the LLM adjudicator at every grid config. Attempt #2
   therefore certifies the configuration "blocking -> LLM adjudication ->
   correlation clustering", with the cheap tier acting as a router only.
   Thresholds tau+/tau- and the grid are NOT changed.
3. Per-certification adjudication cache (memoized_adjudicator): each
   unordered mention pair is judged once per certification and the verdict
   reused across grid configs -- verdict consistency plus ~4x cost
   reduction (5,946 calls instead of 23,784). In-memory, never reused
   across certification runs.

## C. Unchanged, and disclosures

- Cohort, truth tables, and the 34/18 append-only split remain frozen at
  commit b846882. eps_F = 0.25, delta_F = 0.05, the grading key (Sec. 3),
  the loss (Sec. 4), and the grid order (Sec. 5) are unchanged.
- The frozen reference pools are REUSED: every attempt-#2 change is in the
  replay stage (ER onward). Disclosure: the pooling policy that gathered
  them (reference discovery config) internally used the attempt-#1 matcher
  for its stopping statistics; the certificate is, as always in this
  design, conditional on that fixed pooling policy.
- The validation half remains untouched by every diagnostic and fix above;
  it is spent only in the post-certification holdout audit.
- Procedure unchanged: fixed-sequence LTT, stop at first failure, results
  reported regardless of outcome. Any further instrument change after this
  addendum requires Addendum 2 before the replays it affects.
