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
