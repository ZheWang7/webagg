# Exp 2-S Phase 1 — RESULTS (NHL domain, cohort nhl_v1)

Live run 2026-11-03 (29 players, ε = 0.05, δ_M = 0.05). Offline replay
S1/S2 self-tests passed; all tables in `data/experiments/exp2s/results/`.

## Headline

| Metric | Result |
|---|---|
| Record recall (seasons found) | **1.000 on 29/29 players** — zero truth seasons missed anywhere |
| Exact recall (season + points right) | 1.000 on 28/29; Hossa 18/19 (0.947) |
| Aggregation error | 0 on 26/29; Iginla 0.15%, Hossa 4.9%, Aliu +3 pts on a 3-pt truth |
| Spurious records | 2 total across the cohort (Iginla +1, Aliu +1) |
| Certificates | 28 checksum-certified, 1 abandoned (honest) — 0 statistical at every ε arm |
| C1 calibration | **0/92 points above the diagonal** — raw Û alone never understated the unseen mass |
| Shortfall | lost-to-stopping 0, never-discovered 0 |
| Spend | mean $0.06 search/player, max $0.48; ≈ $2 search for the whole cohort |

## The three pre-registered success criteria

1. **Reassembly accuracy on the stars** — Thornton 24 seasons/1,539 pts,
   Marleau 23/1,197, H. Sedin 17/1,070 all EXACT to the digit; Iginla
   within 0.15%; Hossa 95.1%. The full machine (discovery, extraction,
   gate, ER, corroboration, summation) reproduces official career totals
   from the open web with the oracle denied.
2. **C1 (Assumption 1(a) audit)** — claimed unseen mass (Û + ψ) sat on or
   above realized unseen mass at every logged step, 0/92 violations; even
   raw Û never went optimistic on this domain (contrast: the finance pilot
   saw raw Û drift optimistic — consistent with access censoring there,
   none here).
3. **Checksum closure** — 28/29 strata closed by corroborated COUNT/SUM
   claims, including every 1–2-game career (Glennie, McNeill, Jessiman:
   found, closed, graded exact at 0 points). The one abandoned run (Aliu,
   7-game career) ended frontier_exhausted rather than falsely certified.

## Pre-registered null confirmed

Zero statistical stops at every replay arm (ε ∈ {0.05, 0.10, 0.20, 0.30}),
exactly as the ψ arithmetic in the pre-registration predicted (n ≤ 24
records per stratum vs n ≥ 2,770 required). Bad outcome 3 is the expected
null, not a finding against the calibration. The statistical-stopping
guarantee is Phase 2's question (clinical trials, large-n queries).

## The three deviations — each a known hazard class, now with a specimen

- **Hossa 2007-08 (−56 pts): traded-season split adoption.** His trade
  year was ATL 56 + PIT 10 = 66; the deficit is exactly the ATL split,
  i.e. corroboration adopted a per-team split row over the season total —
  the ε_F-class hazard documented in the prereg before the run (web pages
  show traded seasons as splits; splits are wrong season-total
  assertions). Record recall unaffected; value channel only.
  Confirm from the ledger:
  `python -c "from webaggs.storage import Ledger; l=Ledger('data/experiments/exp2s/runs/exp2s_nhl_v1_marian_hossa.sqlite'); [print(r) for r in l.resolved('exp2s_nhl_v1_marian_hossa') if '2007-08' in r[0]]"`
- **Iginla +1 spurious (+2 pts): playoff-row contamination.** Truth has 20
  regular seasons; the agent asserted 21. Iginla's 1996 playoff debut
  (2 pts, no regular-season games in 1995-96) appears on career pages and
  one such row evidently passed the reader. Note the SUM checksum still
  certified (1,302 vs claimed 1,300 = 0.15% < 2% tol) — the textbook case
  for spurious-count as a co-primary metric.
- **Aliu +1 spurious (+3 pts on a 3-pt truth): minor-league bleed-through
  on a fringe career.** Both true seasons found and valued exactly; one
  extra season asserted (likely an AHL row misread as NHL). The 100%
  relative error is a tiny-denominator artifact — the co-primary metrics
  (recall 1.0, exact 1.0, 1 spurious) describe the run truthfully.

## Verdict and gate decision

Phase 1's question — does the pipeline do the end-to-end job correctly on
an accessible domain — is answered yes, with the failure modes confined to
two value-channel specimens of pre-documented hazard classes and zero
completeness failures. The go/no-go gate to Phase 2 (clinical trials,
statistical-stopping guarantee at large n) is **GO**.

Write-up framing: completeness and stopping were flawless but gently
tested here (short careers, well-aggregated web); the domain's value is
the calibration audit (0/92), the checksum mechanism demonstration at
scale, and the clean specimens of split-adoption and playoff-contamination
for the fidelity discussion.
