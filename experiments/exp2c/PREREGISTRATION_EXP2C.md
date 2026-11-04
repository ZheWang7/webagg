# PREREGISTRATION — Experiment 2-C: The Statistical Stopping Guarantee (citations)

**Status: DRAFT — in force only after (1) the PILOT fixes Y_CAP and the
live ε (§6), (2) the supervisor signs §4 and §6, and (3) this file is
committed BEFORE the first cohort run.** New domain, new oracle, new
grading key: a fresh pre-registration, not an addendum.

Date drafted: 2026-11-03
Supervisor sign-off: ______________________  (name / date)

## 1. Purpose and claim scope

Exp 2-S (NHL, complete — see the sports branch) proved the pipeline
end-to-end and audited C1, but its strata were too small for any
statistical stop: zero fired, as pre-registered. Exp 2-C tests the claim
Exp 2-S could not: **C2 — at a statistically certified stop, realized
recall ≥ 1 − ε at rate ≥ 1 − δ_M** (Theorem 1 / §3.2–3.3), plus C1 at a ψ
that is no longer vacuous. Checksums are OUT (§5): aggregate claims are
disabled in this domain, so every certificate is statistical or nothing.

## 2. Query and aggregation

One run per target paper: *"Which papers cite [target]?"* — records are
citing papers, f_Q = COUNT over distinct records (each record carries
value 1 through the unchanged SUM path). Record key: DOI (canonical,
lowercased) when present, else normalized title (≥ 3 words); the SAME
keying function on the truth and extraction sides.

## 3. Oracle and deny list

OpenAlex (api.openalex.org) defines truth and is DENIED at run time
(deny list: openalex.org). Semantic Scholar, Google Scholar, arXiv,
publisher pages and the rest of the web are "the web" (the
hockey-reference role). Truth = the citers list at a pre-registered
CUTOFF date (citing papers published ≤ cutoff; the citation analogue of
"retired players only" — citation lists grow daily, truth must not).
The driver self-validates every walk against the API's own count and
refuses partial truth.

## 4. Cohort rule  *(supervisor signs)*

Papers selected BY RULE, not by hand: OpenAlex works with cited_by_count
inside the band implied by the pilot ladder (§6) at selection time,
publication year ≤ 2021 (settled citation flow), type=article, across ≥ 2
fields; ~5 papers. Optionally ONE deliberately adversarial giant
(≥ 40k citations), pre-registered as a stress probe of Assumption 1(a):
on it the certificate may fire at low true recall — an informative
falsification test, never counted in the headline C2 rate. The PILOT
paper is never in the cohort.

## 5. Pre-registered domain rules

- **Aggregate claims DISABLED** (config.CLAIMS_ENABLED=False): cited-by
  counts differ across indexing universes (Scholar > S2 > OpenAlex), so a
  corroborated-but-wrong COUNT would poison the cardinality brake. All
  certificates are therefore statistical; "abandoned" is the only other
  ending.
- Dedup: doi-URL/case variants of one DOI are ONE record; title-keyed
  records join on the normalized form; mistargeted citations (ER guard:
  the page's cited-paper must match the target title) are rejected.
- Grading: record recall at the (replayed) stop against frozen truth;
  value fidelity is trivial (all values 1) — this experiment is
  completeness-only by design.

## 6. The instrument calibration PILOT (before anything else)

One live run (scripts/pilot_ycap.py, ~$3) on a non-cohort paper measures:
1. **Y_CAP** — the real max per-occasion novelty (Assumption 1(b)). The
   prereg then fixes Y_CAP at the measured max and computes the ladder
   n_needed(ε) = Y_CAP·(1+β)·sqrt(2·τ_max·ln(1/δ_M))/ε. The live ε is the
   smallest value whose n_needed is comfortably below the cohort papers'
   citer counts AND below what the budget can plausibly discover. If no ε
   ≤ 0.5 satisfies both, THAT is the finding (the increment-bound price of
   dense index pages), reported, and the cohort run does not proceed.
2. **The conformal gate** for the citation extractor (hand-labeled rows
   from the pilot, ≥ 20 correct) — the NHL gate does not transfer.

## 7. Run design

LIVE: one run per cohort paper at the pilot-chosen ε (δ_M = 0.05, budget
$15–25/run, τ_max = 200). OFFLINE: the arm ladder replayed from the
ledgers (valid: ε is read only in the stop test); self-tests S1/S2 gate
every table. Idempotent runs index; crashed ledgers set aside, never
appended.

## 8. Pre-registered outcomes

- **Success**: certified stops fire on cohort papers; violation rate of
  recall ≥ 1−ε at most ~δ_M; C1 points on/under the diagonal.
- **Bad outcome 1**: certificates fire with recall < 1−ε beyond δ_M —
  Assumption 1(a) fails under real heterogeneous findability; report as
  the central finding, no re-rolls.
- **Bad outcome 2**: no certificate fires within budget (discovery can't
  reach n_needed) — the guarantee is unpriceable at sane budgets in this
  domain; report with the spend/decay curves.
- **Stress arm (if included)**: expected possible false-fire; reported
  separately, never in the headline.

## 9. Execution order

pytest green → PILOT (→ Y_CAP, ε, gate) → this file finalized → cohort
selected by §4 rule → truth frozen (cutoff fixed) → supervisor signs →
live runs → replay/grade/notebook → write-up.
