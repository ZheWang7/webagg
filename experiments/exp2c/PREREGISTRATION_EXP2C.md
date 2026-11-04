# PREREGISTRATION — Experiment 2-C: The Statistical Stopping Guarantee (citations)

**Status: DRAFT — in force only after (1) the enumerator probe and the
second PILOT fix the second index and the live ε (§6), (2) the supervisor
signs §4, §6 and §10, and (3) this file is committed BEFORE the first
cohort run.** New domain, new oracle, new grading key: a fresh
pre-registration, not an addendum.

Date drafted: 2026-11-03 · Revised 2026-11-04 after PILOT 1 (§6.1)
Supervisor sign-off: ______________________  (name / date)

## 1. Purpose and claim scope

Exp 2-S (NHL, complete — see the sports branch) proved the pipeline
end-to-end and audited C1, but its strata were too small for any
statistical stop: zero fired, as pre-registered. Exp 2-C tests the claim
Exp 2-S could not: **C2 — at a statistically certified stop, realized
recall ≥ 1 − ε at rate ≥ 1 − δ_M** (Theorem 1 / §3.2–3.3), plus C1 at a ψ
that is no longer vacuous. Checksums are OUT (§5): aggregate claims are
disabled in this domain, so every certificate is statistical or nothing.

**Scope statement (added after PILOT 1, see §10.4).** In this domain the
dense evidence lives in paginated citation indexes, so Û is driven by
deterministic index coverage rather than by random recapture. Exp 2-C
therefore certifies *reachability of the statistical stop* and
*recall at the stop against a withheld oracle*; it does NOT stress-test
Good–Turing optimism (Assumption 1(a)) the way a prose-web domain would,
and the C1 calibration plot is reported with that caveat.

## 2. Query and aggregation

One run per target paper: *"Which papers cite [target]?"* — records are
citing papers, f_Q = COUNT over distinct records (each record carries
value 1 through the unchanged SUM path). Record key: DOI (canonical,
lowercased) when present, else normalized title (≥ 3 words, or ≥ 6
non-space characters for unspaced scripts); the SAME keying function on
the truth and extraction sides.

## 3. Oracle and deny list

OpenAlex (api.openalex.org) defines truth and is DENIED at run time
(deny list: openalex.org). Everything else — Semantic Scholar, Europe
PMC, OpenCitations, Google Scholar, arXiv, publisher pages, blogs — is
"the web" (the hockey-reference role; finance precedent: aggregator
sites were legitimate sources, EDGAR was the oracle). Truth = the citers
list at a pre-registered CUTOFF date (citing papers published ≤ cutoff;
the citation analogue of "retired players only" — citation lists grow
daily, truth must not). The driver self-validates every walk against the
API's own count and refuses partial truth.

## 4. Cohort rule  *(supervisor signs)*

Papers selected BY RULE, not by hand: OpenAlex works with cited_by_count
inside the band implied by the ladder (§6.3) at selection time,
publication year ≤ 2021 (settled citation flow), type=article, and —
because double coverage needs two independent enumerators (§6.2) — inside
the field both chosen indexes cover (biomedicine if the second index is
Europe PMC); ~5 papers. Optionally ONE deliberately adversarial giant,
pre-registered as a stress probe of Assumption 1(a): on it the
certificate may fire at low true recall — an informative falsification
test, never counted in the headline C2 rate. The PILOT papers are never
in the cohort.

## 5. Pre-registered domain rules

- **Aggregate claims DISABLED** (config.CLAIMS_ENABLED=False): cited-by
  counts differ across indexing universes (Scholar > S2 > OpenAlex), so a
  corroborated-but-wrong COUNT would poison the cardinality brake. An
  index's reported total is used ONLY as the residual-yield estimate ŷ of
  its unissued pages (frontier credit), never as a claim. All
  certificates are therefore statistical; "abandoned" is the only other
  ending.
- Dedup: doi-URL/case variants of one DOI are ONE record; title-keyed
  records join on the normalized form; mistargeted citations (ER guard:
  the page's cited-paper must match the target title) are rejected.
- Grading: record recall at the (replayed) stop against frozen truth;
  value fidelity is trivial (all values 1) — this experiment is
  completeness-only by design. Shortfall is decomposed as in Exp 2-S:
  *unreachable* (present in no fetched source family — the theorem's
  frontier-coverage term δ_F) vs *reachable-but-missed*.

## 6. Instrument calibration: PILOT 1 (done), enumerator probe, PILOT 2

### 6.1 PILOT 1 result (2026-11-03, "Attention Is All You Need", $0.76, 38 occasions)

Measured max per-occasion novelty **6**; 13 records total (0.33 per
occasion); 194 pages fetched ok → 13 accepted / 0 rejected mentions
(the reader is not the bottleneck). Fetch outcomes: semanticscholar.org
`empty` ×11 (JavaScript app, no citations in the HTML),
scholar.google.com "ok" ×10 but zero records (bot-wall shell), IEEE
mostly `empty`, ScienceDirect/ResearchGate `paywall`. All 13 records came
from blog posts. **Finding:** on the prose web the discovery rate is
~0.07 records per fetched page; the certificate needs ~4 new records per
occasion sustained over τ_max — unreachable at any ε. This is the finance
Exp 2 failure mode reproduced, and it is recorded as a result in its own
right (the fetch-outcomes instrument identified censoring of every dense
HTML source).

### 6.2 Design decision following PILOT 1 (supersedes the original §6)

The public citation data that is dense AND fetchable lives in paginated
JSON APIs, not HTML. The pipeline therefore gains a domain plug-in — a
**source-class recognizer with cursor-style pagination** — and nothing
else changes (frontier, ψ, stop test, claims, replay, ledger untouched):

1. When a fetch hits a recognized citation index, CODE (not the LLM)
   emits the formulation "page k+1 of index I for paper P" — exactly one
   new page formulation per page fetched (cursor-style), with ŷ =
   min(page size, index-reported remaining) and p_success = 1.
2. **Each page fetch is its own capture occasion** (Def. 5).
3. **Y_CAP = the API page size (100)**, a bound known by construction, and
   every occasion's intake — search occasions included — is **truncated
   to Y_CAP** (the implementation guide's sanctioned tightening); excess
   records are not credited and remain unseen mass. Violations are
   impossible by construction, and the per-occasion novelty series is
   still logged for audit.
4. **Two independent indexes** enumerate the same DOI-keyed population so
   records are double-covered and f₁ can fall. Index 1: Semantic Scholar
   (api.semanticscholar.org, 100/page). Index 2 is fixed by a $0
   fetchability probe BEFORE PILOT 2: Europe PMC (paginated, biomedicine
   only → cohort field restricted per §4) is the default; OpenCitations
   returns an unpaginated whole list and is excluded on Y_CAP grounds.
5. Extraction still passes the fetched text (the JSON rendered as text)
   through the certified LLM reader and the conformal gate.
6. **τ_max = 400** (was 200). The page count the certificate needs,
   2·(1+β)·sqrt(2·τ_max·ln(1/δ_M))/ε, is independent of page size and
   must fit inside τ_max; at ε = 0.5 that is ~392 occasions.

### 6.3 Ladder under the revised design (β = 1, δ_M = 0.05, τ_max = 400, Y_CAP = 100)

n_needed(ε) = Y_CAP·(1+β)·sqrt(2·τ_max·ln(1/δ_M))/ε ≈ 9,790/ε:
ε = 0.5 → 19,600 · 0.4 → 24,500 · 0.3 → 32,600 · 0.2 → 49,000.
Candidate live ε: **0.5**, cohort band ≈ 25k–80k citers, pending PILOT 2.
If PILOT 2 shows the two-index walk cannot deliver n_needed inside
τ_max and budget, THAT is the finding and the cohort run does not proceed.

### 6.4 PILOT 2 (after the plug-in is built and the suite is green)

One live run on a non-cohort paper in the chosen field: confirms the
walk completes inside τ_max, logs the per-occasion novelty series (must
never exceed Y_CAP — it cannot), measures the realized discovery curve,
and dumps the conformal rows for the citation extractor (hand-labeled
against the paper's public citation listing, ≥ 20 correct; the gate is
calibrated on the FINAL reader behavior, so PILOT 1's 13 rows are
discarded).

## 7. Run design

LIVE: one run per cohort paper at the pilot-chosen ε (δ_M = 0.05, budget
$15–25/run, τ_max = 400). OFFLINE: the arm ladder replayed from the
ledgers (valid: ε is read only in the stop test); self-tests S1/S2 gate
every table. Idempotent runs index; crashed ledgers set aside, never
appended.

## 8. Pre-registered outcomes

- **Success**: certified stops fire on cohort papers; violation rate of
  recall ≥ 1−ε at most ~δ_M; C1 points on/under the diagonal.
- **Bad outcome 1**: certificates fire with recall < 1−ε beyond δ_M — the
  web-visible population (both indexes) falls short of the oracle by more
  than ε, i.e. the frontier-coverage term δ_F dominates; report as the
  central finding with the shortfall decomposition, no re-rolls.
- **Bad outcome 2**: no certificate fires within budget/τ_max — the
  guarantee is unpriceable at sane budgets in this domain; report with
  the spend/decay curves (PILOT 1 is already one such data point for the
  prose web).
- **Stress arm (if included)**: expected possible false-fire; reported
  separately, never in the headline.

## 9. Execution order

pytest green → PILOT 1 ✔ → enumerator probe (index 2) → plug-in built,
suite green, CI-live-validated → PILOT 2 (→ ε, gate) → this file
finalized → cohort selected by §4 rule → truth frozen (cutoff fixed) →
supervisor signs → live runs → replay/grade/notebook → write-up.

## 10. Consistency of the §6.2 design with Theorem 1  *(supervisor signs)*

The revised design is an instantiation of the paper's abstractions, not a
modification of Assumption 1 or Theorem 1. Checked clause by clause:

**10.1 Code-emitted page formulations (Def. 5).** "A formulation q is a
concrete search; issuing it at step t is a capture occasion returning
sources covering some records." A paginated index request is a concrete
search returning sources covering records; the definition does not
restrict q to a web-search query (the guide's `search(q, k=10)` loop is
one instantiation). The paper already admits a non-LLM proposer ("new
formulations are proposed by an LLM … and by the constraint engine of
§5"); the recognizer is a proposer of the same kind. Theorem 1 requires
only that the choice of q_{t+1} be history-measurable ("survives any
history-dependent choice of q_{t+1}"); "emit page k+1 after reading page
k" is a deterministic function of the history.

**10.2 Y_CAP as page size + intake truncation (Assumption 1(b)).** The
assumption is |M_t − M_{t−1}| ≤ c_t with c_t H_{t−1}-measurable; the
guide instantiates c_t = Y_CAP·(1+β)/N_{t−1}, computed from N BEFORE the
occasion (`pre_occasion_c`). For index pages the novelty bound holds by
construction. For ordinary search occasions it is ENFORCED by truncating
intake to Y_CAP — the guide's own words: "legitimate tightenings: …
truncate each occasion's intake to a true Y_CAP." Truncated records are
not credited and remain unseen mass, the conservative direction.

**10.3 Cursor-style emission (the two-sided bound as written).** Enqueuing
all pages of an index at once would add β·(remaining)/N of frontier
credit in one step — a large DOWNWARD jump of M_t = U_t − Û_t. The proof
survives it (a downward jump only makes the event U_t > Û_t + ψ less
likely; the paper's own LLM follow-ups cause the same kind of jump at
smaller scale), but it exceeds the two-sided |ΔM_t| ≤ c_t as literally
stated. Emitting one page formulation per fetched page keeps the
per-step credit change ≤ β·Y_CAP/N — inside the (1+β) factor of c_t —
so the implementation satisfies Assumption 1(b) as written, not merely as
the proof needs. Residual pages remain in the frontier's promising part
F_active (ŷ = page size ≥ η₀), so conjunct (ii) keeps the agent from
stopping mid-walk, as intended.

**10.4 Two indexes and Assumption 1(a) — what changes.** Assumption 1(a)
(Û stays an over-estimate in conditional expectation) is, in the paper's
words, "the modeling commitment of the open-web regime … empirically
testable: the calibration plot is its direct audit." Walking two complete
indexes collapses f₁ by construction: every DOI present in both becomes a
doubleton whether or not oracle-only citers exist. Records visible in
neither index are not a δ_M failure — they are the theorem's
frontier-coverage caveat ("every true record is reachable by some
eventually-proposed formulation, failing with probability δ_F"). Hence:
(i) the stop remains governed by ψ (n must clear §6.3), so the
statistical certificate is genuinely exercised; (ii) recall against
withheld OpenAlex measures the δ_F-type shortfall, reported through the
§5 decomposition; (iii) the C1 calibration plot is a weaker audit of
Assumption 1(a) here than on the prose web — stated as the §1 scope
caveat, not hidden.

**10.5 Unchanged.** Û (eq. 1), ψ = sqrt(2·V_max·log(1/(δ_M·w_g))), the
two-conjunct stop test, the ε-only-in-the-stop-test replay property, the
ledger/replay substrate, τ_max as a pre-registered budget cap
(Assumption 1(c): raising it to 400 before the freeze changes V_max, not
the argument).
