# PREREGISTRATION — Experiment 2-C: The Statistical Stopping Guarantee (citations)

**Status: DRAFT (rev 3) — in force only after (1) the enumerator probe
fixes ε, τ_max and the cohort band (§6.3), (2) PILOT 2 passes (§6.4),
(3) the supervisor signs §4, §6 and §10, and (4) this file is committed
BEFORE the first cohort run.** New domain, new oracle, new grading key: a fresh
pre-registration, not an addendum.

Date drafted: 2026-11-03 · rev 2 2026-11-04 after PILOT 1 (§6.1) ·
rev 3 2026-11-17: oracle changed to OpenCitations, enumerators fixed as
OpenAlex + Europe PMC, plug-in implemented (§3, §6.2; change log §11)
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

One run per target paper: *"Which DOI-registered works cite [target],
created on or before the cutoff?"* — records are citing works, f_Q =
COUNT over distinct records (each record carries value 1 through the
unchanged SUM path). **Record universe: DOI-registered citing works** —
the oracle's universe. Record key: the canonical lowercased DOI, produced
by the SAME function on the truth and extraction sides; a row without a
DOI is rejected by the typed validator (logged), never keyed on its title.

## 3. Oracle and deny list

**OpenCitations** (api.opencitations.net, the open CC0 citation index)
defines truth and is DENIED at run time (deny list: opencitations.net).
It was chosen over Google Scholar (no official API, no reproducible citer
list) and replaces OpenAlex in the oracle role so that OpenAlex can serve
as an enumerator. OpenCitations returns a paper's whole citing list in one
response: disqualifying for a capture occasion (Y_CAP would equal n),
ideal for an oracle. Everything else — OpenAlex, Europe PMC, publisher
pages, blogs — is "the web".

Truth = works in the OpenCitations citing list of the target that (a)
carry a DOI and (b) have a creation year ≤ the cutoff year. The cutoff is
a pre-registered **31 December**: the oracle's creation dates have year,
month or day precision, and only a year-end cutoff is exact for all
three. Rows without a DOI or without a date are excluded and counted in
the cohort report. The driver self-validates the list length against the
oracle's own /citation-count and refuses to write truth on a mismatch.
The agent's enumerators apply the same cutoff through their own date
filters; the indexes date a work slightly differently at year
boundaries, which is part of the measured shortfall, not corrected.

## 4. Cohort rule  *(supervisor signs)*

Papers selected BY RULE, not by hand: biomedical research articles
(both enumerators must cover them: the paper has a PMID), publication
year ≤ 2021 (settled citation flow), whose oracle truth size at the
cutoff lies inside the feasible band of §6.3; ~5 papers. Optionally ONE deliberately adversarial giant,
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
- Dedup: doi-URL/case variants of one DOI are ONE record; rows without
  a DOI are outside the universe (§2); mistargeted citations (ER guard:
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

### 6.2 Design following PILOT 1 (implemented: webaggs/indexes.py)

The public citation data that is dense AND fetchable lives in paginated
JSON APIs, not HTML. The pipeline gains a domain plug-in — **paginated
citation indexes walked cursor-style** — and nothing else changes
(frontier, ψ, stop test, claims, replay, ledger untouched):

1. Each index's first page is a deterministic seed formulation proposed
   by CODE; after a page is read, code emits exactly one new formulation,
   "page k+1 of index I" (cursor-style), with ŷ = min(page size,
   index-reported remaining) and p_success = 1. A failed page is
   re-enqueued at most twice, then the abort is logged.
2. **Each page fetch is its own capture occasion** (Def. 5).
3. **Y_CAP = the page size we request (50)**, a bound known by
   construction (the plug-in also truncates an over-delivering page).
   Every occasion's intake — search occasions included — is **truncated
   to Y_CAP** (the implementation guide's sanctioned tightening); excess
   records are not credited and remain unseen mass. Truncations are
   logged.
4. **Two independent enumerators** list the same DOI-keyed population so
   records are double-covered and f₁ can fall: **OpenAlex** (cursor
   paging, free key) and **Europe PMC** (CITES: search with cursorMark,
   no key; biomedicine only → cohort restricted per §4). Semantic Scholar
   is implemented as an OPTIONAL third enumerator and is used only if
   named here before the freeze. The oracle cannot be built as an
   enumerator (code-enforced, tested).
5. **Reader for index pages: structured** (changed from rev 2). The API
   rows are read directly, through the same typed validators and gate,
   with confidence 1.0. Reason: an LLM transcribing 50 DOIs per page can
   mistype one, and a mistyped DOI is a false singleton that corrupts f₁
   (eq. 1) — noise unrelated to the claim under test. The certified LLM
   reader + conformal gate still read every prose page, and an LLM-read
   arm for index pages remains available (`--index-reader llm`).
6. LLM follow-up proposals are triggered by SEARCH occasions only; an
   index page's only successor is its next page.

### 6.3 The feasibility arithmetic (webaggs/design.py; fixed by the probe)

The stop needs Û + ψ < ε with ψ = Y_CAP·(1+β)·sqrt(2·τ_max·ln(1/δ_M))/N.
After both walks, Û ≈ u, the share of found records that only ONE
enumerator lists (singletons). The frontier stays hot until every page
is issued, so both walks must also FIT inside τ_max: about (2−u)·N/Y_CAP
page occasions plus ~20 search occasions. Together these give a band of
N in which the certificate can fire, and a minimum τ_max (β = 1,
δ_M = 0.05, Y_CAP = 50, band at least 25% wide):

| ε | u = 0.0 | u = 0.1 | u = 0.2 | u = 0.3 |
|---|---|---|---|---|
| 0.5 | τ 640 · N 12.4k–15.5k | τ 890 · N 18.3k–22.9k | τ 1,390 · N 30.4k–38.1k | τ 2,750 · N 64.2k–80.3k |
| 0.6 | τ 460 · N 8.8k–11.0k | τ 590 · N 11.9k–15.0k | τ 800 · N 17.3k–21.7k | τ 1,250 · N 28.8k–36.2k |
| 0.7 | τ 350 · N 6.5k–8.3k | τ 420 · N 8.4k–10.5k | τ 530 · N 11.3k–14.2k | τ 720 · N 16.4k–20.6k |

If u ≥ ε the certificate cannot fire at any budget. **u is not assumed:
scripts/probe_enumerators.py measures it** (with the recall ceiling
against the oracle) on a non-cohort paper at $0. The live ε is the
smallest value whose row is affordable at the measured u; τ_max and the
cohort band are then read off that cell and written here before the
freeze. If no ε ≤ 0.7 is feasible at the measured u, THAT is the finding
and the cohort run does not proceed with two indexes (the pre-registered
fallback is to name Semantic Scholar as a third enumerator and re-probe).

### 6.4 PILOT 2 (scripts/pilot_walk.py, after the probe)

One short live walk (default 40 occasions) on a non-cohort paper:
confirms pages fetch `ok` and cursors advance, that no occasion credits
more than the page size, and fits the conformal gate WITHOUT hand
labeling — the first index pages are also read by the LLM reader, and
each LLM row is labeled correct iff its key is among the rows the API
returned for that page (≥ 20 correct rows required; PILOT 1's 13 rows are
discarded).

## 7. Run design

LIVE: one run per cohort paper at the probe-chosen ε and τ_max
(δ_M = 0.05, budget $25/run; both are REQUIRED flags of
scripts/run_exp2c.py, never defaults). OFFLINE: the arm ladder replayed from the
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

pytest green ✔ → PILOT 1 ✔ → plug-in built, suite green ✔ → online
preflight → enumerator probe (→ u, ε, τ_max, band) → PILOT 2 (→ gate) →
this file finalized → cohort selected by §4 rule → truth frozen (cutoff
fixed) → supervisor signs → live runs → replay/grade/notebook →
write-up.

## 10. Consistency of the §6.2 design with Theorem 1  *(supervisor signs)*

The revised design is an instantiation of the paper's abstractions, not a
modification of Assumption 1 or Theorem 1. Checked clause by clause:

**10.1 Code-emitted page formulations (Def. 5).** (Rev 3: an index's
first page is a deterministic seed rather than a recognizer firing on a
fetched page; the argument is unchanged.) "A formulation q is a
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
statistical certificate is genuinely exercised, and records only one
index lists stay singletons, so Û does not collapse to zero unless the
indexes agree; (ii) recall against
withheld OpenCitations measures the δ_F-type shortfall, reported through the
§5 decomposition; (iii) the C1 calibration plot is a weaker audit of
Assumption 1(a) here than on the prose web — stated as the §1 scope
caveat, not hidden.

**10.5 Unchanged.** Û (eq. 1), ψ = sqrt(2·V_max·log(1/(δ_M·w_g))), the
two-conjunct stop test, the ε-only-in-the-stop-test replay property, the
ledger/replay substrate, τ_max as a pre-registered budget cap
(Assumption 1(c): fixing it from the probe before the freeze changes
V_max, not the argument).

## 11. Change log

- **rev 2 (2026-11-04)** — PILOT 1 recorded; index-pagination design;
  §10 consistency argument.
- **rev 3 (2026-11-17)** — (a) oracle OpenAlex → OpenCitations (supervisor
  asked for an open alternative; Google Scholar has no API); OpenAlex
  becomes an enumerator, so the Semantic Scholar key is no longer needed;
  (b) record universe restricted to DOI-registered works, cutoff must be
  a 31 December; (c) Y_CAP = page size 50; (d) index pages read by the
  structured reader instead of the LLM reader (§6.2 item 5); (e) §6.3
  replaced by the feasibility band, which adds the singleton share u that
  rev 2's ladder ignored — rev 2's "τ_max = 400, n ≥ 19,600" assumed
  u = 0 and no search overhead and is withdrawn; (f) index entry pages
  are deterministic seeds. Nothing has been run under rev 2 or rev 3.
