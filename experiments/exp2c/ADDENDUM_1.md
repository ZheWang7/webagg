# Exp 2-C — Addendum 1 (2026-11-17, 23:20 EST)

**Status: DRAFT for supervisor review. It changes nothing in the signed
pre-registration (rev 4): same cohort, same ε = 0.6, same τ_max = 1,150,
same rules. It records one finding and puts predictions on record BEFORE
the remaining four runs.** Written after run 1 of 5; no truth table or
grade was opened to write it (only the run-1 ledger and
cohort/selection_report.csv).

Supervisor review: ______________________  (name / date)

## 1. Run 1 outcome

*Clinical course and risk factors for mortality of adult inpatients with
COVID-19* — both index walks finished (OpenAlex 500 pages ok, Europe PMC
336 ok + 1 empty end page; no aborted walk). N = 25,618, singletons
f₁ = 9,665, Û = 0.377, ψ = 0.324, Û + ψ = 0.701 ≥ 0.6 → **abandoned**
(pre-registered outcome). Records by source: both indexes 15,949 (62.3%),
OpenAlex only 8,996 (35.1%), Europe PMC only 673 (2.6%).

## 2. Finding: the u column of prereg §4 is biased low

The selection rule estimated the overlap from the FIRST 100 citers of
each index. For this paper it estimated that 96% of OpenAlex's citers
are in Europe PMC; the walk shows 64%. The head of OpenAlex's list is
much better covered by Europe PMC than its tail. The estimate was also
internally impossible: Europe PMC's total (16,760) is 67% of OpenAlex's
(24,955), so no more than 67% could overlap — the script averaged the two
directions instead of checking that bound. The reverse direction was
accurate (Europe PMC's citers in OpenAlex: 0.96 sampled, 0.96 realized).

Consequence: the "certificate fires if u stays below" margins in §4 were
computed against under-estimated u. The cohort is NOT changed (it was
selected by the signed rule before any run).

## 3. Predictions for the remaining four runs (made before running them)

Using only the index totals and the reliable direction b (share of
Europe PMC's citers also in OpenAlex): I = b·n_EP, N = n_OA + n_EP − I,
u = (N − I)/N, ψ = Y_CAP·(1+β)·sqrt(2·τ_max·ln(1/δ_M))/N.

| Paper | n_OA | n_EP | b | N | u | ψ | Û + ψ | at ε = 0.6 |
|---|---|---|---|---|---|---|---|---|
| KEGG | 23,753 | 19,609 | 0.96 | 24,537 | 0.233 | 0.338 | 0.571 | borderline pass |
| edgeR | 28,421 | 25,183 | 0.83 | 32,702 | 0.361 | 0.254 | 0.615 | narrow miss |
| PLINK | 24,629 | 21,514 | 0.86 | 27,641 | 0.331 | 0.300 | 0.631 | miss |
| Sequence-variant guidelines | 19,877 | 16,831 | 0.92 | 21,223 | 0.270 | 0.391 | 0.662 | miss |

b rests on samples of 100 and one confirmed data point; these are rough.

## 4. What is done with the result

The remaining four papers are run unchanged. The pre-registered offline
arms (ε = 0.9, 0.8, 0.7, 0.6) are replayed from the five ledgers — valid
because ε is read only in the stop test and the frontier must be
exhausted before any arm can stop — and C2 is graded at every arm where a
certificate fires. Live-arm outcomes are reported as they fall,
including "abandoned".
