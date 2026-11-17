# Exp 2-C probe — Highly accurate protein structure prediction with AlphaFold
DOI 10.1038/s41586-021-03819-2 · PMID 34265844 · cutoff 2023-12-31 · page size 50

## 1. Oracle: OpenCitations
- full list fetched and SELF-VALIDATED against /citation-count in 3s
- raw rows 40830: no DOI 1291, undated 1161, after cutoff 23741, duplicates 0
- **truth size at cutoff: 14637 DOI-registered citers**

## 2. Enumerators
- openalex: handle W3177828909, index total 14853, first page 50 rows (asked 50), 50 with DOI, next cursor yes
- europepmc: handle 34265844, index total 12325, first page 50 rows (asked 50), 50 with DOI, next cursor yes
- NOTE: OPENALEX_API_KEY not set — OpenAlex anonymous access is metered per IP; set the free key before a full walk

## 3. Reach (recall ceiling against the oracle)
- sample of 120 truth records: in OpenAlex 82%, in Europe PMC 79%, in BOTH 70%, in NEITHER 8%
- DOI aliasing: for 19 of the 120 sampled truth DOIs OpenAlex lists the citing work under a DIFFERENT primary DOI (counted as not reachable: the keys would not join)
- **recall ceiling ~ 92%** (truth records reachable through at least one enumerator)

## 4. Overlap between the enumerators (singleton share u)
- of OpenAlex's citers, 84% are also in Europe PMC; of Europe PMC's, 83% are also in OpenAlex
- estimated distinct records N ~ 15,804; **singleton share u ~ 0.28**
- of OpenAlex's sampled citers, 96% are in the oracle's truth (the rest are spurious for grading, not harmful to recall)

## 5. Can the certificate fire? (needs u + psi < eps)
| eps | min tau_max | N must be between | this paper's N | verdict |
|---|---|---|---|---|
| 0.5 | 2340 | 53,905 – 67,455 | 15,804 | too few citers |
| 0.6 | 1130 | 25,741 – 32,274 | 15,804 | too few citers |
| 0.7 | 670 | 15,098 – 18,899 | 15,804 | fits |
| 0.8 | 450 | 9,992 – 12,503 | 15,804 | too many for this tau |
| 0.9 | 330 | 7,176 – 9,013 | 15,804 | too many for this tau |

(min tau_max is the smallest budget with a 25%-wide band; a larger tau_max widens the band and moves it up.)
