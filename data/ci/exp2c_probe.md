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
- sample of 120 truth records: in OpenAlex 98%, in Europe PMC 79%, in BOTH 70%, in NEITHER -7%
- **recall ceiling ~ 108%** (truth records reachable through at least one enumerator)

## 4. Overlap between the enumerators (singleton share u)
- of OpenAlex's citers, 84% are also in Europe PMC; of Europe PMC's, 100% are also in OpenAlex
- estimated distinct records N ~ 14,777; **singleton share u ~ 0.16**
- of OpenAlex's sampled citers, 96% are in the oracle's truth (the rest are spurious for grading, not harmful to recall)

## 5. Can the certificate fire? (needs u + psi < eps)
| eps | min tau_max | N must be between | this paper's N | verdict |
|---|---|---|---|---|
| 0.5 | 1150 | 24,473 – 30,720 | 14,777 | too few citers |
| 0.6 | 700 | 14,746 – 18,487 | 14,777 | fits |
| 0.7 | 480 | 9,946 – 12,506 | 14,777 | too many for this tau |
| 0.8 | 350 | 7,164 – 8,971 | 14,777 | too many for this tau |
| 0.9 | 280 | 5,541 – 7,068 | 14,777 | too many for this tau |

(min tau_max is the smallest budget with a 25%-wide band; a larger tau_max widens the band and moves it up.)
