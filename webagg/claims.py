"""ClaimsEngine -- MINIMAL CHAPTER-7 STUB.

The full engine (guide §11: claim corroboration as an attribute with
components/supersession/noisy-OR, COUNT/SUM checksums, scope discipline,
gap-direction) lands in a later chapter. Chapter 7 needs exactly three
things from it, so that is all this stub provides:

  1. ingest(claim)      -- persist every stratum-level Claim the extractor
                           emits (they are the checksum's raw material).
  2. the CARDINALITY BRAKE (paper App. E) -- the guide says to wire this
     "extra" in EARLY because it dominates statistics for small strata:
     a provisionally-corroborated COUNT claim sets
     StratumState.claimed_count, and all_strata_pass() refuses to stop
     while N_g < claimed_count. "Provisional" = >= 2 distinct sources
     agreeing on the same integer with clean scope words; §11 replaces
     this with real noisy-OR corroboration.
  3. checksum(g, view)  -- always returns certified=False for now (no
     stratum can close via checksum before §11), but does report a
     count_gap so one templated gap-directed formulation can be proposed,
     with insert-once dedup per (stratum, gap signature) (guide §7.3).
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import uuid

from .frontier import Formulation, FrontierState, normalize_surface


@dataclass
class ChecksumStatus:
    """Result of evaluating a stratum's checksum (guide §11 shape)."""
    certified: bool = False
    kind: str | None = None       # "COUNT" | "SUM"
    belief: float = 0.0           # b_n / b_V (enters the interval, §13)
    delta_plus: float | None = None   # SUM value-gap bound
    gap: dict | None = None       # residual -> gap-direction
    conflict: bool = False        # claim vs. statistics -> verify (§12)


@dataclass
class CoverageView:
    """What the checksum gets to see about a stratum. Chapter 7 only needs
    the record count; ch. 9 adds ER's fragile pairs (in-stratum pairs whose
    match decision sat in the band); §11 adds sum, E_g.

    fragile_pairs: [(mention_id_a, mention_id_b, theta)] from
    ERResult.fragile_by_stratum()[g]. Empty pre-ER (the loop cannot know
    them yet), populated in the post-ER pass -- the count-sensitivity
    check below only bites when they are supplied."""
    n_records: int
    fragile_pairs: tuple = ()


# Scope words that DEMOTE a claim (guide §11 scope discipline, minimal
# version): demoted claims may direct gaps but never arm the brake.
SCOPE_FORBID = ("debt", "to date", "including debt", "total raised")


class ClaimsEngine:
    def __init__(self, session, tol_rel: float = 0.02):
        self.session = session
        self.tol_rel = tol_rel
        # normalized stratum surface -> list[Claim] (in-memory this run)
        self._claims: dict[str, list] = defaultdict(list)
        # insert-once memory for gap formulations: {(stratum, signature)}
        self._gaps_emitted: set[tuple[str, str]] = set()
        # ch. 9 count-sensitivity: fragile ER pairs that vetoed a COUNT
        # certification, queued for verification (an LLM re-adjudication or
        # a human look). Each item: (stratum, mention_id_a, mention_id_b, theta).
        self.verification_queue: list[tuple[str, str, str, float]] = []

    # -- 1. ingestion -------------------------------------------------------

    def ingest(self, claim) -> None:
        """Persist and index one stratum-level claim."""
        self.session.merge(claim.to_row())     # merge: same claim can recur
        self._claims[normalize_surface(claim.stratum_surface)].append(claim)

    def rekey(self, surface_to_entity: dict[str, str]) -> None:
        """Ch. 9: after frontier.rekey_strata moves strata to entity_id keys,
        the claim index must follow -- COUNT claims asserted about
        "Acme Corp" and "ACME" now back the SAME entity stratum's checksum
        (and two one-source claims can become the >= 2 witnesses the
        provisional brake needs). Surfaces ER never saw keep their key."""
        rekeyed: dict[str, list] = defaultdict(list)
        for surf, cs in self._claims.items():
            rekeyed[surface_to_entity.get(surf, surf)].extend(cs)
        self._claims = rekeyed

    # -- 2. cardinality brake (App. E), provisional corroboration -----------

    def provisional_claimed_count(self, g: str) -> int | None:
        """>= 2 distinct sources asserting the SAME integer COUNT with clean
        scope -> that integer. Else None. (Placeholder for §11 noisy-OR.)"""
        by_value: dict[int, set[str]] = defaultdict(set)
        for c in self._claims.get(g, []):
            if c.functional != "COUNT":
                continue
            if any(w in (c.scope or "").lower() for w in SCOPE_FORBID):
                continue                        # demoted: never arms the brake
            by_value[int(round(c.value_num))].add(c.source_id)
        for n, srcs in sorted(by_value.items()):
            if len(srcs) >= 2:                  # two independent-ish witnesses
                return n
        return None

    def update_cardinality_brakes(self, state: FrontierState) -> None:
        """Refresh StratumState.claimed_count for every uncertified stratum.
        Called once per loop step (pipeline.py step 3)."""
        for g, S in state.strata.items():
            if S.certified is None:
                S.claimed_count = self.provisional_claimed_count(g)

    # -- 3. checksum stub + gap direction -----------------------------------

    def checksum(self, g: str, view: CoverageView) -> ChecksumStatus:
        """Chapter-7 stub + the ch.-9 COUNT-SENSITIVITY CHECK.

        Certification itself stays a §11 feature, but the guard the guide
        mandates ("Before a COUNT checksum certifies, ask ER for the
        stratum's fragile pairs ... do not certify -- queue the pair for
        verification") is wired NOW, so when §11 flips the certify branch
        on, the "no false merge/split" condition is already honest: a
        checksum that matches only because one borderline ER decision fell
        the right way must not close the stratum."""
        st = ChecksumStatus()
        n_claimed = self.provisional_claimed_count(g)
        if n_claimed is None:
            return st
        if view.n_records < n_claimed:
            # shortfall: report the gap so the loop can steer searches at it
            st.gap = {"count_gap": n_claimed - view.n_records}
            return st
        if view.n_records == n_claimed:
            # --- ch. 9 count-sensitivity check (runs BEFORE any certify) ---
            # |D_g| == n_g, but if one fragile pair flipped, a split would
            # give n_g + 1 records (or a merge n_g - 1) and the checksum
            # would no longer match: the agreement is one coin toss deep.
            if view.fragile_pairs:
                st.conflict = True               # -> verify (§12), never certify
                for (a, b, theta) in view.fragile_pairs:
                    item = (g, a, b, float(theta))
                    if item not in self.verification_queue:
                        self.verification_queue.append(item)
                return st
            # §11 will set st.certified/kind/belief RIGHT HERE (count matches
            # and no fragile pair can flip it); the stub leaves it False.
        return st

    def gap_formulations(self, g: str, gap: dict) -> list[Formulation]:
        """One templated gap-directed formulation per (stratum, gap kind),
        emitted at most once (guide §7.3: the engine re-detects the same
        gap every step it persists -- dedup or the frontier balloons)."""
        sig = ",".join(sorted(gap))
        if (g, sig) in self._gaps_emitted:
            return []
        self._gaps_emitted.add((g, sig))
        missing = gap.get("count_gap", 1)
        return [Formulation(
            formulation_id=str(uuid.uuid4())[:8],
            query=f'"{g}" additional funding round announcement',
            p_success=0.5,                     # gap-directed: a claim says it exists
            yield_if_success=float(missing),
            stratum=g,
            gap_directed=True,
        )]
