"""ClaimsEngine -- FULL ENGINE (guide §11, design paper §5.2).

This file replaces the chapter-7 stub. The paper's idea: the web states not
only PARTS ("Acme raised $40M in 2025") but claims about the WHOLE ("Acme
has raised three rounds totaling $63M"). A corroborated claim about an
aggregate is a CHECKSUM for the assembled table (paper Definition 11), and
Theorem 4 turns a matching checksum into a completeness certificate:

  (a) SUM  => value-completeness: missing dollars <= Delta+_g
             = max(0, V - assembled) + t_V + E_g, however many records hide;
  (b) COUNT => exact completeness: |D_g| == n_g means recall 1;
              |D_g| < n_g means the stratum MUST NOT stop (hard brake).

Four duties, all here:
  1. ingest Claims and corroborate them with EXACTLY the §8 machinery
     (a claim is a value whose "attribute" is an aggregate -- it gets
     derivation components, supersession along annual-report chains,
     a noisy-OR belief, a margin);
  2. evaluate the per-stratum checksum and certify strata whose
     checksums close (Theorem 4);
  3. emit GAP-DIRECTED formulations for residuals (Definition 12:
     "the residual is a query" -- the algebra says what is missing);
  4. expose the CARDINALITY BRAKE (App. E), now armed by the real
     corroborated COUNT instead of ch. 7's two-witness provisional rule.

Three discipline rules (guide §11), each a paper clause:
  R1. Certification is conditional AND SAYS SO. The belief b is stored on
      the status and on the stratum; (1 - b) propagates into the stratum's
      interval (§13/§14). A stratum certified on b_V = 0.92 is a NAMED 8%,
      not certainty.
  R2. Scope mismatches DEMOTE, never certify. "Raised $80M including debt"
      must not certify an equity-rounds stratum. Demoted claims may still
      direct gaps. Demotions are LOGGED -- the measured demotion rate is
      the claim-semantics limitation the paper owns in its §11.
  R3. Conflicts OUTRANK both sides. Found > claimed, or certified-by-claim
      while Chao insists records remain: stop certifying, push a
      verification item. Two certified mechanisms disagreeing is the
      highest-information human check there is.
"""
from __future__ import annotations

import json
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

from . import config
from .corroboration import QTable, corroborate
from .frontier import Formulation, FrontierState, normalize_surface
from .llm import call_llm

# The gap-direction prompt (guide §11 verbatim); loaded lazily so unit
# tests that stub the LLM never need the file present.
GAP_PROMPT_PATH = config.ROOT_DIR / "prompts" / "gap_formulations.txt"


@dataclass
class ChecksumStatus:
    """Result of evaluating one stratum's checksum (guide §11 shape)."""
    certified: bool = False
    kind: str | None = None       # "COUNT" | "SUM"
    belief: float = 0.0           # b_n or b_V -- STORED (rule R1); enters the interval
    delta_plus: float | None = None   # SUM value-gap bound Delta+_g (Thm 4a)
    gap: dict | None = None       # residual -> gap-direction payload (Def. 12)
    conflict: bool = False        # claim vs. statistics -> verify (rule R3)


@dataclass
class CoverageView:
    """What the checksum gets to see about one stratum.

    Grew chapter by chapter: ch. 7 needed only the record count; ch. 9
    added ER's fragile pairs; §11 adds the assembled SUM, the extraction
    tolerance E_g, the Chao remainder estimate, and the covered eras /
    stages (raw material for gap-directed formulations). Everything past
    n_records defaults to "unknown/empty" so older call sites still work.
    """
    n_records: int
    # ch. 9: in-stratum ER pairs whose match decision sat in the band,
    # as (mention_id_a, mention_id_b, theta). A COUNT that matches only
    # because one of these fell the right way must not certify.
    fragile_pairs: tuple = ()
    # --- §11 additions -----------------------------------------------------
    sum: float = 0.0        # assembled aggregate over the stratum's records
    E_g: float = 0.0        # extraction tolerance on that sum (Thm 4 conditions on it)
    chao_m0: float = 0.0    # Chao1 unseen-record estimate (pass 0 when not finite)
    years: tuple = ()       # eras COVERED so far (gap prompt targets the complement)
    stages: tuple = ()      # round stages covered so far (same idea)


@dataclass
class _ClaimShim:
    """Adapter that lets a Claim ride through corroborate() unchanged.

    corroborate() was written for Mentions but only ever reads two fields
    off each assertion: .source_id (to build the derivation graph over the
    asserting SOURCES) and .mention_id (for the provenance list). So a
    claim needs no more than this to be corroborated "exactly like an
    attribute" (guide §11) -- no parallel corroborator, no drift.
    """
    source_id: str
    mention_id: str


def implied_value_tolerance(x: float) -> float:
    """Precision a stated figure implies: half its last significant digit,
    capped at 1% of the value (press figures are rarely better than ~1%).
    "$12M" (12_000_000) -> min(0.5e6, 0.12e6) = 0.12e6. Summed per record,
    this is the assembled-sum extraction tolerance E_g that Theorem 4
    conditions on ("assembled values are within extraction tolerance").
    Deliberately conservative: a fat E_g can only DELAY a SUM cert."""
    x = abs(float(x))
    if x == 0:
        return 0.0
    xi, k = int(round(x)), 0
    while xi > 0 and xi % 10 == 0:      # count trailing zeros = stated granularity
        xi //= 10
        k += 1
    return min(0.5 * 10 ** k, 0.01 * x)


class ClaimsEngine:
    def __init__(self, session, tol_rel: float = config.CLAIM_TOL_REL):
        self.session = session
        self.tol_rel = tol_rel          # SUM certifies when Delta+ <= tol_rel * V
        # normalized stratum key -> list[Claim] (in-memory index this run)
        self._claims: dict[str, list] = defaultdict(list)
        # insert-once memory for gap formulations: {(stratum, gap signature)}
        self._gaps_emitted: set[tuple[str, str]] = set()
        # ch. 9 count-sensitivity: fragile ER pairs that vetoed a COUNT
        # certification. SHAPE FROZEN since ch. 9 (tests pin it):
        # (stratum, mention_id_a, mention_id_b, theta).
        self.verification_queue: list[tuple[str, str, str, float]] = []
        # rule R3: conflict verification items (found>claimed, Chao-vs-checksum)
        self.conflicts: list[dict] = []
        # source_id -> Source. When populated (pipeline registers each fetched
        # source), claim corroboration runs the FULL §8 machinery; when empty
        # (stub sessions in unit tests), a conservative degenerate noisy-OR
        # is used instead -- see corroborated().
        self.sources: dict[str, object] = {}
        # rule R2 bookkeeping: which claim_ids were considered / demoted
        self._seen_ids: set[str] = set()
        self._demoted_ids: set[str] = set()
        self.demotions: list[dict] = []      # one row per demotion, auditable

    # -- 0. plumbing --------------------------------------------------------

    def register_source(self, src) -> None:
        """Give the engine the Source behind future claims, enabling full
        §8 corroboration (derivation graph, chains, class priors)."""
        self.sources[src.source_id] = src

    def ingest(self, claim) -> None:
        """Persist and index one stratum-level claim."""
        self.session.merge(claim.to_row())   # merge: the same claim can recur
        self._claims[normalize_surface(claim.stratum_surface)].append(claim)

    def rekey(self, surface_to_entity: dict[str, str]) -> None:
        """Ch. 9: after frontier.rekey_strata moves strata to entity_id keys,
        the claim index must follow -- COUNT claims asserted about
        "Acme Corp" and "ACME" now back the SAME entity stratum's checksum.
        Surfaces ER never saw keep their key."""
        rekeyed: dict[str, list] = defaultdict(list)
        for surf, cs in self._claims.items():
            rekeyed[surface_to_entity.get(surf, surf)].extend(cs)
        self._claims = rekeyed

    # -- 1. scope discipline (rule R2) --------------------------------------

    def _claim_key(self, c) -> str:
        """Stable id for demotion bookkeeping (stub claims lack claim_id)."""
        return getattr(c, "claim_id", None) or \
            f"{c.source_id}:{getattr(c, 'functional', '?')}:{getattr(c, 'value_num', '?')}"

    def _scope_ok(self, c) -> bool:
        """SCOPE DISCIPLINE: claims whose scope words mismatch the query
        scope are DEMOTED -- usable for gap-direction, never to certify.

        Config-driven (guide §11): FORBID words ("debt", "to date") demote
        wherever they appear in the scope or supporting passage; REQUIRE
        words ("equity", "round") are checked only when the claim STATES a
        scope -- an unscoped "three rounds totaling $63M" has nothing to
        mismatch, but a stated scope must contain a required word."""
        scope = (getattr(c, "scope", "") or "").lower()
        text = f"{scope} {(getattr(c, 'passage', '') or '').lower()}"
        if any(w in text for w in config.CLAIM_SCOPE_FORBID):
            return False
        if scope and config.CLAIM_SCOPE_REQUIRE and \
                not any(w in text for w in config.CLAIM_SCOPE_REQUIRE):
            return False
        return True

    def _clean_claims(self, g: str, functional: str) -> list:
        """Claims for (stratum, functional) that survive scope discipline.
        Every demotion is LOGGED once per claim (rule R2: the measured
        demotion rate is a limitation the paper owns)."""
        out = []
        for c in self._claims.get(g, []):
            if getattr(c, "functional", None) != functional:
                continue
            key = self._claim_key(c)
            self._seen_ids.add(key)
            if self._scope_ok(c):
                out.append(c)
            elif key not in self._demoted_ids:
                self._demoted_ids.add(key)
                self.demotions.append({
                    "claim": key, "stratum": g, "functional": functional,
                    "scope": getattr(c, "scope", "")})
        return out

    @property
    def demotion_rate(self) -> float:
        """Demoted / considered, over all claims this engine has judged."""
        return len(self._demoted_ids) / max(len(self._seen_ids), 1)

    # -- 2. claim corroboration (the §8 machinery, guide §11) ---------------

    def _supersede_by_asof(self, claims: list) -> list:
        """Supersession by as_of ALONG THE SAME AUTHORITY CHAIN (paper
        Def. 11: "a newer annual report supersedes last year's total").
        Within one chain, any claim strictly older (by t_asof) than the
        chain's newest dated claim is dropped. Claims without a chain or
        without a date are kept -- staleness must be PROVEN, not guessed.
        Needs registered Sources for chain ids; degrades to a no-op
        without them (source-level supersession inside corroborate()
        still applies on the full path)."""
        by_chain: dict[str, list] = defaultdict(list)
        keep = []
        for c in claims:
            s = self.sources.get(c.source_id)
            chain = getattr(s, "authority_chain_id", None) if s else None
            (by_chain[chain] if chain else keep).append(c)
        for cs in by_chain.values():
            dated = [c for c in cs if getattr(c, "t_asof", None)]
            if not dated:
                keep.extend(cs)
                continue
            newest = max(c.t_asof for c in dated)
            keep.extend(c for c in cs
                        if not getattr(c, "t_asof", None) or c.t_asof >= newest)
        return keep

    def _group_values(self, claims: list, functional: str) -> dict[str, list]:
        """Group claims by CANONICAL value (the "competing values" of the
        corroborator). COUNT: exact integer. SUM: values that agree within
        the larger of their stated tolerances (or 0.5%, the same slack the
        A/B extraction agreement uses) are ONE candidate value -- "$63M"
        and "$63.0M" corroborate instead of competing."""
        if functional == "COUNT":
            groups: dict[str, list] = defaultdict(list)
            for c in claims:
                groups[str(int(round(c.value_num)))].append(c)
            return dict(groups)
        groups = {}
        for c in sorted(claims, key=lambda c: c.value_num):
            for rep, cs in groups.items():
                tol = max(getattr(c, "tolerance", 0.0),
                          max((getattr(x, "tolerance", 0.0) for x in cs), default=0.0),
                          0.005 * abs(c.value_num))
                if abs(c.value_num - float(rep)) <= tol:
                    cs.append(c)
                    break
            else:
                groups[f"{float(c.value_num):.6g}"] = [c]
        return groups

    def corroborated(self, g: str, functional: str):
        """Adopt ONE claimed value for (stratum, functional), corroborated
        exactly like an attribute (guide §11): scope discipline, as_of
        supersession along chains, then components + noisy-OR + margin via
        corroborate(). Returns (value_num, belief, tolerance) or None.

        Full path (Sources registered): claims become _ClaimShims and go
        through §8's corroborate() -- derivation components, registry-chain
        supersession, class priors, the qbar cap, all for free.
        Degenerate path (no Sources -- stub sessions in unit tests): every
        distinct source_id is ONE unanchored origin at the adversarial cap
        qbar, the most conservative reading. Two such witnesses give
        1-(1-0.3)^2 = 0.51 -- deliberately mirroring ch. 7's ">= 2
        witnesses" provisional rule at the default brake threshold."""
        claims = self._clean_claims(g, functional)
        if not claims:
            return None
        claims = self._supersede_by_asof(claims)
        groups = self._group_values(claims, functional)
        qtable = QTable()
        if all(c.source_id in self.sources for cs in groups.values() for c in cs):
            by_value = {v: [_ClaimShim(c.source_id, self._claim_key(c)) for c in cs]
                        for v, cs in groups.items()}
            cv = corroborate(by_value, self.sources, qtable)
            adopted, value, belief = groups[cv.value], float(cv.value), cv.belief
        else:
            beliefs = {v: 1.0 - (1.0 - qtable.qbar) ** len({c.source_id for c in cs})
                       for v, cs in groups.items()}
            v_star = max(beliefs, key=beliefs.get)   # noisy-OR argmax, like §8
            adopted, value, belief = groups[v_star], float(v_star), beliefs[v_star]
        # the adopted value's tolerance t_V: the loosest stated precision
        # among its supporting claims ("$63M" means +/- 0.5e6, not exact)
        tol = max((getattr(c, "tolerance", 0.0) for c in adopted), default=0.0)
        return value, belief, tol

    # -- 3. cardinality brake (App. E) --------------------------------------

    def provisional_claimed_count(self, g: str) -> int | None:
        """Ch. 7's provisional rule (>= 2 distinct clean-scope sources on
        the same integer). SUPERSEDED by corroborated() for the brake, kept
        as a cheap diagnostic and for back-compat (ch. 9 tests pin it)."""
        by_value: dict[int, set[str]] = defaultdict(set)
        for c in self._claims.get(g, []):
            if getattr(c, "functional", None) != "COUNT" or not self._scope_ok(c):
                continue
            by_value[int(round(c.value_num))].add(c.source_id)
        for n, srcs in sorted(by_value.items()):
            if len(srcs) >= 2:
                return n
        return None

    def update_cardinality_brakes(self, state: FrontierState) -> None:
        """Refresh StratumState.claimed_count for every uncertified stratum
        (called once per loop step). §11 change: the brake now runs on the
        REAL corroborated COUNT. It arms once belief clears
        config.CLAIM_BRAKE_MIN_BELIEF (default 0.5 = two independent capped
        witnesses, or one anchored registry alone); App. E prices wrong
        claims by exactly this belief and the margin."""
        for g, S in state.strata.items():
            if S.certified is None:
                cc = self.corroborated(g, "COUNT")
                S.claimed_count = (int(round(cc[0]))
                                   if cc and cc[1] >= config.CLAIM_BRAKE_MIN_BELIEF
                                   else None)

    # -- 4. the checksum (Theorem 4) ----------------------------------------

    def _push_conflict(self, g: str, kind: str, **details) -> None:
        """Rule R3: a conflict is a VERIFICATION ITEM, not a judgment call.
        Deduplicated on (stratum, kind) so re-evaluations don't spam."""
        if not any(c["stratum"] == g and c["kind"] == kind for c in self.conflicts):
            self.conflicts.append({"stratum": g, "kind": kind, **details})

    def checksum(self, g: str, view: CoverageView) -> ChecksumStatus:
        """Evaluate stratum g's checksum against the assembled view.

        COUNT branch (Thm 4b): deficit -> hard-brake gap; equality -> certify
        (unless a fragile ER pair could flip the count -- ch. 9 veto);
        surplus -> CONFLICT (we found MORE than the web claims: verify).
        SUM branch (Thm 4a), only if COUNT didn't already close and nothing
        conflicts: Delta+ = max(0, V - assembled) + t_V + E_g. Small enough
        (<= tol_rel * V) and no count gap -> certify with delta_plus stored;
        a residual beyond tolerance -> gap-direction payload (Def. 12).
        Finally, Chao vs. checksum: claim says done but capture-recapture
        screams a remainder -> withdraw the cert, flag (rule R3)."""
        st = ChecksumStatus()
        cc = self.corroborated(g, "COUNT")
        sc = self.corroborated(g, "SUM")
        if cc:
            n, b_n, _ = cc
            n = int(round(n))
            if view.n_records < n:
                # shortfall: a hard brake AND a steer -- the loop must not
                # stop (App. E) and gets told exactly how many are missing
                st.gap = {"count_gap": n - view.n_records}
            elif view.n_records == n:
                if view.fragile_pairs:
                    # ch. 9 count-sensitivity check: the match is one coin
                    # toss deep -> verify, never certify
                    st.conflict = True
                    for (a, b, theta) in view.fragile_pairs:
                        item = (g, a, b, float(theta))
                        if item not in self.verification_queue:
                            self.verification_queue.append(item)
                    return st
                # Thm 4(b): |D_g| == n_g and no borderline decision can
                # flip it -> recall 1. Belief STORED (rule R1).
                st.certified, st.kind, st.belief = True, "COUNT", b_n
            else:
                # found MORE than claimed -> the claim or our table is wrong;
                # verify, trust neither (rule R3)
                st.conflict = True
                self._push_conflict(g, "found_more_than_claimed",
                                    found=view.n_records, claimed=n)
        if sc and not st.certified and not st.conflict:
            # (the "not st.conflict" guard is rule R3 made explicit: a
            # count conflict must not be papered over by a SUM cert)
            V, b_V, tol = sc
            dp = max(0.0, V - view.sum) + tol + view.E_g     # Delta+_g (Thm 4a)
            if dp <= self.tol_rel * V and not st.gap:
                st.certified, st.kind, st.belief, st.delta_plus = \
                    True, "SUM", b_V, dp
            elif V - view.sum > tol + view.E_g:
                # residual beyond tolerance: Definition 12 -- everything the
                # algebra knows about what's missing becomes the gap payload
                gap = st.gap or {}
                gap.update({"value_gap": V - view.sum,
                            "eras_covered": list(view.years),
                            "stages": list(view.stages)})
                st.gap = gap
        if st.certified and view.chao_m0 > 2:
            # claim says done, Chao screams remainder -> flag, don't trust
            st.certified, st.conflict = False, True
            self._push_conflict(g, "chao_vs_checksum",
                                chao_m0=float(view.chao_m0))
        return st

    # -- 5. gap-directed search (Definition 12) -----------------------------

    def gap_formulations(self, g: str, gap: dict) -> list[Formulation]:
        """"The residual is a query": turn the gap payload into 1-4 targeted
        formulations via the LLM (prompts/gap_formulations.txt). Emitted at
        most ONCE per (stratum, gap signature) -- the engine re-detects the
        same gap every step it persists, and without the dedup the frontier
        balloons (guide §7.3). Gap-directed formulations enter the frontier
        with unusually high yield estimates: they search for something a
        corroborated claim says EXISTS."""
        sig = ",".join(sorted(gap))
        if (g, sig) in self._gaps_emitted:
            return []
        self._gaps_emitted.add((g, sig))
        payload = {"entity": g,
                   **{k: (list(v) if isinstance(v, (set, tuple)) else v)
                      for k, v in gap.items()}}
        try:
            out = call_llm(system=GAP_PROMPT_PATH.read_text(),
                           user=json.dumps(payload),
                           purpose="gap_formulations")["formulations"]
            return [Formulation(str(uuid.uuid4())[:8], f["query"],
                                float(f.get("p_success", 0.5)),
                                float(f.get("yield_if_success", 1.0)),
                                stratum=g, gap_directed=True)
                    for f in out[:4]]
        except Exception:
            # LLM/prompt unavailable (offline tests, transient API failure):
            # fall back to ch. 7's template rather than fail the loop. The
            # gap still gets ONE search pointed at it.
            return [Formulation(str(uuid.uuid4())[:8],
                                f'"{g}" additional funding round announcement',
                                0.5, float(gap.get("count_gap", 1)),
                                stratum=g, gap_directed=True)]
