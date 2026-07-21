"""Frontier data structures and per-stratum stopping statistics.

SIGMOD guide ch. 7 / design paper §3. The frontier is the set of *intended*
searches. Discovery is tracked per capture OCCASION (one issued search =
one occasion), and each stratum g stops only when the anytime-valid bound
on unseen mass, U_hat_g + psi_g, drops below eps_g AND no pending search
still promises material yield (the two-conjunct rule, paper §3.3).

THE UNIT THAT MATTERS (guide §7.3, "the silent unit bug"): `covered`
stores FORMULATION ids, never source ids. A record found by four pages of
the same search is still a singleton -- one occasion saw it. If sources
were counted instead, f1 would collapse and the agent would stop absurdly
early with a clean-looking certificate.
"""
from __future__ import annotations
from dataclasses import dataclass, field, InitVar
from collections import defaultdict
import math
import re


# ---------------------------------------------------------------------------
# Pre-ER stratum keying (guide §7.3: "strata before ER are surface forms").
# One vocabulary only: this normalizer defines both the record key and the
# stratum key until entity resolution re-keys everything by entity_id (§14).
# ---------------------------------------------------------------------------

_CORP_SUFFIX = {"inc", "incorporated", "corp", "corporation", "llc", "ltd",
                "limited", "co", "company", "gmbh", "plc", "sa", "ag"}


def normalize_surface(s: str) -> str:
    """Lowercase, strip punctuation and corporate suffixes.

    'Acme, Inc.' and 'ACME INC' -> 'acme', so trivially different surface
    forms of one entity share a stratum even before ER runs. (Real
    disambiguation is ER's job, ch. 9; this only removes noise.)
    """
    toks = [t for t in re.findall(r"\w+", s.lower()) if t not in _CORP_SUFFIX]
    return " ".join(toks) or s.lower().strip()


@dataclass
class Formulation:
    """One intended search. The LLM prices it with a two-point yield model
    (paper App. B): with probability p_success it surfaces yield_if_success
    new records in expectation; otherwise nothing."""
    formulation_id: str
    query: str
    p_success: float = 1.0          # LLM estimate: P(>=1 new record)
    yield_if_success: float = 1.0   # LLM estimate: E[new records | success]
    cost: float = 0.02              # $ per issuance (config.SEARCH_COST_USD)
    stratum: str | None = None      # None = generic (may serve any stratum)
    gap_directed: bool = False      # came from the claims engine (§11)
    issued: bool = False
    realized_yield: int = 0         # filled in after issuance
    # Backward-compat constructor arg (pre-SIGMOD callers pass a single
    # expected_yield). Mapped to p=1, yield=expected_yield; not stored.
    expected_yield: InitVar[float | None] = None

    def __post_init__(self, expected_yield):
        if expected_yield is not None:
            self.p_success, self.yield_if_success = 1.0, float(expected_yield)
        # remember the LLM's ORIGINAL p so global calibration shrink
        # (update_yield_estimates) is idempotent, never compounding
        self._p0 = self.p_success

    @property
    def residual_yield(self) -> float:
        """\\hat y(q) in records: expected new records if issued now.
        Zero once issued -- an issued search promises nothing further."""
        return 0.0 if self.issued else self.p_success * self.yield_if_success

    def reservation_index(self, lam: float) -> float:
        """Economic ordering, paper App. B (an ORDERING HEURISTIC, not a
        certificate -- guide §7). Under the two-point yield model, issuing
        is worth it iff lam * yield_if_success > cost / p_success; the
        index sigma is the margin. lam = value of one new record ($);
        lam = 0 makes every index negative -> exploration halts."""
        if self.issued or self.p_success <= 0:
            return -math.inf
        return lam * self.yield_if_success - self.cost / self.p_success


@dataclass
class StratumState:
    """Per-stratum stopping state (persisted via StratumStateRow)."""
    name: str
    certified: str | None = None     # None | "checksum" | "registry"
    claimed_count: int | None = None  # corroborated cardinality claim (App. E brake)
    V: float = 0.0                   # running sum of realized c_t^2 (tighter psi than V_max)


@dataclass
class FrontierState:
    formulations: dict[str, Formulation] = field(default_factory=dict)
    # covered[record_key] = set of FORMULATION ids that surfaced it (OCCASIONS!)
    covered: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    record_stratum: dict[str, str] = field(default_factory=dict)   # record_key -> stratum
    strata: dict[str, StratumState] = field(default_factory=dict)  # stratum -> state
    T: int = 0                       # capture occasions so far (= searches issued)
    Y_CAP: int = 12                  # per-occasion novelty cap (paper Assumption (b))
    BETA: float = 1.0                # frontier-credit weight in U_hat

    # ---- per-pool counting ------------------------------------------------
    # A "pool" is a set of stratum names treated as one certification unit.
    # Pre-ER every stratum is its own pool; post-ER re-keying (§14) can merge
    # surface strata that ER decided are one entity.

    def _records(self, pool: set[str]) -> list[set[str]]:
        """Occasion-sets of the records that belong to this pool."""
        return [occs for rk, occs in self.covered.items()
                if self.record_stratum.get(rk) in pool]

    def N(self, pool: set[str]) -> int:
        """Distinct records discovered in the pool."""
        return len(self._records(pool))

    def f(self, k: int, pool: set[str]) -> int:
        """Frequency-of-frequencies: records seen in exactly k occasions.
        f(1) = singletons (Good-Turing's raw material), f(2) = doubletons."""
        return sum(1 for occs in self._records(pool) if len(occs) == k)

    # ---- the statistics ---------------------------------------------------

    def U_hat(self, pool: set[str]) -> float:
        """Frontier-adjusted Good-Turing estimate of unseen mass (paper
        eq. (GTF)): singleton rate + BETA * (frontier credit) / n.

        Frontier credit is the SUM OF RESIDUAL YIELDS of pending
        formulations that could serve this pool -- not their count. Ten
        junk formulations at 0.001 expected records must not weigh like
        ten registry sweeps at 27 (guide §7.4 locked test)."""
        n = self.N(pool)
        if n == 0:
            return 1.0            # nothing found yet: assume everything unseen
        credit = sum(fm.residual_yield for fm in self.formulations.values()
                     if not fm.issued and (fm.stratum in pool or fm.stratum is None))
        return self.f(1, pool) / n + self.BETA * credit / n

    def psi(self, pool: set[str], delta_M: float, w_g: float,
            max_occasions: int, V_realized: float | None = None) -> float:
        """Anytime-valid confidence radius for U_hat (paper's per-group
        theorem, self-normalized empirical-Bernstein form):

            psi = sqrt( 2 * V * log(1 / (delta_M * w_g)) )

        where V bounds the sum of squared per-occasion increments. The
        conservative default V_max = max_occasions * c^2 uses the worst
        case c = Y_CAP*(1+BETA)/n per occasion; passing the REALIZED
        V_t = sum c_t^2 (StratumState.V) is the guide's sanctioned
        tightening. w_g is this stratum's share of the confidence budget
        (union bound across strata: delta_M * w_g per stratum).

        HONESTY NOTE (guide: "the radius is honest, and at toy scale it
        is large"): with N_g <= 10 this dwarfs eps_g by design -- tiny
        strata close via claims or registries, not statistics. Do not
        shrink Y_CAP below what runs exhibit or drop psi to force a stop;
        the §15 calibration plot will tattle.
        """
        n = max(self.N(pool), 1)
        c = self.Y_CAP * (1 + self.BETA) / n     # per-occasion increment bound c_t
        V = V_realized if V_realized else max_occasions * c * c
        return math.sqrt(2 * V * math.log(1 / (delta_M * w_g)))

    # ---- backward-compat shims (pre-SIGMOD callers) -----------------------

    @property
    def covered_records(self):
        """Old name for `covered`. NOTE the semantic change: values are now
        FORMULATION ids (occasions), no longer source ids -- guide §7.3."""
        return self.covered

    def active_frontier_size(self, yield_threshold: float = 0.5) -> int:
        """Count of pending formulations still promising material yield.
        (Diagnostic / legacy; U_hat's credit uses summed yields, not this.)"""
        return sum(1 for f in self.formulations.values()
                   if not f.issued and f.residual_yield >= yield_threshold)

    def chao_m0(self, pool: set[str]) -> float:
        """OPTIONAL capture-recapture brake (paper App. C): Chao1 lower
        bound on the number of unseen records, m0 ~ f1^2 / (2 f2), with
        the bias-corrected f1(f1-1)/2 form when f2 = 0. It can only
        FORBID stopping (behind config.USE_CHAO_BRAKE), never enable it."""
        f1, f2 = self.f(1, pool), self.f(2, pool)
        if self.T < 2:
            return float("inf")   # too few occasions to estimate anything
        corr = (self.T - 1) / self.T
        return corr * f1 * f1 / (2 * f2) if f2 > 0 else corr * f1 * (f1 - 1) / 2.0


# ---------------------------------------------------------------------------
# Stratum bookkeeping helpers used by the loop (pipeline.py)
# ---------------------------------------------------------------------------

def stratum_of(mention, state: FrontierState) -> str:
    """Pre-ER stratum key = normalized entity surface. Registers a
    StratumState the first time a stratum appears. After ER (§14) the
    whole record_stratum map is re-keyed by entity_id in one pass --
    never keep two stratum vocabularies (guide §7.3)."""
    g = normalize_surface(mention.entity_surface)
    if g not in state.strata:
        state.strata[g] = StratumState(name=g)
    return g


def stratum_pools(state: FrontierState):
    """Yield (stratum, pool) certification units. Pre-ER: one per stratum."""
    for g in state.strata:
        yield g, {g}


def uncertified_strata(state: FrontierState) -> list[str]:
    return [g for g, s in state.strata.items() if not s.certified]


def w_g(state: FrontierState, g: str) -> float:
    """Stratum g's share of the confidence budget delta_M: a uniform
    union-bound split. sum_g w_g = 1, so P(any stratum's radius fails)
    <= sum_g delta_M * w_g = delta_M."""
    return 1.0 / max(len(state.strata), 1)


def add_if_novel(state: FrontierState, fm: Formulation) -> bool:
    """Insert a formulation unless an order-free token duplicate is already
    on the frontier (issued or not). Prevents the LLM's rephrasings -- and
    the claims engine's re-proposed gap formulations (guide §7.3) -- from
    inflating frontier credit with the same search twice."""
    key = " ".join(sorted(re.findall(r"\w+", fm.query.lower())))
    for existing in state.formulations.values():
        if " ".join(sorted(re.findall(r"\w+", existing.query.lower()))) == key:
            return False
    state.formulations[fm.formulation_id] = fm
    return True


def update_yield_estimates(state: FrontierState, fm: Formulation) -> float:
    """Global multiplicative calibration of the LLM's p_success estimates.

    NOTE: the guide names this hook but leaves its body unspecified; this
    is our (flagged) implementation choice. After each issuance, compare
    total realized yield against total predicted yield over all issued
    formulations and shrink pending, non-gap-directed formulations'
    p_success by the (clipped, add-one smoothed) ratio -- re-derived from
    each formulation's ORIGINAL _p0, so repeated calls never compound.
    Only shrinks (factor <= 1): frontier credit may deflate toward what
    searches actually deliver, never inflate. Returns the factor."""
    issued = [f for f in state.formulations.values() if f.issued]
    predicted = sum(f._p0 * f.yield_if_success for f in issued)
    realized = sum(f.realized_yield for f in issued)
    factor = min(1.0, max(0.25, (realized + 1.0) / (predicted + 1.0)))
    for f in state.formulations.values():
        if not f.issued and not f.gap_directed:
            f.p_success = f._p0 * factor
    return factor


def prune_formulations(state: FrontierState, g: str) -> int:
    """Once stratum g is certified (checksum/registry), pending searches
    aimed at g promise nothing: zero their p_success. The formulation row
    stays in the log (auditable) but residual_yield -> 0, so U_hat's
    credit drops and the argmax never picks it. Generic (stratum=None)
    formulations survive -- they may still serve other strata."""
    dropped = 0
    for f in state.formulations.values():
        if not f.issued and f.stratum == g:
            f.p_success = 0.0
            f._p0 = 0.0
            dropped += 1
    return dropped


# ---------------------------------------------------------------------------
# Single-class frontier prune (design doc Definition 17 / Sec. 6.6).
# Kept from the pre-SIGMOD implementation; called by the fragmentation
# layer once single_class_sufficiency() fires. Same zero-don't-delete
# policy as prune_formulations above.
# ---------------------------------------------------------------------------

def prune_for_single_class(state: FrontierState, keep_class,
                           formulation_class_predictor) -> int:
    """Prune pending formulations not expected to yield keep_class."""
    dropped = 0
    for f in state.formulations.values():
        if f.issued:
            continue              # already spent; nothing to save
        if formulation_class_predictor(f.query) != keep_class:
            f.p_success = 0.0     # residual_yield property -> 0
            f._p0 = 0.0
            dropped += 1
    return dropped


# ---------------------------------------------------------------------------
# Ch.-9 integration duty #1: RE-KEY STRATA after entity resolution.
# ---------------------------------------------------------------------------

def rekey_strata(state: FrontierState, mention_to_entity: dict[str, str],
                 mentions: list) -> dict:
    """Map every record's stratum from SURFACE FORM to entity_id and rebuild
    the frontier's bookkeeping in one pass (guide ch. 9: "Re-key strata").

    Why this moves the statistics: if "Acme Corp" and "ACME" merge, their
    capture histories merge -- a record covered once under each key becomes
    a DOUBLETON (occasion sets are UNIONED), so f1 drops, f2 rises, and both
    U_hat and Chao move. N also drops when two surface records collapse into
    one entity record. The per-stratum methods (N, f, U_hat, psi) read
    `covered` / `record_stratum` directly, so re-keying those maps IS the
    recomputation.

    Merge semantics for StratumState (flagged repo choices -- the guide
    specifies the re-key, not the merged-state fields):
      * V             -> SUMMED (both histories contributed occasion
                         variance; a larger, conservative radius).
      * claimed_count -> MAX of members (the strongest cardinality brake).
      * certified     -> kept only if EVERY merged member carried the SAME
                         certificate; any mixed merge RESETS to None --
                         a certificate issued for half an entity says
                         nothing about the union. (Sec.-13 revalidation
                         re-derives certificates post-ER; resetting here is
                         the conservative direction.)

    Surfaces with no resolved mention keep their surface key (nothing to
    re-key). If ER SPLIT one surface across several entities, the surface
    maps to the entity that owns the majority of its mentions, and the
    ambiguity is reported in the returned info dict.

    Returns an info dict for the mandated `strata_rekey` log event:
    {n_strata_before, n_strata_after, n_merges, ambiguous_surfaces}.
    """
    # 1. surface -> entity_id, by majority vote over that surface's mentions
    votes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for m in mentions:
        eid = mention_to_entity.get(m.mention_id)
        if eid is not None:
            votes[normalize_surface(m.entity_surface)][eid] += 1
    surface_to_entity: dict[str, str] = {}
    ambiguous: list[str] = []
    for surf, tally in votes.items():
        # deterministic winner: highest count, ties broken by entity id
        winner = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        surface_to_entity[surf] = winner
        if len(tally) > 1:
            ambiguous.append(surf)          # ER split this surface: report it

    n_before = len(state.strata)

    # 2. re-key covered + record_stratum in ONE pass, unioning occasion sets
    new_covered: dict[str, set[str]] = defaultdict(set)
    new_record_stratum: dict[str, str] = {}
    for rk, occs in state.covered.items():
        surf, _, kind = rk.rpartition("|")   # record key = "surface|kind" (§7.3)
        g_new = surface_to_entity.get(surf, surf)   # unresolved: keep surface
        nrk = f"{g_new}|{kind}"
        new_covered[nrk] |= occs             # the union IS the doubleton effect
        new_record_stratum[nrk] = g_new

    # 3. merge StratumState objects under their new keys
    new_strata: dict[str, StratumState] = {}
    members: dict[str, list[StratumState]] = defaultdict(list)
    for g, S in state.strata.items():
        members[surface_to_entity.get(g, g)].append(S)
    for g_new, group in members.items():
        certs = {S.certified for S in group}
        new_strata[g_new] = StratumState(
            name=g_new,
            # unanimous certificate survives; any disagreement resets (see above)
            certified=certs.pop() if len(certs) == 1 else None,
            claimed_count=max((S.claimed_count for S in group
                               if S.claimed_count is not None), default=None),
            V=sum(S.V for S in group),
        )

    state.covered = new_covered
    state.record_stratum = new_record_stratum
    state.strata = new_strata
    return {"n_strata_before": n_before,
            "n_strata_after": len(new_strata),
            "n_merges": n_before - len(new_strata),
            "ambiguous_surfaces": ambiguous,
            # the map itself, so downstream indexes keyed by surface (the
            # claims engine's, ch. 9) can follow the same re-key. Callers
            # should drop it from log extras -- it can be large.
            "surface_to_entity": surface_to_entity}
