from dataclasses import dataclass, field
from collections import defaultdict
import math


@dataclass
class Formulation:
    formulation_id: str
    query: str                 # the actual search string, e.g. "Acme funding round"
    expected_yield: float                       # the LLM's initial estimate
    issued: bool = False                        # have we run this search yet?
    realized_yield: int = 0                     # how many new records it actually found. Filled in after issuance
    residual_yield: float = field(init=False)   # how promising it still looks。 Updated as we discover things

    def __post_init__(self):
        self.residual_yield = self.expected_yield


@dataclass
class FrontierState:
    formulations: dict[str, Formulation] = field(default_factory=dict)
    # covered_records[record_key] = set of source_ids that covered it
    covered_records: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    N: int = 0                                  # cumulative distinct records found

    def singletons(self) -> int:
        """r_t: records covered by exactly one source so far."""
        return sum(1 for srcs in self.covered_records.values() if len(srcs) == 1)

    def active_frontier_size(self, yield_threshold: float = 0.5) -> int:
        return sum(1 for f in self.formulations.values()
                   if not f.issued and f.residual_yield >= yield_threshold)

    def U_hat(self, beta: float = 1.0) -> float:
        if self.N == 0:
            return 1.0
        return self.singletons() / self.N + beta * self.active_frontier_size() / self.N


# ---------------------------------------------------------------------------
# Single-class frontier prune (design doc Definition 17 / Sec. 6.6)
# Called from the pipeline once single_class_sufficiency() fires.
# ---------------------------------------------------------------------------

def prune_for_single_class(state: FrontierState, keep_class,
                           formulation_class_predictor) -> int:
    """Prune pending formulations not expected to yield keep_class.

    We zero residual_yield instead of deleting: the formulation stays in the
    log (auditable, per the design doc's data contracts) but argmax-by-
    residual in the loop will never pick it, and active_frontier_size() no
    longer counts it -- so U_hat's frontier-credit term drops accordingly.
    Returns how many formulations were pruned.
    """
    dropped = 0
    for f in state.formulations.values():
        if f.issued:
            continue  # already spent; nothing to save
        if formulation_class_predictor(f.query) != keep_class:
            f.residual_yield = 0.0
            dropped += 1
    return dropped
