from dataclasses import dataclass, field
from collections import defaultdict
import math


@dataclass
class Formulation:
    formulation_id: str
    query: str
    expected_yield: float                       # the LLM's initial estimate
    issued: bool = False
    realized_yield: int = 0                     # filled in after issuance
    residual_yield: float = field(init=False)   # updated as we discover things

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
