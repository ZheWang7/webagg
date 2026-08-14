"""The fidelity-certification harness (seam A3; guide Sec. 13 + 15, paper §7).

Supplies the two callables Sec. 13's learn_then_test() has been waiting for
-- run_pipeline(g, lam) and truth(g) -- and the machinery around them.

DESIGN: ONE FETCHED POOL PER ENTITY, LAMBDA VARIED IN REPLAY.

The paper certifies Pi_lambda, "the map from fetched sources to contributed
records" -- and every knob the pre-committed LTT grid bundles (tau_+/-,
delta_E, qbar) acts strictly AFTER fetching: the gate filters extracted
mentions, the matcher thresholds ER, qbar caps corroboration. So the
harness runs the expensive discovery ONCE per calibration entity under the
paper-default configuration (the reference run, registry DENIED), and then
evaluates every lambda by REPLAYING resolve_and_aggregate over that frozen
pool -- exact, not an approximation, because Pi_lambda's input (the fetched
pool) is held fixed and Pi_lambda itself is recomputed in full.

  Documented interpretation (the guide does not spell the mechanics): a
  full refetch per lambda would also let lambda influence WHICH sources get
  fetched (acceptance feeds the frontier's stopping statistics). We treat
  the pool as Pi_lambda's input, per the paper's definition, and note two
  practical consequences: (i) 4 lambdas x N entities costs N live runs, not
  4N; (ii) the certificate covers the composition applied to pools gathered
  under the reference config -- which is exactly the config live runs use.

  Hard constraint inherited from storage: persisted mentions are the live
  gate's SURVIVORS (rejects are never stored), so replay can only TIGHTEN
  delta_E. The grid satisfies this (0.05 -> 0.02); replay() asserts it.

  Replay is offline for search/fetch/extraction but LIVE for the ER
  adjudicator: band pairs escalate to the real LLM, per lambda. That is the
  configuration being certified -- stubbing the adjudicator here would
  certify a pipeline that never runs (the fake-in-path rule).

WHY THE REGISTRY IS DENIED IN REFERENCE RUNS: the same cohort's truth
tables came from EDGAR (build_truth.py). An agent that read EDGAR would be
graded against its own answer key and every loss would be a fiction --
the exam analogy's whole point. reference_runs() therefore hard-requires a
non-empty denylist covering the registry.

SEAM DISCIPLINE (both enforced, not advised): certification REFUSES to run
with an unfitted gate (no calibration set) or a cold matcher (no labeled
pairs). Bootstrap modes exist so the pipeline can run before data exists;
a certificate produced through them would be a number wearing a costume.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from . import config
from .calibration import ConformalGate, load_calibration_set
from .er_pairs import load_fitted_matcher
from .formd import load_truth_cohort
from .pipeline import resolve_and_aggregate, run_query
from .risk_control import TruthEntity
from .storage import get_session

# The withheld registry for the Form D cohort: deny ALL its hosts (the
# denylist matches by registered domain, but being explicit costs nothing).
REGISTRY_DENY = ("sec.gov", "www.sec.gov", "data.sec.gov")

# The attributes certification queries ask for: amount feeds the loss,
# date feeds the (kind|date) truth key. Same pair run_query.py uses.
CERT_ATTRS = frozenset({"amount", "date"})


def entity_query(entity_name: str) -> str:
    """The one query template all certification runs share -- part of the
    frozen configuration (a per-entity prompt tweak would be tuning)."""
    return f"total funding raised by {entity_name}"


def query_name(manifest: dict, eid: str, names: dict | None) -> str:
    """The name the OPEN-WEB query uses for entity eid.

    Attempt-#2 lesson (pre-registered): the manifest carries the REGISTRY
    legal name ("Maplebear Inc."), which the open web barely uses -- a
    reference run under it probes a name, not a pipeline. `names` maps
    eid -> the common name the press uses ("Instacart"); entities without
    an override keep the registry name. The chosen name is stored in the
    runs index, so the certificate's provenance shows exactly what was
    asked.
    """
    return (names or {}).get(eid) or manifest["entities"][eid]["entity_name"]


def _runs_index_path(domain: str) -> Path:
    return config.FIDELITY_CERT_DIR / f"{domain}.runs.json"


def load_runs_index(domain: str) -> dict:
    """The reference-run ledger: entity_id -> {db, run_id, delta_E_ref, ...}.
    {} when no reference runs exist yet."""
    p = _runs_index_path(domain)
    return json.loads(p.read_text()) if p.exists() else {}


# ---------------------------------------------------------------------------
# Phase 1: reference runs (live, expensive, once per entity)
# ---------------------------------------------------------------------------

def reference_runs(manifest: dict, *, domain: str,
                   deny: tuple = REGISTRY_DENY,
                   eps: float = 0.10, delta: float = 0.10,
                   max_steps: int = 60,
                   budget_usd: float = config.BUDGET_USD,
                   refresh: bool = False,
                   names: dict | None = None,
                   refresh_entities: tuple = ()) -> dict:
    """Run discovery ONCE per CALIBRATION entity; return the runs index.

    Idempotent: an entity whose run DB already exists is skipped unless
    refresh=True (the stale-output lesson: the skip is printed, never
    silent). The validation half is deliberately untouched -- its runs
    belong to experiment 8, after certification, and doing them now would
    burn the holdout.
    """
    if not deny:
        raise ValueError(
            "reference runs REQUIRE the registry denylist: the cohort's "
            "truth tables come from EDGAR, and an agent that can read the "
            "registry is grading itself against its own answer key.")
    index = load_runs_index(domain)
    for eid in manifest["split"]["calibration"]:
        name = query_name(manifest, eid, names)
        run_id = f"cert_{domain}_{eid}"
        db = str(config.RUNS_DIR / f"{run_id}.sqlite")
        force = refresh or eid in refresh_entities
        if not force and eid in index and Path(index[eid]["db"]).exists():
            print(f"[certify] {eid} ({name}): reference run exists -- skip")
            continue
        if force:
            Path(db).unlink(missing_ok=True)   # stale-output lesson: a
            # forced refresh deletes the old pool, else run_query appends
            # into a mixed-vintage DB
        print(f"[certify] {eid} ({name}): live reference run "
              f"(deny={list(deny)}, query={entity_query(name)!r}) ...")
        state, session = run_query(entity_query(name), run_id=run_id,
                                   eps=eps, delta=delta,
                                   max_steps=max_steps,
                                   budget_usd=budget_usd,
                                   query_attributes=CERT_ATTRS,
                                   deny=tuple(deny))
        engine = session.get_bind()
        session.close()
        engine.dispose()                 # Windows: release the sqlite handle
        index[eid] = {
            "db": db, "run_id": run_id, "entity_name": name,
            "query": entity_query(name),
            # the reference gate level: replay may only TIGHTEN below this
            "delta_E_ref": config.DELTA_E,
            "deny": list(deny), "eps": eps, "delta": delta,
            "max_steps": max_steps,
        }
        p = _runs_index_path(domain)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(index, indent=2, sort_keys=True))
    return index


# ---------------------------------------------------------------------------
# Phase 2: lambda replay over the frozen pools
# ---------------------------------------------------------------------------

def _fitted_gate(delta_E: float) -> ConformalGate:
    """A gate at lambda's delta_E, fitted from the A1 calibration set.
    REFUSES bootstrap: certifying through an accept-all gate is meaningless."""
    cal = load_calibration_set(config.CALIBRATION_SET)
    if not cal:
        raise RuntimeError(
            f"no calibration set at {config.CALIBRATION_SET} -- the gate "
            f"would run in bootstrap accept-all mode, and a certificate "
            f"produced through it certifies nothing (seam A1 must be "
            f"closed before A3).")
    return ConformalGate(delta_E=delta_E).fit(cal)


def _fitted_matcher(lam: dict, cache: dict):
    """lambda's matcher (tau thresholds from the grid), fitted from the A2
    pair set and cached per (tau-, tau+). REFUSES cold start."""
    key = (lam["tau_minus"], lam["tau_plus"])
    if key not in cache:
        m = load_fitted_matcher(tau_minus=lam["tau_minus"],
                                tau_plus=lam["tau_plus"])
        if m.alpha is None:
            raise RuntimeError(
                "matcher is cold (no/too few labeled pairs at "
                f"{config.MATCH_PAIRS}) -- certification refuses the "
                "hand-tuned fallback (seam A2 must be closed before A3).")
        cache[key] = m
    return cache[key]


def replay(db_path: str, lam: dict, *, delta_E_ref: float,
           matcher_cache: Optional[dict] = None) -> list:
    """Pi_lambda over one frozen pool: re-gate, re-resolve, re-corroborate.

    Returns the resolved records (the pipeline's plain dicts) --
    exactly what fidelity_loss consumes. state=None on purpose: checksum
    revocation and report regimes do not change the assembled VALUES, and
    the certificate is about values.
    """
    if lam["delta_E"] > delta_E_ref + 1e-12:
        raise ValueError(
            f"lambda wants delta_E={lam['delta_E']} but the reference pool "
            f"was gated at {delta_E_ref}: rejected mentions were never "
            f"persisted, so replay can only TIGHTEN the gate. Fix the grid "
            f"or refresh the reference runs at the loosest delta_E.")
    session = get_session(db_path)
    try:
        result = resolve_and_aggregate(
            session, run_id=f"replay_{Path(db_path).stem}",
            query_attributes=set(CERT_ATTRS),
            aggregate_attr="amount",
            gate=_fitted_gate(lam["delta_E"]),
            matcher=_fitted_matcher(lam, matcher_cache
                                    if matcher_cache is not None else {}),
            qbar=lam["qbar"],
            state=None)
        return result["records"]
    finally:
        engine = session.get_bind()
        session.close()
        engine.dispose()


def make_callables(runs_index: dict, cohort_dir: Path
                   ) -> tuple[Callable, Callable]:
    """The (run_pipeline, truth) pair learn_then_test() takes.

    run_pipeline(g, lam) -> resolved records of entity g under lambda
    truth(g)             -> the TruthEntity answer key from build_truth.py
    """
    truths: dict[str, TruthEntity] = load_truth_cohort(cohort_dir)
    matcher_cache: dict = {}             # shared across entities AND lambdas

    def run_pipeline(g: str, lam: dict):
        info = runs_index[g]
        return replay(info["db"], lam,
                      delta_E_ref=info["delta_E_ref"],
                      matcher_cache=matcher_cache)

    def truth(g: str) -> TruthEntity:
        return truths[g]

    return run_pipeline, truth
