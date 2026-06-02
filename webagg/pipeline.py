import uuid
from .frontier import FrontierState, Formulation
from .search import SerperBackend
from .fetch import fetch_url
from .extract import is_relevant, extract_mentions
from .llm import call_llm
from .metrics import log_measurement, log_formulation
from .storage import get_session
from .type_defs import Source, Mention


def seed_formulations(query: str) -> list[Formulation]:
    sys = ("Propose 5 diverse initial search formulations for the user's query. "
           'Return JSON: {"formulations":[{"query":"...","expected_yield":1-10}]}')
    out = call_llm(system=sys, user=query)["payload"]
    return [Formulation(formulation_id=str(uuid.uuid4())[:8],
                        query=f["query"],
                        expected_yield=float(f["expected_yield"]))
            for f in out["formulations"]]


def propose_followups(record_kind: str, entity_surface: str,
                      already_tried: list[str]) -> list[Formulation]:
    sys = open("prompts/propose_followups.txt").read()
    user = (f"record_kind: {record_kind}\nentity: {entity_surface}\n"
            f"already_tried: {already_tried[-20:]}")      # cap context
    out = call_llm(system=sys, user=user)["payload"]
    return [Formulation(formulation_id=str(uuid.uuid4())[:8],
                        query=f["query"],
                        expected_yield=float(f["expected_yield"]))
            for f in out.get("formulations", [])]


def run_query(query: str, *, run_id: str, eps: float = 0.10,
              delta: float = 0.10, eta: float = 0.5, max_steps: int = 200):
    session = get_session(f"data/runs/{run_id}.sqlite")
    search = SerperBackend()
    state = FrontierState()
    for f in seed_formulations(query):
        state.formulations[f.formulation_id] = f
        log_formulation(session, run_id, f, step=0)

    for step in range(1, max_steps + 1):
        # 1. pick highest-residual unissued formulation
        candidates = [f for f in state.formulations.values() if not f.issued]
        if not candidates:
            break
        f = max(candidates, key=lambda x: x.residual_yield)

        # 2. issue search
        results = search.search(f.query, k=10)
        new_records_this_step = 0

        # 3. fetch + extract
        for r in results:
            src = fetch_url(r["url"], formulation_id=f.formulation_id)
            if src is None or not is_relevant(src, query):
                continue
            session.add(src.to_row())
            mentions = extract_mentions(src, query)
            for m in mentions:
                session.add(m.to_row())
                record_key = f"{m.entity_surface}|{m.record_kind}"
                was_new = record_key not in state.covered_records
                state.covered_records[record_key].add(src.source_id)
                if was_new:
                    new_records_this_step += 1
                    state.N += 1
                    # grow the frontier
                    for nf in propose_followups(
                            m.record_kind, m.entity_surface,
                            already_tried=[x.query for x in state.formulations.values()]):
                        if nf.query not in {x.query for x in state.formulations.values()}:
                            state.formulations[nf.formulation_id] = nf

        f.issued = True
        f.realized_yield = new_records_this_step
        f.residual_yield = 0.0

        # 4. log measurements
        U = state.U_hat()
        log_measurement(session, run_id, step, "U_hat", U,
                        extra={"r": state.singletons(),
                               "N": state.N,
                               "frontier_active": state.active_frontier_size(),
                               "formulation": f.query,
                               "new_records": new_records_this_step})
        session.commit()

        # 5. stopping test
        best_residual = max((x.residual_yield for x in state.formulations.values()
                             if not x.issued), default=0.0)
        if U < eps / 2 and best_residual < eta:
            log_measurement(session, run_id, step, "stop", 1.0,
                            extra={"reason": "U<eps/2 and frontier exhausted"})
            break

    return state, session
