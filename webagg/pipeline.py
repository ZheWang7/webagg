import uuid
from .frontier import FrontierState, Formulation
from .search import SerperBackend
from .fetch import fetch_url
from .extract import is_relevant, extract_certified
from .llm import call_llm, set_llm_logger, set_llm_step
from .fetch import clear_fetch_cache
from .metrics import log_measurement, log_formulation
from .storage import get_session, RejectedSourceRow
from .calibration import ConformalGate, load_calibration_set
from . import config
from .type_defs import Source, Mention
from collections import defaultdict
from .storage import load_sources, load_mentions
from .canonicalize import canonicalize_value
from .corroboration import corroborate
from .fragmentation import classify_all_records, entity_mentioned



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
    set_llm_logger(session, run_id)   # ch. 5: every LLM call -> measurements
    clear_fetch_cache()               # ch. 5: URL cache is per-run

    # ch. 6: the conformal gate. Unfitted (no calibration file yet) it runs
    # in bootstrap accept-all mode and stamps 'gate_uncalibrated' on every
    # mention -- see ConformalGate.accept.
    gate = ConformalGate(delta_E=config.DELTA_E)
    cal = load_calibration_set(config.CALIBRATION_SET)
    if cal:
        gate.fit(cal)

    search = SerperBackend()
    state = FrontierState()
    seen_urls: set[str] = set()
    for f in seed_formulations(query):
        state.formulations[f.formulation_id] = f
        log_formulation(session, run_id, f, step=0)

    for step in range(1, max_steps + 1):
        # 1. pick highest-residual unissued formulation
        candidates = [f for f in state.formulations.values() if not f.issued]
        if not candidates:
            break
        f = max(candidates, key=lambda x: x.residual_yield)

        set_llm_step(step)            # cost rows carry the agent step
        # 2. issue search -- one call = one capture occasion
        # results come back tagged with the formulation that surfaced them.
        results = search.search(f.query, k=10, formulation_id=f.formulation_id)
        new_records_this_step = 0

        # 3. fetch + extract
        for r in results:
            if r["url"] in seen_urls:
                continue          # already processed this run (any verdict)
            seen_urls.add(r["url"])
            src = fetch_url(r["url"], formulation_id=r["formulation_id"])
            if src is None:
                continue
            # -- stage 1 (guide §6.1): rejections are LOGGED, not dropped.
            ok, conf = is_relevant(src, query)
            if not ok:
                session.merge(RejectedSourceRow(
                    source_id=src.source_id, url=str(src.url),
                    rejection_score=conf,       # audit stratifies on this
                    main_text=src.main_text))   # audit RE-READS this later
                continue
            session.add(src.to_row())
            # -- stages 2-4: dual extraction -> validators -> conformal gate.
            mentions, claims, info = extract_certified(src, query, gate=gate)
            for c in claims:
                session.merge(c.to_row())       # merge: same claim can recur
            log_measurement(session, run_id, step, "extract_agreed",
                            info["agreed"], extra=info)
            log_measurement(session, run_id, step, "extract_abstained",
                            # each contested slot counted once (A-side);
                            # b_only is informational, in extract_agreed.extra
                            info["disagreed"] + info["gate_abstains"])
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


def resolve_and_aggregate(session, *, run_id: str, query_attributes: set[str],
                          aggregate_attr: str = "amount",
                          eps: float = 0.10, eps_er: float = 0.05,
                          cluster_fn=None, matcher=None):
    """Stages 2-5 of the pipeline (design Sec. 7.1) over an already-populated
    run DB. Split out from end_to_end so it can run offline on a fixture DB
    (no search, no LLM) and re-run on a finished live DB without re-fetching.

    cluster_fn: optional (mentions, source_lookup) -> {mention_id: entity_id}.
    Injectable like the ER adjudicator / schema relevance_fn, so tests can
    bypass the real matcher (which needs sentence-transformers + torch).
    """
    # 0. reload what discovery persisted -- the DB is the source of truth,
    # and every object still carries its provenance handle (impl Sec. 4.2)
    mentions = load_mentions(session)
    sources = {s.source_id: s for s in load_sources(session)}

    # 1. entity resolution -> the inferred join key (design Sec. 5)
    if cluster_fn is not None:
        mention_to_entity = cluster_fn(mentions, sources)
    else:
        # lazy import: pulls in sentence-transformers/torch only when the
        # real matcher is actually wanted
        from .entity_resolution import cluster_entities, Matcher
        mention_to_entity = cluster_entities(mentions, matcher or Matcher(),
                                             sources)

    # 2. re-group mentions by the RESOLVED key (entity_id, record_kind) --
    # this replaces run_query's provisional pre-ER "surface|kind" key
    by_record: dict[tuple, list] = defaultdict(list)
    for m in mentions:
        by_record[(mention_to_entity[m.mention_id], m.record_kind)].append(m)

    # 3. fragmentation: build M(rho), pick scan/join/redundant per record
    # (design Sec. 6, Algorithm 3); logs frag_case measurements (impl 10.8)
    reports = classify_all_records(by_record, sources, query_attributes,
                                   session=session, run_id=run_id)

    # 4. corroborate one value per attribute (design Sec. 3), guarding
    # fragmenting attributes against cross-entity contamination (Sec. 6.9)
    resolved = []
    for (eid, kind), rep, ms in reports:
        entity_surfaces = list({m.entity_surface for m in ms})
        record = {"entity_id": eid, "record_kind": kind,
                  "frag_case": rep.case, "attributes": {}}
        by_attr = defaultdict(list)
        for m in ms:
            if m.attribute in query_attributes:
                by_attr[m.attribute].append(m)
        for attr, attr_mentions in by_attr.items():
            by_value = defaultdict(list)
            for m in attr_mentions:
                # canonicalize BEFORE grouping, so "$40M" and "40 million USD"
                # are one candidate value, not two competitors (impl 16 #5)
                by_value[canonicalize_value(m.value)].append(m)
            cv = corroborate(by_value, sources)
            if attr in rep.fragmenting_attrs:
                # a fragmenting attribute rides on ER alone (single class, no
                # cross-class corroboration): demand that at least one
                # asserting page literally NAMES the entity, else halve
                # belief (design Sec. 6.9).
                # NOTE deviation from the guide: it checks only the FIRST
                # mention's source; we accept ANY asserting source naming
                # the entity, which is strictly less trigger-happy.
                src_ids = {m.source_id for m in attr_mentions}  # pydantic models aren't hashable; dedupe on the id
                if not any(entity_mentioned(sources[sid], entity_surfaces)
                           for sid in src_ids):
                    cv.belief = cv.belief * 0.5
            record["attributes"][attr] = cv
        resolved.append(record)

    # 5. aggregate f_Q over distinct resolved records with the three-part
    # interval (design Corollary 2); reuse Experiment 3's implementation
    from .eval_er import aggregate_with_ci   # lazy: numpy/pandas only
    rows = []
    for r in resolved:
        cv = r["attributes"].get(aggregate_attr)
        if cv is None:
            continue                      # record lacks the aggregate attr
        try:
            rows.append({"value": float(cv.value), "belief": cv.belief})
        except ValueError:
            continue                      # non-numeric value; skip, don't crash
    ci = aggregate_with_ci(rows, eps=eps, eps_er=eps_er)
    log_measurement(session, run_id, 0, "answer", ci["answer"], extra=ci)
    session.commit()
    return {"answer": ci["answer"], "ci": ci,
            "records": resolved, "reports": reports}


def end_to_end(query: str, *, run_id: str, query_attributes: set[str],
               aggregate_attr: str = "amount", mode: str = "open_web",
               schema_driver=None, query_filter=None,
               eps: float = 0.10, delta: float = 0.10, eta: float = 0.5,
               max_steps: int = 200, eps_er: float = 0.05,
               cluster_fn=None, matcher=None):
    """The full pipeline (impl Sec. 11.1): discovery, then resolve + aggregate.

    mode="open_web" runs the frontier loop (probabilistic completeness,
    Theorem 1); mode="schema" sweeps a key universe (deterministic
    completeness, Theorem 3), in which case the completeness slack eps is 0
    over the addressable closure -- the interval's recall term vanishes.
    """
    if mode == "schema":
        from .schema_addressable import run_schema_addressable
        out = run_schema_addressable(query, schema_driver,
                                     query_filter=query_filter, run_id=run_id)
        session, state = out["session"], None
        eps_effective = 0.0               # Theorem 3: delta_F = 0 over K*
    else:
        state, session = run_query(query, run_id=run_id, eps=eps,
                                   delta=delta, eta=eta, max_steps=max_steps)
        eps_effective = eps

    result = resolve_and_aggregate(session, run_id=run_id,
                                   query_attributes=query_attributes,
                                   aggregate_attr=aggregate_attr,
                                   eps=eps_effective, eps_er=eps_er,
                                   cluster_fn=cluster_fn, matcher=matcher)
    result["state"] = state
    return result
