import uuid
from .frontier import (FrontierState, Formulation, StratumState,
                       stratum_of, stratum_pools, uncertified_strata, w_g,
                       add_if_novel, update_yield_estimates,
                       prune_formulations, normalize_surface)
from .search import SerperBackend
from .fetch import fetch_url
from .extract import is_relevant, extract_certified
from .llm import call_llm, set_llm_logger, set_llm_step
from .fetch import clear_fetch_cache
from .metrics import (log_measurement, log_formulation, log_stop,
                      persist_stratum_states)
from .storage import get_session, RejectedSourceRow
from .calibration import ConformalGate, load_calibration_set
from . import config
from .type_defs import Source, Mention
from collections import defaultdict
from .storage import load_sources, load_mentions
from .canonicalize import canonicalize_value
from .corroboration import corroborate
from .fragmentation import classify_all_records, entity_mentioned
from .claims import ClaimsEngine, CoverageView



def seed_formulations(query: str) -> list[Formulation]:
    """Ask the LLM for initial searches, priced with the two-point yield
    model (guide ch. 7 / paper App. B): p_success = P(>=1 new record),
    yield_if_success = E[new records | success]."""
    sys = ("Propose 5 diverse initial search formulations for the user's query. "
           "For each, estimate p_success (probability in [0,1] that the search "
           "surfaces at least one NEW relevant record) and yield_if_success "
           "(expected number of new records if it succeeds, 1-10). Return JSON: "
           '{"formulations":[{"query":"...","p_success":0.0,"yield_if_success":1}]}')
    out = call_llm(system=sys, user=query)["payload"]
    fs = []
    for f in out["formulations"]:
        fs.append(Formulation(
            formulation_id=str(uuid.uuid4())[:8],
            query=f["query"],
            # tolerate old-style replies: expected_yield -> p=1, y=value
            p_success=float(f.get("p_success", 1.0)),
            yield_if_success=float(f.get("yield_if_success",
                                         f.get("expected_yield", 1.0))),
            cost=config.SEARCH_COST_USD))
    return fs


def propose_followups(record_kind: str, entity_surface: str,
                      already_tried: list[str]) -> list[Formulation]:
    """Frontier growth after a discovery. Signature kept from ch. 5/6 (the
    e2e harness patches this seam); the guide's pseudocode passes (m, state)
    but only uses these three fields."""
    sys = open("prompts/propose_followups.txt").read()
    user = (f"record_kind: {record_kind}\nentity: {entity_surface}\n"
            f"already_tried: {already_tried[-20:]}")      # cap context
    out = call_llm(system=sys, user=user)["payload"]
    return [Formulation(formulation_id=str(uuid.uuid4())[:8],
                        query=f["query"],
                        p_success=float(f.get("p_success", 1.0)),
                        yield_if_success=float(f.get("yield_if_success",
                                               f.get("expected_yield", 1.0))),
                        cost=config.SEARCH_COST_USD)
            for f in out.get("formulations", [])]


def all_strata_pass(state: FrontierState, eps_g: float, delta_M: float,
                    eta: float, max_steps: int) -> bool:
    """The per-stratum stop test (guide ch. 7 / paper §3.3, Def. in §3.3).

    For every uncertified stratum g, BOTH conjuncts must hold:
      (i)  U_hat_g + psi_g < eps_g      -- the anytime-valid certificate
      (ii) no pending formulation that could serve g still promises
           residual yield >= eta        -- frontier exhausted for g
    Plus two brakes that can only FORBID stopping:
      * cardinality (App. E, wired early per the guide): a corroborated
        COUNT claim of n while N_g < n is a hard no.
      * Chao capture-recapture (App. C), behind config.USE_CHAO_BRAKE.
    """
    for g, pool in stratum_pools(state):
        S = state.strata[g]
        if S.certified:
            continue                    # closed by checksum/registry: exempt
        if S.claimed_count is not None and state.N(pool) < S.claimed_count:
            return False                # cardinality hard brake (App. E)
        U = state.U_hat(pool)
        psi = state.psi(pool, delta_M, w_g(state, g), max_steps,
                        V_realized=S.V or None)
        hot = any(not f.issued and f.residual_yield >= eta
                  and f.stratum in (g, None)
                  for f in state.formulations.values())
        ok = (U + psi < eps_g) and (not hot)   # the TWO conjuncts (paper §3.3)
        if config.USE_CHAO_BRAKE:              # optional third (App. C)
            m0 = state.chao_m0(pool)
            ok = ok and m0 / (state.N(pool) + m0 + 1e-9) <= eps_g
        if not ok:
            return False
    return True


def run_query(query: str, *, run_id: str, eps: float = config.EPS_G,
              delta: float = config.DELTA_M, eta: float = config.ETA,
              max_steps: int = 200, budget_usd: float = config.BUDGET_USD):
    """The frontier loop (guide ch. 7 / paper §3): maintain a frontier of
    intended searches, track discovery per capture occasion, stop PER
    STRATUM by the two-conjunct rule. eps/delta/eta keep their ch. 5/6
    kwarg names but now mean eps_g / delta_M / the hot-frontier threshold."""
    session = get_session(f"data/runs/{run_id}.sqlite")
    set_llm_logger(session, run_id)   # ch. 5: every LLM call -> measurements
    clear_fetch_cache()               # ch. 5: URL cache is per-run

    # ch. 6: the conformal gate. Unfitted (no calibration file yet) it runs
    # in bootstrap accept-all mode and stamps 'gate_uncalibrated'.
    gate = ConformalGate(delta_E=config.DELTA_E)
    cal = load_calibration_set(config.CALIBRATION_SET)
    if cal:
        gate.fit(cal)

    search = SerperBackend()
    state = FrontierState(Y_CAP=config.Y_CAP, BETA=config.BETA)
    claims_engine = ClaimsEngine(session)     # ch. 7 stub; full engine in §11
    seen_urls: set[str] = set()
    for fm in seed_formulations(query):
        state.formulations[fm.formulation_id] = fm
        log_formulation(session, run_id, fm, step=0)
    spent = 0.0
    step = 0

    for step in range(1, max_steps + 1):
        # ---- 1. pick the next search --------------------------------------
        # Reservation-index ordering (App. B, behind a flag). lam = value of
        # one new record; once every stratum is certified, lam := 0 makes all
        # indices negative -> exploration halts ECONOMICALLY (not a
        # certificate -- log_stop records the difference).
        cands = [f for f in state.formulations.values() if not f.issued]
        if not cands:
            log_stop(session, run_id, step, reason="frontier_exhausted")
            break
        if config.USE_ECONOMIC_ORDER:
            # lam -> 0 only once strata EXIST and are ALL certified (App. B).
            # COLD-START FIX: before the first mention arrives, state.strata
            # is empty -- an empty set means "not started", not "all done";
            # reading it as done zeroed lam and economic-stopped at step 1.
            all_done = bool(state.strata) and not uncertified_strata(state)
            lam = 0.0 if all_done else config.LAMBDA_PER_RECORD
            fm = max(cands, key=lambda x: x.reservation_index(lam))
            if fm.reservation_index(lam) <= 0:
                log_stop(session, run_id, step, reason="economic")   # NOT a certificate
                break
        else:
            fm = max(cands, key=lambda x: x.residual_yield)

        set_llm_step(step)            # cost rows carry the agent step

        # ---- 2. issue the search = ONE capture occasion -------------------
        state.T += 1
        new = 0
        for r in search.search(fm.query, k=10, formulation_id=fm.formulation_id):
            if r["url"] in seen_urls:
                continue              # already processed this run (any verdict)
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
            # -- stages 2-4 (ch. 6): dual extraction -> validators -> gate.
            mentions, claims, info = extract_certified(src, query, gate=gate)
            for c in claims:
                claims_engine.ingest(c)         # persisted + indexed (ch. 7)
            log_measurement(session, run_id, step, "extract_agreed",
                            info["agreed"], extra=info)
            log_measurement(session, run_id, step, "extract_abstained",
                            info["disagreed"] + info["gate_abstains"])
            for m in mentions:
                session.add(m.to_row())
                # record key AND stratum key share one normalizer (§7.3:
                # never keep two stratum vocabularies)
                rk = f"{normalize_surface(m.entity_surface)}|{m.record_kind}"
                state.record_stratum[rk] = stratum_of(m, state)
                was_new = rk not in state.covered
                state.covered[rk].add(fm.formulation_id)   # OCCASIONS, not sources!
                if was_new and new < state.Y_CAP:          # per-occasion novelty cap
                    new += 1
                    for nf in propose_followups(
                            m.record_kind, m.entity_surface,
                            already_tried=[x.query for x in state.formulations.values()]):
                        if add_if_novel(state, nf):
                            log_formulation(session, run_id, nf, step=step)
        fm.issued, fm.realized_yield = True, new
        spent += fm.cost
        log_formulation(session, run_id, fm, step=step)    # update issued/realized
        # realized-V tightening (guide: track V_t = sum c_t^2 in lieu of V_max)
        for g, pool in stratum_pools(state):
            n = max(state.N(pool), 1)
            c = state.Y_CAP * (1 + state.BETA) / n
            state.strata[g].V += c * c
        update_yield_estimates(state, fm)     # calibrate pending p_success (our hook)

        # ---- 3. claims engine: cardinality brake + gap direction ----------
        # (checksum CERTIFICATION is a §11 feature; the ch.-7 stub can only
        # arm the App. E brake and point searches at known count gaps.)
        claims_engine.update_cardinality_brakes(state)
        for g in uncertified_strata(state):
            st = claims_engine.checksum(g, CoverageView(n_records=state.N({g})))
            if st.certified:                   # unreachable until §11; kept for shape
                state.strata[g].certified = "checksum"
                prune_formulations(state, g)
            elif st.gap:
                for gf in claims_engine.gap_formulations(g, st.gap):
                    if add_if_novel(state, gf):
                        log_formulation(session, run_id, gf, step=step)

        # ---- 4. per-stratum measurements (the §15 calibration plot eats these)
        for g, pool in stratum_pools(state):
            S = state.strata[g]
            log_measurement(
                session, run_id, step, "U_hat", state.U_hat(pool), stratum=g,
                extra={"N": state.N(pool), "f1": state.f(1, pool),
                       "f2": state.f(2, pool),
                       "psi": state.psi(pool, delta, w_g(state, g), max_steps,
                                        V_realized=S.V or None),
                       "m0": state.chao_m0(pool), "T": state.T, "new": new,
                       "claimed_count": S.claimed_count,
                       "formulation": fm.query})
        session.commit()

        # ---- 5. stop test -------------------------------------------------
        if spent >= budget_usd:
            log_stop(session, run_id, step, reason="budget",
                     extra={"spent": spent})               # NOT a certificate
            break
        # COLD-START GUARD: with zero strata all_strata_pass is vacuously
        # True; require at least one discovered stratum before a
        # "certified" stop can fire (an empty run cannot certify anything).
        if state.strata and all_strata_pass(state, eps, delta, eta, max_steps):
            log_stop(session, run_id, step, reason="certified")   # the real thing
            break

    persist_stratum_states(session, run_id, state, step)   # snapshot certificates
    session.commit()
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
