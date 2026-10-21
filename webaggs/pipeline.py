"""FrontierDiscovery loop and end-to-end wiring (webaggr.pdf Alg. 1; impl §14.1).

Exp 2-S instantiation: one run = one player; one stratum per run (the
player's career), records are (player, season), f_Q = SUM of points.

Ordering (impl guide §14.1): FrontierDiscovery (certified extraction, claim
ingestion, checksum checks, gap-direction inside) -> ER -> re-key + SECOND
checksum pass (can revoke) -> corroborate -> aggregate f_Q -> report.
"""
from __future__ import annotations

from collections import defaultdict

from . import config, canon
from .aggregate import Answer, aggregate_with_ci
from .claims import ClaimsEngine
from .corroborate import corroborate_record
from .er import resolve_entity
from .fetch import fetch_url as real_fetch
from .frontier import Formulation, FrontierState, StratumState, all_strata_pass
from .llm import call_llm, load_prompt
from .types import sha16


# ---------------------------------------------------------------------------
def seed_formulations(player: str, *, llm=call_llm, ledger=None, run_id="",
                      depth: int = config.SEED_DEPTH) -> list[Formulation]:
    """Seed the frontier (webaggr.pdf Def. 5). Depth matters: finance Exp 2
    pilots died frontier_exhausted with budget unspent at shallow depth, so
    the seeder asks for `depth` diverse formulations and adds two
    deterministic anchors the LLM cannot forget to propose."""
    system = load_prompt("seed.txt")
    out = llm(system, f"Player: {player}\nPropose {depth} formulations.",
              ledger=ledger, run_id=run_id, purpose="seed")
    fms = []
    for f in out.get("formulations", [])[:depth]:
        q = str(f.get("query", "")).strip()
        if not q:
            continue
        fms.append(Formulation(
            formulation_id=sha16(f"seed|{q}")[:8], query=q,
            p_success=min(max(float(f.get("p_success", 0.5)), 0.05), 1.0),
            yield_if_success=max(float(f.get("yield_if_success", 2.0)), 0.1)))
    for q in (f"{player} career stats season by season",
              f"{player} total career points NBA"):
        fms.append(Formulation(formulation_id=sha16(f"seed|{q}")[:8], query=q,
                               p_success=0.9, yield_if_success=5.0))
    # dedup by query text
    uniq = {}
    for fm in fms:
        uniq.setdefault(fm.query.lower(), fm)
    return list(uniq.values())


def propose_followups(player: str, new_records: list[str], unissued: list[str],
                      *, llm=call_llm, ledger=None, run_id="") -> list[Formulation]:
    """LLM proposes follow-up searches from what was just found (Def. 5:
    the frontier is the agent's explicit model of where it still intends
    to look). One call per step, only when the step surfaced new records."""
    system = load_prompt("followups.txt")
    user = (f"Player: {player}\nSeasons found so far: {sorted(new_records)}\n"
            f"Pending queries: {unissued[:15]}")
    out = llm(system, user, ledger=ledger, run_id=run_id, purpose="followups")
    fms = []
    for f in out.get("formulations", [])[:5]:
        q = str(f.get("query", "")).strip()
        if q:
            fms.append(Formulation(
                formulation_id=sha16(f"fu|{q}")[:8], query=q,
                p_success=min(max(float(f.get("p_success", 0.4)), 0.05), 1.0),
                yield_if_success=max(float(f.get("yield_if_success", 1.0)), 0.1)))
    return fms


def add_if_novel(state: FrontierState, fm: Formulation) -> None:
    """Dedup by query text so near-duplicate proposals can't freeze stopping
    by inflating frontier credit with copies (webaggr.pdf §3.2 remark)."""
    if not any(f.query.lower() == fm.query.lower()
               for f in state.formulations.values()):
        state.formulations[fm.formulation_id] = fm


# ---------------------------------------------------------------------------
def run_query(player: str, run_id: str, ledger, *, search, gate,
              eps_g: float = config.EPS_G, delta_m: float = config.DELTA_M,
              eta: float = config.ETA, budget_usd: float = config.BUDGET_USD,
              max_steps: int = config.MAX_STEPS,
              llm=call_llm, fetch=real_fetch):
    """FrontierDiscovery (webaggr.pdf Alg. 1). Returns (state, artifacts).

    `search`, `gate` are required collaborators; `llm` and `fetch` are
    injectable for TESTS ONLY — experiment scripts pass the real ones.
    """
    stratum = canon.normalize_name(player)
    state = FrontierState()
    state.strata[stratum] = StratumState(name=stratum)
    ce = ClaimsEngine()
    mentions_by_record: dict[str, list] = defaultdict(list)
    sources: dict[str, object] = {}
    url_cache: dict[str, list] = {}          # url -> record keys it covered
    spent = 0.0
    # Default label if the loop runs out of occasions without any break:
    # tau_max reached is its own (honest) stop reason, not frontier exhaustion.
    stop_reason, stop_step = "max_steps", 0

    for fm in seed_formulations(player, llm=llm, ledger=ledger, run_id=run_id):
        add_if_novel(state, fm)

    for step in range(1, max_steps + 1):
        stop_step = step
        # 1. pick next search by reservation index (App. B); lambda drops to 0
        #    once every stratum is certified -> economic stop ends the run
        lam = config.LAMBDA_PER_RECORD if state.uncertified_strata() else 0.0
        cands = [f for f in state.formulations.values() if not f.issued]
        if not cands:
            stop_reason = "frontier_exhausted"
            break
        fm = max(cands, key=lambda x: x.reservation_index(lam))
        if fm.reservation_index(lam) <= 0:
            stop_reason = "economic"          # NOT a certificate (App. B)
            break

        # 2. issue the search = ONE capture occasion (Def. 4)
        state.T += 1
        new = 0
        new_keys_this_step: list[str] = []
        for r in search.search(fm.query, k=10):
            url = r["url"]
            if url in url_cache:
                # same page under a new formulation: a legitimate second
                # capture occasion for its records — credit, don't refetch
                for rk in url_cache[url]:
                    was_new = rk not in state.covered
                    state.covered[rk].add(fm.formulation_id)
                    if was_new and new < state.Y_CAP:
                        new += 1
                continue
            src = fetch(url, fm.formulation_id)
            ledger.log_fetch_outcome(run_id, step, url, src.domain, src.status,
                                     len(src.main_text))
            ledger.add_source(src)
            if src.status != "ok":            # denied/paywall/empty/error:
                url_cache[url] = []           # a CENSORED observation, logged,
                continue                      # never an evidence-of-absence vote
            ok, score = is_relevant_cached(src, player)
            if not ok:
                ledger.add_rejected(src.source_id, url, score,
                                    src.main_text[:2000])
                url_cache[url] = []
                continue
            from .extract import extract_certified   # local import avoids cycle
            ms, claims, n_rej = extract_certified(
                src, player, gate, ledger=ledger, run_id=run_id, llm=llm)
            if n_rej:                         # delta_E audit trail (§7.1):
                ledger.log_measurement(       # gate/validator rejections per
                    run_id, step, stratum,    # source, persisted not discarded
                    "gate_rejected", n_rej, extra={"source_id": src.source_id})
            sources[src.source_id] = src
            for c in claims:
                ce.ingest(c)
                ledger.add_claim(c)
            covered_keys = []
            for m in ms:
                ent = resolve_entity(m.entity_surface, player)
                if ent is None:
                    ledger.add_mention(m, accepted=False)  # contamination guard
                    continue
                ledger.add_mention(m, accepted=True)
                rk = m.record_key(ent)
                state.record_stratum[rk] = stratum
                mentions_by_record[rk].append(m)
                was_new = rk not in state.covered
                state.covered[rk].add(fm.formulation_id)   # OCCASIONS, not sources
                covered_keys.append(rk)
                if was_new:
                    ledger.log_coverage(run_id, step, rk, fm.formulation_id)
                    new_keys_this_step.append(rk)
                    if new < state.Y_CAP:     # per-occasion novelty cap (A1b)
                        new += 1
            url_cache[url] = covered_keys
        fm.issued, fm.realized_yield = True, new
        spent += fm.cost

        if new_keys_this_step:
            unissued = [f.query for f in state.formulations.values() if not f.issued]
            for nf in propose_followups(player, [k.split("|")[1] for k in
                                                 state.covered], unissued,
                                        llm=llm, ledger=ledger, run_id=run_id):
                add_if_novel(state, nf)

        # 3. claims engine: checksums, certification, gap-direction (§5.2)
        for g in state.uncertified_strata():
            n_g = state.N({g})
            asum = _assembled_sum(mentions_by_record, sources, g)
            st = ce.checksum(g, n_g, asum)
            state.strata[g].claimed_count = ce.claimed_count(g)
            if st.certified:
                state.strata[g].certified = "checksum"
                for f in state.formulations.values():   # prune: stratum leaves P
                    if f.stratum == g and not f.issued:
                        f.p_success = 0.0
            elif st.gap:
                seasons = {rk.split("|")[1] for rk in state.covered}
                for gf in ce.gap_formulations(g, player, st.gap, seasons):
                    add_if_novel(state, gf)

        # 4. per-stratum measurement spine — EVERYTHING replay needs (Exp 2)
        n_strata = max(len(state.strata), 1)
        for g, pool in state.stratum_pools():
            u = state.U_hat(pool)
            psi = state.psi(pool, delta_m, 1.0 / n_strata, max_steps)
            s = state.strata[g]
            ledger.log_measurement(run_id, step, g, "U_hat", u, extra={
                "N": state.N(pool), "f1": state.f(1, pool),
                "f2": state.f(2, pool), "psi": psi,
                "hot": bool(state.hot_frontier(g, eta)),
                "claimed_count": s.claimed_count,
                "certified": s.certified,       # "" not None (NaN-proof)
                "T": state.T, "new": new, "spent": round(spent, 4)})
        ledger.commit()

        # 5. stop test (eps read ONLY here -> looser-eps replay is valid)
        if spent >= budget_usd:
            stop_reason = "budget"
            break
        if all_strata_pass(state, eps_g, delta_m, eta, max_steps):
            stop_reason = "certified"
            break

    ledger.log_stop(run_id, stop_step, stop_reason,
                    extra={"eps_g": eps_g, "delta_m": delta_m, "spent": spent})
    ledger.commit()
    return state, dict(claims_engine=ce, mentions_by_record=mentions_by_record,
                       sources=sources, stop_reason=stop_reason,
                       stop_step=stop_step, spent=spent)


def is_relevant_cached(src, player):
    from .extract import is_relevant
    return is_relevant(src, player)


def _assembled_sum(mentions_by_record, sources, stratum) -> int:
    """Current best-effort SUM for the checksum comparison (webaggr.pdf §5.2)."""
    total = 0
    for rk, ms in mentions_by_record.items():
        if rk.split("|")[0] != stratum:
            continue
        total += corroborate_record(rk, rk.split("|")[1], ms, sources).value
    return total


# ---------------------------------------------------------------------------
def rekey_and_revalidate(state: FrontierState, ce: ClaimsEngine,
                         mentions_by_record, sources, ledger, run_id) -> list[str]:
    """Post-ER second checksum pass (impl guide §14.1, pitfall #3): counts are
    only meaningful once clusters settle, so a COUNT certificate granted on
    pre-ER counts is re-checked and REVOKED if the settled N disagrees.
    With deterministic ER the keys are canonical at ingestion, so live
    revocations should be rare — each one is logged (a write-up number)."""
    revoked = []
    for g in list(state.strata):
        s = state.strata[g]
        if s.certified != "checksum":
            continue
        n_g = state.N({g})
        st = ce.checksum(g, n_g, _assembled_sum(mentions_by_record, sources, g))
        if not st.certified:
            s.certified = ""                   # revoke ("" not None)
            revoked.append(g)
            ledger.log_measurement(run_id, -1, g, "certificate_revoked", 1.0,
                                   extra={"N_post_er": n_g})
    return revoked


def end_to_end(player: str, run_id: str, ledger, *, search, gate,
               llm=call_llm, fetch=real_fetch, **kw) -> Answer:
    """Full pipeline (impl guide §14.1): discovery -> ER (done at ingestion,
    deterministic) -> second checksum pass -> corroborate -> aggregate."""
    state, art = run_query(player, run_id, ledger, search=search, gate=gate,
                           llm=llm, fetch=fetch, **kw)
    ce, mbr, sources = (art["claims_engine"], art["mentions_by_record"],
                        art["sources"])
    rekey_and_revalidate(state, ce, mbr, sources, ledger, run_id)

    stratum = canon.normalize_name(player)
    resolved = [corroborate_record(rk, rk.split("|")[1], ms, sources)
                for rk, ms in sorted(mbr.items())]
    for r in resolved:
        ledger.add_resolved(run_id, r.record_key, r.season, r.value,
                            r.belief, r.n_domains)

    # Certificate labeling, reported SEPARATELY per path (pre-registration):
    # checksum-certified never counts toward the statistical C1/C2 headline.
    s = state.strata[stratum]
    eps_g = kw.get("eps_g", config.EPS_G)
    if s.certified == "checksum":
        cert, eps_c = "checksum", eps_g
    elif art["stop_reason"] == "certified":
        cert, eps_c = "statistical", eps_g
    else:
        # abandoned: report the honest level actually reached (U_hat + psi)
        n_strata = max(len(state.strata), 1)
        eps_c = min(1.0, state.U_hat({stratum}) +
                    state.psi({stratum}, kw.get("delta_m", config.DELTA_M),
                              1.0 / n_strata, kw.get("max_steps", config.MAX_STEPS)))
        cert = "abandoned"

    dplus = ce.checksum(stratum, state.N({stratum}),
                        sum(r.value for r in resolved)).delta_plus
    ans = aggregate_with_ci(stratum, resolved, eps_c=eps_c,
                            eps_f=config.EPS_F_CALIBRATED, certificate=cert,
                            stop_reason=art["stop_reason"], delta_plus=dplus)
    ledger.log_measurement(run_id, art["stop_step"], stratum, "final_total",
                           ans.total, extra={"certificate": cert,
                                             "eps_c": eps_c,
                                             "n_records": ans.n_records})
    ledger.commit()
    return ans
