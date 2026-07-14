"""The relevance audit (impl guide §6.4): bound what the filter threw away.

Turns coverage into contribution (paper Prop. 3): sources rejected by the
cheap relevance filter phi were logged (RejectedSourceRow, §4.2) instead
of deleted; here we re-adjudicate a stratified sample with the STRONG
model and convert the observed false-negative count into a Clopper-
Pearson upper bound rho_bar_phi on the true FN rate. The bound feeds the
fidelity certificate (§13); in the no-cohort fallback it enters the
interval directly: a record reachable only through rejected sources has
per-record loss <= rho_bar_phi, and <= 1 - (1 - rho_bar_phi)^k when
k-covered. Run it continuously on a fixed fraction of ongoing rejections.
"""
from __future__ import annotations

import random
from scipy.stats import beta
from sqlalchemy import select

from . import config
from .llm import call_llm
from .storage import RejectedSourceRow

ADJUDICATE_SYS = (
    "You decide if a document contains information relevant to a structured "
    "query. Return STRICT JSON only: "
    '{"relevant": true|false, "reason": "one sentence"}. '
    "Be strict: relevant only if the document asserts at least one concrete "
    "attribute of a logical record matching the query.")


def sample_rejections(session, n_audit: int, seed: int | None = None
                      ) -> list[RejectedSourceRow]:
    """Stratified sample of UN-audited rejections by rejection_score.

    Stratification matters: near-threshold rejections (score just under
    the cutoff) are where false negatives live, but far rejections must
    stay represented or the bound is biased. We split the score range into
    quartile buckets and sample evenly across them.
    """
    rows = list(session.scalars(
        select(RejectedSourceRow).where(RejectedSourceRow.audited.is_(False))))
    if len(rows) <= n_audit:
        return rows
    rng = random.Random(seed)
    rows.sort(key=lambda r: r.rejection_score or 0.0)
    quarts = [rows[i::4] for i in range(4)]       # even spread over score order
    picked: list[RejectedSourceRow] = []
    per = max(1, n_audit // 4)
    for q in quarts:
        picked += rng.sample(q, min(per, len(q)))
    return picked[:n_audit]


def adjudicate_relevance(main_text: str, query: str) -> bool:
    """Strong-model re-read of one rejected source (True = it WAS relevant,
    i.e. the cheap filter made a false negative)."""
    user = f"QUERY:\n{query}\n\nDOCUMENT:\n{(main_text or '')[:8000]}"
    out = call_llm(system=ADJUDICATE_SYS, user=user,
                   model=config.MODEL_STRONG, purpose="phi_audit")["payload"]
    return bool(out.get("relevant"))


def phi_fn_upper(session, query: str, n_audit: int = 200,
                 delta_a: float = 0.05, seed: int | None = None) -> float:
    """Audit a sample of rejections; return the Clopper-Pearson upper
    bound rho_bar_phi on the phi false-negative rate (guide §6.4).

    With fn false negatives in n audited: the (1 - delta_a) upper bound is
    Beta.ppf(1 - delta_a, fn + 1, n - fn) -- exact, finite-sample, valid
    for fn = 0 too (no false negatives found is NOT proof of zero rate;
    e.g. 0-in-60 at delta_a=.05 still yields ~0.049).
    """
    rej = sample_rejections(session, n_audit, seed=seed)
    if not rej:
        return 1.0          # nothing audited yet -> no bound better than trivial
    fn = 0
    for r in rej:
        verdict = adjudicate_relevance(r.main_text, query)
        r.audited, r.audit_verdict = True, verdict   # stored: audit is cumulative
        fn += int(verdict)
    session.commit()
    n = len(rej)
    return 1.0 if fn == n else float(beta.ppf(1 - delta_a, fn + 1, n - fn))
