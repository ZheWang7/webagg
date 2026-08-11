"""Sec. 15 / Sec. 6.3 -- the calibration-set harvester (seam A1), offline.

One test per duty (repo convention):
  test_surface_in_scope                 -> alias containment keeps the
                                           entity, keeps strangers out
  test_legit_amounts_from_oracle_db     -> only oracle amount mentions of
                                           THAT entity; deduped
  test_label_mention_three_way          -> exact / within-tolerance auto-
                                           labels; farther -> review queue;
                                           unparseable kept, not dropped
  test_check_split_leakage_loud         -> validation-half entity refused
  test_check_run_conditions             -> undenied refused (warn only with
                                           the explicit override); a
                                           contaminated artifact refused
                                           even when denial was claimed
  test_gather_refuses_gated_run         -> circularity guard: a mention
                                           accepted by a FITTED gate kills
                                           the harvest
  test_harvest_end_to_end_offline       -> fixture run + truth DB -> auto
                                           + queue buckets, idempotent
                                           appends, ch. 6 loader unchanged
  test_review_decisions_and_no_requeue  -> y/n judgments move queue rows
                                           into the file; decided mentions
                                           never reappear for review
  test_threshold_preview_semantics      -> all-correct set = calibrated
                                           accept-all; >delta_E wrong at
                                           low conf = a real threshold
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import pytest

from webagg import config
from webagg.calharvest import (append_calibration, append_queue,
                               check_run_conditions, check_split,
                               gather_mentions, harvest, label_mention,
                               legit_amounts, record_decision,
                               surface_in_scope, threshold_preview)
from webagg.calibration import load_calibration_set
from webagg.storage import (MeasurementRow, MentionRow, SourceRow,
                            get_session)


# --------------------------------------------------------------------------- #
# fixtures: tiny run DB + truth DB built row by row
# --------------------------------------------------------------------------- #

_NAME = "SPACE EXPLORATION TECHNOLOGIES CORP"


def _mention(mid, value, value_num, conf, *, surface="SpaceX",
             attr="amount", flags=("gate_uncalibrated",),
             extractor="A", accepted=True) -> MentionRow:
    return MentionRow(mention_id=mid, source_id="s1", entity_surface=surface,
                      record_kind="funding_round", attribute=attr,
                      value=value, value_num=value_num, self_conf=conf,
                      validator_flags=list(flags), accepted=accepted,
                      extractor_id=extractor, extracted_at=datetime.utcnow())


@pytest.fixture()
def truth_db(tmp_path):
    """Oracle DB: two filings of one round (D 40M, D/A 45M) + a stranger."""
    s = get_session(str(tmp_path / "truth.sqlite"))
    s.add(_mention("t1", "40000000", 40_000_000.0, 1.0, surface=_NAME,
                   flags=(), extractor="oracle"))
    s.add(_mention("t2", "45000000", 45_000_000.0, 1.0, surface=_NAME,
                   flags=(), extractor="oracle"))
    s.add(_mention("t3", "99000000", 99_000_000.0, 1.0,
                   surface="OTHER CORP", flags=(), extractor="oracle"))
    s.commit()
    yield s
    s.close()


@pytest.fixture()
def run_db(tmp_path):
    """A denied bootstrap-mode open-web run over the same entity."""
    s = get_session(str(tmp_path / "run.sqlite"))
    s.add(MeasurementRow(run_id="r1", step=0, metric="denylist_active",
                         value=1.0, extra={"suffixes": ["sec.gov"]}))
    # faithful read of the amended figure, spelled the press way
    s.add(_mention("m1", "$45M", 45_000_000.0, 0.95))
    # faithful read of the ORIGINAL figure (stale page) -> still correct
    s.add(_mention("m2", "40000000", 40_000_000.0, 0.90))
    # rounded press figure within CLAIM_TOL_REL of the filed 45M -> auto-tol
    s.add(_mention("m7", "45300000", 45_300_000.0, 0.88))
    # a wrong extraction, nearest to 45M
    s.add(_mention("m3", "46000000", 46_000_000.0, 0.40))
    # garbled: not a number at all -- must be KEPT (scores 2.0)
    s.add(_mention("m4", "forty-five-ish", None, 0.30))
    # another company's round on the same pages -> out of scope
    s.add(_mention("m5", "99000000", 99_000_000.0, 0.99, surface="Other Corp"))
    s.commit()
    yield s
    s.close()


_ALIASES = [_NAME, "SpaceX"]


# --------------------------------------------------------------------------- #
# 1. scoping
# --------------------------------------------------------------------------- #
def test_surface_in_scope():
    assert surface_in_scope("SpaceX", _ALIASES)
    assert surface_in_scope("Space Exploration Technologies", _ALIASES)
    assert surface_in_scope("SpaceX rocket division", _ALIASES)   # contains
    assert not surface_in_scope("Other Corp", _ALIASES)
    assert not surface_in_scope("", _ALIASES)


# --------------------------------------------------------------------------- #
# 2. the legitimate value set
# --------------------------------------------------------------------------- #
def test_legit_amounts_from_oracle_db(truth_db):
    legit = legit_amounts(truth_db, _NAME)
    assert sorted(n for _, n in legit) == [40_000_000.0, 45_000_000.0]
    # the stranger's 99M never enters this entity's label pool
    assert all(n != 99_000_000.0 for _, n in legit)


# --------------------------------------------------------------------------- #
# 3. labeling
# --------------------------------------------------------------------------- #
def test_label_mention_three_way():
    legit = [("40000000", 40_000_000.0), ("45000000", 45_000_000.0)]
    # canonical equality: "$45M" == "45000000" -> registry decides, exact
    true, src, rel = label_mention("$45M", 45_000_000.0, legit)
    assert (true, src, rel) == ("45000000", "registry_exact", 0.0)
    # within CLAIM_TOL_REL (0.67%): auto-correct, true := pred so the
    # gate's own equality test scores it correct
    true, src, rel = label_mention("45300000", 45_300_000.0, legit)
    assert src == "registry_tol" and true == "45300000"
    assert rel == pytest.approx(300_000 / 45_000_000)
    # beyond tolerance (2.2%): the registry cannot decide -> review, with
    # the nearest filed value as context
    true, src, rel = label_mention("46000000", 46_000_000.0, legit)
    assert src == "review" and true == "45000000"
    # unparseable: also review (a human should glance at garbled output)
    true, src, rel = label_mention("forty-five-ish", None, legit)
    assert src == "review" and rel == float("inf")


# --------------------------------------------------------------------------- #
# 4. split-leakage guard
# --------------------------------------------------------------------------- #
def test_check_split_leakage_loud():
    man = {"split": {"calibration": ["cikA"], "validation": ["cikB"]}}
    check_split(man, "cikA")                                  # passes
    with pytest.raises(RuntimeError, match="VALIDATION half"):
        check_split(man, "cikB")
    with pytest.raises(RuntimeError, match="not in the cohort"):
        check_split(man, "cikC")


# --------------------------------------------------------------------------- #
# 5. run-condition guard
# --------------------------------------------------------------------------- #
def test_check_run_conditions(tmp_path, run_db):
    check_run_conditions(run_db)                              # denied + clean

    # an UNDENIED run refuses by default, warns only on explicit override
    bare = get_session(str(tmp_path / "bare.sqlite"))
    with pytest.raises(RuntimeError, match="NOT a denied run"):
        check_run_conditions(bare)
    with pytest.warns(UserWarning, match="NOT a denied run"):
        check_run_conditions(bare, allow_undenied=True)
    bare.close()

    # denial CLAIMED but the artifact is contaminated -> refused anyway
    run_db.add(SourceRow(source_id="bad", url="https://www.sec.gov/x",
                         domain="www.sec.gov", fetch_time=datetime.utcnow()))
    run_db.commit()
    with pytest.raises(RuntimeError, match="withheld-registry violation"):
        check_run_conditions(run_db)


# --------------------------------------------------------------------------- #
# 6. circularity guard
# --------------------------------------------------------------------------- #
def test_gather_refuses_gated_run(run_db):
    kept, stats = gather_mentions(run_db, _ALIASES)           # bootstrap: fine
    assert len(kept) == 5 and stats["out_of_scope"] == 1

    run_db.add(_mention("m6", "45000000", 45_000_000.0, 0.99, flags=()))
    run_db.commit()
    with pytest.raises(RuntimeError, match="FITTED gate"):
        gather_mentions(run_db, _ALIASES)


# --------------------------------------------------------------------------- #
# 7. end-to-end harvest, file contract, idempotency
# --------------------------------------------------------------------------- #
def test_harvest_end_to_end_offline(tmp_path, run_db, truth_db):
    auto, review, stats = harvest(run_db, truth_db, run_id="r1",
                                  entity_id="cik1", entity_name=_NAME,
                                  aliases=_ALIASES)
    assert stats == {"amount_mentions": 6, "out_of_scope": 1,
                     "harvested": 5, "auto_exact": 2, "auto_tol": 1,
                     "queued_for_review": 2}
    by_id = {r["mention_id"]: r for r in auto + review}
    # the stale-but-faithful 40M read auto-labels CORRECT (reading fidelity)
    assert by_id["m2"]["label_source"] == "registry_exact"
    # the rounded 45.3M press figure auto-labels correct with true := pred
    assert by_id["m7"]["true"] == "45300000"
    # review rows carry NO 'true' yet, but do carry the reviewer's context
    assert "true" not in by_id["m3"] and by_id["m3"]["registry_nearest"] == "45000000"
    assert "passage" in by_id["m4"] and "url" in by_id["m4"]

    cal = tmp_path / "extraction_cal.json"
    q = tmp_path / "review_queue.json"
    assert append_calibration(cal, auto) == (3, 3)
    assert append_queue(q, cal, review) == (2, 2)
    # idempotent: the same harvest adds nothing to either file
    assert append_calibration(cal, auto) == (0, 3)
    assert append_queue(q, cal, review) == (0, 2)

    # the ch. 6 loader reads the DECIDED file unchanged; the gate fits
    loaded = load_calibration_set(cal)
    assert len(loaded) == 3 and all(len(t) == 3 for t in loaded)
    assert threshold_preview(cal)["n"] == 3


def test_review_decisions_and_no_requeue(tmp_path, run_db, truth_db):
    auto, review, _ = harvest(run_db, truth_db, run_id="r1",
                              entity_id="cik1", entity_name=_NAME,
                              aliases=_ALIASES)
    cal = tmp_path / "extraction_cal.json"
    q = tmp_path / "review_queue.json"
    append_calibration(cal, auto)
    append_queue(q, cal, review)

    # 'faithful' -> correct extraction: true := pred, human_faithful
    row = record_decision(cal, q, "m3", faithful=True)
    assert row["true"] == row["pred"] == "46000000"
    assert row["label_source"] == "human_faithful"
    # 'garbled' -> extraction error: true := nearest filed value
    row = record_decision(cal, q, "m4", faithful=False)
    assert row["true"] == "40000000" and row["label_source"] == "human_error"

    assert json.loads(q.read_text()) == []                # queue drained
    assert threshold_preview(cal)["n"] == 5               # all five decided
    # a DECIDED mention must never reappear for re-review: re-harvesting
    # the same run adds nothing to the queue
    assert append_queue(q, cal, review) == (0, 0)
    with pytest.raises(KeyError):
        record_decision(cal, q, "m3", faithful=True)      # already decided


# --------------------------------------------------------------------------- #
# 8. threshold semantics (what the receipt tells the user)
# --------------------------------------------------------------------------- #
def test_threshold_preview_semantics(tmp_path):
    # 100% correct -> every score is 1 - self_conf < 1... but the +1
    # finite-sample correction takes the MAX score at small n; still < 1,
    # so the gate accepts everything it did before -- now with a valid
    # delta_E claim behind it
    good = [{"pred": "40000000", "true": "40000000",
             "self_conf": 0.9, "mention_id": f"g{i}"} for i in range(30)]
    p1 = tmp_path / "all_good.json"
    p1.write_text(json.dumps(good))
    pv = threshold_preview(p1)
    assert pv["threshold"] < 1.0 and pv["scores_ge_1"] == 0

    # 20% wrong (>> delta_E = 0.05) -> the 95th-percentile score sits in
    # the wrong mass (>= 1), so t_hat >= 1: the gate's only lever at
    # deployment (1 - self_conf <= 1) cannot separate them -- preview says
    # accepts_everything, telling the user the label mix, not the gate,
    # is the binding constraint
    mixed = good[:24] + [{"pred": "99", "true": "40000000",
                          "self_conf": 0.2, "mention_id": f"b{i}"}
                         for i in range(6)]
    p2 = tmp_path / "mixed.json"
    p2.write_text(json.dumps(mixed))
    pv2 = threshold_preview(p2)
    assert pv2["scores_ge_1"] == 6 and pv2["accepts_everything"]
