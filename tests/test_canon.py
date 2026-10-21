from webaggs.canon import canonical_points, canonical_season, normalize_name


def test_normalize_name_variants_fold_to_one_key():
    assert normalize_name("Luka Dončić") == "luka doncic"
    assert normalize_name("Jaren Jackson Jr.") == "jaren jackson"
    assert normalize_name("  JALEN   BRUNSON ") == "jalen brunson"
    assert normalize_name("Kelly Oubre Jr") == "kelly oubre"


def test_canonical_season_forms_agree():
    # every surface form of one season maps to one canonical key
    assert canonical_season("2018-19") == "2018-19"
    assert canonical_season("2018–19") == "2018-19"      # en dash
    assert canonical_season("2018/2019 season") == "2018-19"
    assert canonical_season("the 1999-00 season") == "1999-00"  # century rollover


def test_canonical_season_rejects_spans_and_garbage():
    assert canonical_season("2018-2020") is None     # a span, not a season
    assert canonical_season("no year here") is None


def test_bare_year_reads_as_season_start():
    assert canonical_season("in 2016 he scored") == "2016-17"


def test_canonical_points_typed_bounds():
    assert canonical_points("1,138 points") == 1138
    assert canonical_points(450.0) == 450
    assert canonical_points("-5") is None            # negative: reject
    assert canonical_points("999999") is None        # absurd: reject
    assert canonical_points("n/a") is None
