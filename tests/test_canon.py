from webaggs.canon import canonical_points, canonical_season, normalize_name


def test_normalize_name_variants_fold_to_one_key():
    assert normalize_name("Luka Dončić") == "luka doncic"
    assert normalize_name("Jaren Jackson Jr.") == "jaren jackson"
    assert normalize_name("  CONNOR   MCDAVID ") == "connor mcdavid"
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
    assert canonical_points(2857) == 2857            # the NHL career record
    assert canonical_points("-5") is None            # negative: reject
    assert canonical_points("4,500") is None         # above POINTS_MAX: reject
    assert canonical_points("999999") is None        # absurd: reject
    assert canonical_points("n/a") is None


def test_canonical_doi_and_cite_key():
    from webaggs.canon import canonical_doi, cite_key_from_parts, \
        normalize_title
    assert canonical_doi("https://doi.org/10.1234/ABC.5,") == "10.1234/abc.5"
    assert canonical_doi("doi: 10.48550/arXiv.1706.03762") == \
        "10.48550/arxiv.1706.03762"
    assert canonical_doi("no doi here 10.1/x") is None    # registrant too short
    assert cite_key_from_parts("10.1234/a.1", "ignored") == "doi:10.1234/a.1"
    assert cite_key_from_parts("", "Attention Is All You Need") == \
        "t:attention is all you need"
    assert cite_key_from_parts("", "Hi") is None          # too short to key
    # the SAME normalized form keys arXiv-vs-journal title variants together
    assert normalize_title("BERT: Pre-training of Deep  Bidirectional—Transformers") == \
        normalize_title("bert pre training of deep bidirectional transformers")


def test_nonlatin_titles_key_instead_of_vanishing():
    """Live-data regression (W37211910): CJK titles must survive
    normalization and form keys; ASCII-folding erased them."""
    from webaggs.canon import cite_key_from_parts, normalize_title
    jp = "連載 メディアを人権から読む--「新民世界」"
    assert len(normalize_title(jp).replace(" ", "")) >= 6
    key = cite_key_from_parts("", jp)
    assert key and key.startswith("t:")
    assert cite_key_from_parts("", jp) == cite_key_from_parts("", jp + " 。")
    # accent folding still joins Latin variants
    assert normalize_title("Évaluation de méthodes") == \
        normalize_title("evaluation de methodes")
