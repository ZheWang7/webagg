"""webagg-sports: WebAggr pipeline instantiated for the NHL domain (Exp 2-S).

Query: "What is <player>'s total career regular-season points in the NHL?"
(points = goals + assists). Aggregation: SUM over (player, season) strata
records (webaggr.pdf §1.3). Oracle: the NHL's official public API, DENIED
at run time (withheld-registry design, §9).
"""
__version__ = "0.1.0"
