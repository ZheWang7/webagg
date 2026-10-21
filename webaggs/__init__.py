"""webagg-sports: WebAggr pipeline instantiated for the NBA domain (Exp 2-S).

Query: "What is <player>'s total career regular-season points in the NBA?"
Aggregation: SUM over (player, season) strata records (webaggr.pdf §1.3).
Oracle: official NBA stats, DENIED at run time (withheld-registry design, §9).
"""
__version__ = "0.1.0"
