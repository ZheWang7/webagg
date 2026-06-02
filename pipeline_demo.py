import datetime
from collections import defaultdict

from sqlalchemy import select

from webagg.pipeline import run_query
from webagg.storage import MentionRow

# Fresh run_id each time -> a clean per-run database, so the records printed
# below are exactly what THIS run discovered (not leftovers from earlier runs).
run_id = "demo_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

state, session = run_query(
    "Series A funding rounds for AI startups in 2024",
    run_id=run_id, max_steps=3,
)

print(f"\nrun_id: {run_id}")
print(f"distinct records (state.N): {state.N} | U_hat: {state.U_hat():.3f}\n")

# Re-assemble records from the mentions table: group every extracted mention
# by (entity, record_kind), then collect the value(s) seen for each attribute.
mentions = session.scalars(select(MentionRow)).all()

records = defaultdict(lambda: defaultdict(set))
for m in mentions:
    records[(m.entity_surface, m.record_kind)][m.attribute].add(m.value)

print(f"{len(records)} records assembled from {len(mentions)} mentions")
print("=" * 70)

for (entity, kind), attrs in sorted(records.items()):
    n_sources = len(state.covered_records.get(f"{entity}|{kind}", set()))
    print(f"\n{entity}   [{kind}]   ({n_sources} source(s))")
    for attr in sorted(attrs):
        values = sorted(attrs[attr])
        if len(values) == 1:
            print(f"    {attr:<16} {values[0]}")
        else:
            # several distinct values = a conflict the corroboration layer resolves
            print(f"    {attr:<16} {' | '.join(values)}   <-- conflicting")

print("\n" + "=" * 70)
