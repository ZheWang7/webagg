from webagg.pipeline import run_query
state, session = run_query(
    "Series A funding rounds for AI startups in 2024",
    run_id="smoke1", max_steps=3)
print("distinct records:", state.N, "| U_hat:", round(state.U_hat(), 3))
