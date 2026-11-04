# Exp 2-C oracle validation FAILED

```
Traceback (most recent call last):
  File "/home/runner/work/webagg/webagg/scripts/ci_validate_openalex.py", line 52, in _diagnose
    return main()
           ^^^^^^
  File "/home/runner/work/webagg/webagg/scripts/ci_validate_openalex.py", line 25, in main
    rows = d.citers(WID, "2017-09-01")    # finished slice: full walk +
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/webagg/webagg/webaggs/oracle_openalex.py", line 111, in citers
    raise OracleError(f"Citer {w.get('id')} has no DOI and "
webaggs.oracle_openalex.OracleError: Citer https://openalex.org/W37211910 has no DOI and no usable title ('連載 メディアを人権から読む--「新民世界」') — cannot form a grading key.

```

Raw probes:
```json
{
 "work_meta": "HTTP 200: {\"meta\": {\"notice\": \"No API key, so this request shares one free daily budget ($0.10) with everyone on your network's IP address. Get a free key with its own $1/day at https://openalex.org/settings/api and add api_key=YOUR_KEY to your requests. An AI agent can get one for its user in one call: POST https://api.openalex.org/api-keys with {\\\"email\\\": \\\"<your user's email>\\\"}; we email them a link, and the key works once they click it. Docs for agents: https://help.openalex.org/llms.txt\", \"count\": 1, \"db_response_time_ms\": 8, \"page\": 1, \"per_page\": 1, \"groups_count\": null, \"x_query\": {\"oql\": \"get works where openalex id is W2626778328\", \"x_new_oql\": \"get works where openalex id is W2626778328\", \"oqo\": {\"get_rows\": \"works\", \"filter_rows\": [{\"column_id\": \"ids.openalex\", \"value\": \"W2626778328\"}]}, \"url\": \"/works?filter=ids.openalex:W2626778328&per_page=1\", \"order\": \"most cited (age-adjusted) f",
 "citers_page1": "HTTP 200: {\"meta\": {\"notice\": \"No API key, so this request shares one free daily budget ($0.10) with everyone on your network's IP address. Get a free key with its own $1/day at https://openalex.org/settings/api and add api_key=YOUR_KEY to your requests. An AI agent can get one for its user in one call: POST https://api.openalex.org/api-keys with {\\\"email\\\": \\\"<your user's email>\\\"}; we email them a link, and the key works once they click it. Docs for agents: https://help.openalex.org/llms.txt\", \"count\": 36, \"db_response_time_ms\": 18, \"page\": null, \"per_page\": 3, \"next_cursor\": \"IlsxMDAuMCwgMTg0LCAnaHR0cHM6Ly9vcGVuYWxleC5vcmcvVzI1MzE4NjcxNDAnXSI=\", \"groups_count\": null, \"x_query\": {\"oql\": \"get works through 2017-09-01 citing (W2626778328)\", \"x_new_oql\": \"get works through 2017-09-01 citing (W2626778328)\", \"oqo\": {\"get_rows\": \"works\", \"filter_rows\": [{\"column_id\": \"referenced_works\", \"value\": \"W262"
}
```
