import json

state = json.loads(open("out_eval_prod/run_state.json").read())
# Wait, run_state.json doesn't contain the list of frame IDs!
# It's generated from ingestion!
