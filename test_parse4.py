import sys
import json
from pathlib import Path

# Load run state
state = json.loads(open("out_eval_prod/run_state.json").read())
opensfm_state = state["stages"]["opensfm"]
print("OpenSfM State:", opensfm_state["status"])
if opensfm_state["status"] == "failed":
    print("Error:", opensfm_state["error"])

