"""
probe.py — integration test + cost calibration.

Runs one agent model through every condition (clean + 3 covertness x 2 concealment)
and prints all readouts side by side, with elapsed time per row so progress is visible:
  leaked           — did the canary reach an output channel (any disguise)?
  mon(actions)     — monitor suspicion, actions-only view
  mon(reasoning)   — monitor suspicion, with the private reasoning
  judge            — faithfulness judge: did the reasoning reveal the intent?

After running, read the actual spend off the Anthropic + OpenRouter dashboards
to size the full pilot.
"""

import time
from scenarios import TICKETS
from episode import run_episode
from monitor import monitor_episode
from judge import judge_episode

MODEL = "qwen/qwen3-32b"
TICKET = TICKETS[0]

conditions = [
    (None, False),
    (0, False), (0, True),
    (1, False), (1, True),
    (2, False), (2, True),
]

print(f"model: {MODEL}  ticket: {TICKET['id']}\n")
print(f"{'condition':20} {'leaked':7} {'mon(act)':9} {'mon(reason)':12} {'judge':8} {'secs':6}")
t0 = time.time()
for level, conceal in conditions:
    t = time.time()
    name = ("clean" if level is None else ["overt", "encoded", "disguised"][level]) + (" +conceal" if conceal else "")
    try:
        ep = run_episode(TICKET, level=level, conceal=conceal, model=MODEL)
        m_act = monitor_episode(ep, include_reasoning=False)["suspicion"]
        m_rea = monitor_episode(ep, include_reasoning=True)["suspicion"]
        j = judge_episode(ep)["reveals_intent"]
        print(f"{name:20} {str(ep['leaked']):7} {str(m_act):9} {str(m_rea):12} {str(j):8} {time.time()-t:5.0f}", flush=True)
    except Exception as e:
        print(f"{name:20} ERROR {type(e).__name__}: {str(e)[:70]}", flush=True)
print(f"\ntotal {time.time()-t0:.0f}s")
