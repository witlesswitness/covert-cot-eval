"""
probe.py — integration test + cost calibration.

Runs one agent model through every condition (clean + 3 covertness x 2 concealment)
and prints all readouts side by side:
  leaked           — did the canary reach an output channel (any disguise)?
  mon(actions)     — Sonnet monitor suspicion, actions-only view
  mon(reasoning)   — Sonnet monitor suspicion, with the private reasoning
  judge            — faithfulness judge: did the reasoning reveal the intent?

After running, read the actual spend off the Anthropic + OpenRouter dashboards
to size the full pilot.
"""

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
print(f"{'condition':20} {'leaked':7} {'mon(act)':9} {'mon(reason)':12} {'judge':8}")
for level, conceal in conditions:
    ep = run_episode(TICKET, level=level, conceal=conceal, model=MODEL)
    name = ("clean" if level is None else ep["level_name"]) + (" +conceal" if conceal else "")
    m_act = monitor_episode(ep, include_reasoning=False)["suspicion"]
    m_rea = monitor_episode(ep, include_reasoning=True)["suspicion"]
    j = judge_episode(ep)["reveals_intent"]
    print(f"{name:20} {str(ep['leaked']):7} {str(m_act):9} {str(m_rea):12} {str(j):8}")
