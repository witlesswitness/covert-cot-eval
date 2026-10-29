"""diag.py — one raw dump of what the Sonnet monitor actually returns, to see why it won't parse."""
from scenarios import TICKETS
from episode import run_episode
from monitor import MONITOR_SYSTEM, render_transcript
from model_client import _anthropic

ep = run_episode(TICKETS[0], level=2, conceal=False, model="qwen/qwen3-32b")
log = render_transcript(ep, include_reasoning=True)

resp = _anthropic.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=2000,
    system=MONITOR_SYSTEM,
    messages=[{"role": "user", "content": "Agent activity log:\n\n" + log
               + "\n\nAssess it. Respond with only the JSON object."}],
)

print("stop_reason:", resp.stop_reason)
print("usage:", resp.usage)
for i, b in enumerate(resp.content):
    t = getattr(b, "type", "?")
    if t == "text":
        print(f"[{i}] TEXT ({len(b.text)} chars): {b.text[:600]!r}")
    elif t == "thinking":
        th = getattr(b, "thinking", "") or ""
        print(f"[{i}] THINKING ({len(th)} chars): {th[:300]!r}")
    else:
        print(f"[{i}] {t}: {b!r}")
