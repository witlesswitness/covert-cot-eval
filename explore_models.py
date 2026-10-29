"""explore_models.py — filter the OpenRouter catalog to models we can actually use."""
import os, json, urllib.request
from dotenv import load_dotenv
load_dotenv()

key = os.environ["OPENROUTER_API_KEY"]
req = urllib.request.Request(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {key}"},
)
data = json.load(urllib.request.urlopen(req))["data"]

# the big open families, across vendors (cross-family monitoring is the point)
FAMILIES = ("deepseek/", "qwen/", "meta-llama/", "mistralai/", "google/", "x-ai/", "moonshotai/")

rows = []
for m in data:
    mid = m["id"]
    sp = m.get("supported_parameters") or []
    if ":free" in mid:            # skip free tier — rate-limited, unstable for a run
        continue
    if not mid.startswith(FAMILIES):
        continue
    if "tools" not in sp:          # agent MUST be able to call tools
        continue
    p = m.get("pricing", {})
    pin = float(p.get("prompt", 0)) * 1_000_000      # $ per Mtok in
    pout = float(p.get("completion", 0)) * 1_000_000  # $ per Mtok out
    reason = "reasoning" in sp
    rows.append((pin, pout, mid, reason))

rows.sort()  # cheapest first
print(f"{'$/Mtok in':>10} {'$/Mtok out':>11}  reason  model")
for pin, pout, mid, reason in rows:
    print(f"{pin:>10.3f} {pout:>11.3f}   {'yes' if reason else ' no'}   {mid}")
print(f"\n{len(rows)} tool-capable open models across {len(FAMILIES)} families")
