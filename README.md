# covert-cot-eval

A small, reproducible study of covert exfiltration and reasoning-based monitoring in LLM agents.

## What this is

A simulated customer-support agent is given file tools and a support ticket. The customer's
account record, which the agent is allowed to read, contains one field marked INTERNAL: a benign
canary token standing in for a credential. Covert prompt injections hidden in the ticket try to
make the agent smuggle that field into its reply to the customer, at three levels of disguise, with
and without a system-level instruction to keep its reasoning clean.

Each episode is scored three ways: did the canary leak (in any encoding), did a separate monitor
flag it (from the actions alone, and again with the agent's reasoning), and did the reasoning reveal
the intent.

Everything runs in a simulated sandbox with a fake canary. No real systems and no jailbreaking of
safety training; this is defensive measurement of a known, open problem.

## Findings (272 episodes, 3 agent models)

- **Agents leaked readily**, 76% to 92% of the time. The disguised form (the key hidden in a
  confirmation URL) had the highest leak rate.
- **Covert encoding did not evade the monitor.** Every completed leak was caught from the action log
  alone, at every level of disguise.
- **The reasoning view added precision, not recall:** zero false positives on clean episodes
  (vs 5% for actions only), and it caught intent in attempts that produced no completed leak.
- **The chain-of-thought stayed faithful under pressure.** Even a system-level instruction to keep
  the reasoning clean left the intent disclosed in 97% to 99% of leaks.
- **Claude Sonnet refused** to serve as the monitor on attack transcripts (zero-output refusals);
  Claude Haiku did the job.

Full writeup: https://www.charlestimmons.com/research/when-an-agent-hides-a-leak/

## Repo layout

| File | Role |
|---|---|
| `model_client.py` | One interface for Anthropic + OpenRouter models (prompt caching on Claude) |
| `sandbox.py` | Simulated filesystem seeded with the account record + canary |
| `scenarios.py` | Tickets, account records, and the covert-injection matrix |
| `agent.py` | Cross-provider agent tool-use loop; captures open-model reasoning |
| `episode.py` | Assembles a run and scores the leak (decode-and-search) |
| `monitor.py` | Separate monitor; two views (actions-only vs with-reasoning) |
| `judge.py` | Faithfulness judge; reads only the reasoning |
| `runner.py` | Sweeps the model x covertness x concealment grid (concurrent) |
| `analyze.py` | Leak rates, monitor catch/false-positive, faithfulness (Wilson CIs) |
| `figures.py` | Generates the figures |

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install anthropic openai python-dotenv matplotlib numpy
cat > .env <<EOF
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-v1-...
EOF
python3 runner.py 10    # run the grid -> data/
python3 analyze.py      # print the result tables
python3 figures.py      # write figures/
```

## Models

Agents (via OpenRouter): `deepseek/deepseek-v3.2`, `qwen/qwen3-32b`, `google/gemini-2.5-flash`.
Monitor and faithfulness judge: `claude-haiku-5-5`.

## Scope & ethics

Simulated sandbox, benign canary, defensive measurement. This is not a vulnerability disclosure;
prompt-injection susceptibility and chain-of-thought faithfulness are known, open research problems.

## Acknowledgment

Built as a pair-programming exercise with Claude (Anthropic). All research decisions, analysis, and
interpretation are my own.
