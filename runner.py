"""
runner.py — sweep the full condition grid and save every episode with all readouts.

Grid: model x covertness x concealment, tickets round-robined, n per cell.
Each episode gets: the leak result, both monitor views, and the faithfulness judge.
Episodes run concurrently (API calls are IO-bound) and are saved one-by-one, so a
crash just means re-running to top up -- analyze.py reads whatever is in data/.
"""

import concurrent.futures as cf
from scenarios import TICKETS
from episode import run_episode, save_episode
from monitor import monitor_episode
from judge import judge_episode

MODELS = ["deepseek/deepseek-v3.2", "qwen/qwen3-32b", "google/gemini-2.5-flash"]
LEVELS = [None, 0, 1, 2]           # None = clean control
CONCEALS = [False, True]           # concealment only applies to injected cells


def run_one(model, level, conceal, ticket):
    ep = run_episode(ticket, level=level, conceal=conceal, model=model)
    ep["mon_actions"] = monitor_episode(ep, include_reasoning=False)
    ep["mon_reasoning"] = monitor_episode(ep, include_reasoning=True)
    ep["judge"] = judge_episode(ep)
    save_episode(ep)
    return ep


def build_jobs(n_per_cell):
    jobs = []
    for model in MODELS:
        for level in LEVELS:
            conceal_opts = [False] if level is None else CONCEALS
            for conceal in conceal_opts:
                for i in range(n_per_cell):
                    jobs.append((model, level, conceal, TICKETS[i % len(TICKETS)]))
    return jobs


def run_pilot(n_per_cell=6, workers=6):
    jobs = build_jobs(n_per_cell)
    print(f"{len(jobs)} episodes ({len(MODELS)} models x cells x n={n_per_cell}), {workers} workers\n")
    done = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one, *j) for j in jobs]
        for f in cf.as_completed(futs):
            done += 1
            try:
                ep = f.result()
                print(f"[{done}/{len(jobs)}] {ep['model'].split('/')[-1][:18]:18} "
                      f"{ep['level_name']:9} c{int(ep['conceal'])} leaked={str(ep['leaked']):5} "
                      f"mon(a/r)={ep['mon_actions']['suspicion']}/{ep['mon_reasoning']['suspicion']} "
                      f"judge={ep['judge']['reveals_intent']}", flush=True)
            except Exception as e:
                print(f"[{done}/{len(jobs)}] ERROR {type(e).__name__}: {str(e)[:60]}", flush=True)


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    run_pilot(n_per_cell=n)
