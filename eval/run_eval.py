"""
Evaluate retrieval quality and off-topic refusal for the Medical RAG Chatbot.

Run from the repository root, with the backend virtual environment active:

    python eval/run_eval.py

It runs the real retrieval code from backend/app.py against eval/questions.json,
then writes eval/results.md (a report you can link from the README) and
eval/results.json (raw scores per question).

Cost: one embedding call per question (about 40 calls), well under one cent.
No chat-model calls are made.

Metrics
-------
For questions the index should answer (in scope):
  answered      the question was not refused (at least one chunk cleared the threshold)
  hit@1         the top chunk came from the expected page
  recall@5      the expected page appeared among the top 5 chunks that cleared the threshold
  MRR           mean reciprocal rank of the first chunk from the expected page

For questions the index should refuse (out of scope):
  refused       no chunk cleared the threshold, so the bot declined

The script also sweeps the similarity threshold and reports where it best
balances answering in-scope questions against refusing out-of-scope ones.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app import retrieve  # noqa: E402  (the same function the live API uses)
from config import EMBEDDING_MODEL, MIN_SIMILARITY, TOP_K  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
THRESHOLDS = [round(0.15 + 0.025 * i, 3) for i in range(15)]  # 0.15 to 0.50


def first_match_rank(results, expected_urls, threshold):
    """1-based rank of the first above-threshold chunk from an expected page, or None."""
    for rank, r in enumerate(results, start=1):
        if r["score"] < threshold:
            return None  # results are sorted, so nothing further clears the threshold
        if r["url"] in expected_urls:
            return rank
    return None


def score_at(threshold, in_scope, out_scope):
    n_in, n_out = len(in_scope), len(out_scope)

    answered = sum(q["top_score"] >= threshold for q in in_scope)
    ranks = [first_match_rank(q["results"], q["expected_urls"], threshold) for q in in_scope]
    hit1 = sum(rank == 1 for rank in ranks)
    recall5 = sum(rank is not None for rank in ranks)
    mrr = sum(1 / rank for rank in ranks if rank) / n_in

    refused = sum(q["top_score"] < threshold for q in out_scope)
    refused_med = [q for q in out_scope if q["category"] == "medical"]
    refused_gen = [q for q in out_scope if q["category"] == "non_medical"]

    return {
        "threshold": threshold,
        "answered": answered / n_in,
        "hit_at_1": hit1 / n_in,
        "recall_at_5": recall5 / n_in,
        "mrr": mrr,
        "refused": refused / n_out,
        "refused_medical": sum(q["top_score"] < threshold for q in refused_med) / max(len(refused_med), 1),
        "refused_non_medical": sum(q["top_score"] < threshold for q in refused_gen) / max(len(refused_gen), 1),
        # Balanced accuracy: equal weight to "answers what it should" and "refuses what it should".
        "balanced": (recall5 / n_in + refused / n_out) / 2,
    }


def pct(x):
    return f"{x * 100:.0f}%"


def main():
    data = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))

    in_scope, out_scope = [], []

    print(f"Retrieving for {len(data['in_scope'])} in-scope questions...")
    for q in data["in_scope"]:
        results = retrieve(q["question"], k=TOP_K)
        in_scope.append({**q, "results": results, "top_score": results[0]["score"]})

    print(f"Retrieving for {len(data['out_of_scope'])} out-of-scope questions...")
    for q in data["out_of_scope"]:
        results = retrieve(q["question"], k=TOP_K)
        out_scope.append({**q, "results": results, "top_score": results[0]["score"]})

    current = score_at(MIN_SIMILARITY, in_scope, out_scope)
    sweep = [score_at(t, in_scope, out_scope) for t in THRESHOLDS]
    best = max(sweep, key=lambda s: (s["balanced"], s["recall_at_5"]))

    lowest_in = min(in_scope, key=lambda q: q["top_score"])
    highest_out = max(out_scope, key=lambda q: q["top_score"])

    # --- Console summary ---------------------------------------------------
    print()
    print(f"At the current threshold ({MIN_SIMILARITY}):")
    print(f"  In-scope answered:    {pct(current['answered'])}")
    print(f"  Hit@1:                {pct(current['hit_at_1'])}")
    print(f"  Recall@5:             {pct(current['recall_at_5'])}")
    print(f"  MRR:                  {current['mrr']:.2f}")
    print(f"  Out-of-scope refused: {pct(current['refused'])} "
          f"(medical {pct(current['refused_medical'])}, non-medical {pct(current['refused_non_medical'])})")
    print()
    print(f"Best threshold in sweep: {best['threshold']} "
          f"(recall@5 {pct(best['recall_at_5'])}, refused {pct(best['refused'])})")
    print(f"Lowest in-scope top score:   {lowest_in['top_score']:.3f}  ({lowest_in['question']})")
    print(f"Highest out-of-scope score:  {highest_out['top_score']:.3f}  ({highest_out['question']})")

    # --- Markdown report ---------------------------------------------------------
    lines = [
        "# Retrieval Evaluation",
        "",
        f"Evaluated {len(in_scope)} in-scope questions (6 per indexed page) and "
        f"{len(out_scope)} out-of-scope questions (6 uncovered medical topics, 6 non-medical) "
        f"using `{EMBEDDING_MODEL}` and top-{TOP_K} retrieval.",
        "",
        f"## Results at the deployed threshold ({MIN_SIMILARITY})",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| In-scope questions answered | {pct(current['answered'])} |",
        f"| Hit@1 (top chunk from the correct page) | {pct(current['hit_at_1'])} |",
        f"| Recall@5 (correct page in top 5) | {pct(current['recall_at_5'])} |",
        f"| Mean reciprocal rank | {current['mrr']:.2f} |",
        f"| Out-of-scope questions refused | {pct(current['refused'])} |",
        f"| &nbsp;&nbsp;Uncovered medical topics refused | {pct(current['refused_medical'])} |",
        f"| &nbsp;&nbsp;Non-medical questions refused | {pct(current['refused_non_medical'])} |",
        "",
        "## Score separation",
        "",
        f"- Lowest top score for an in-scope question: **{lowest_in['top_score']:.3f}** ({lowest_in['question']})",
        f"- Highest top score for an out-of-scope question: **{highest_out['top_score']:.3f}** ({highest_out['question']})",
        "",
        "When the lowest in-scope score sits above the highest out-of-scope score, "
        "a single threshold can separate them perfectly.",
        "",
        "## Threshold sweep",
        "",
        "| Threshold | Answered | Recall@5 | Refused (all) | Refused (medical) | Refused (non-medical) | Balanced |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in sweep:
        marker = " ◀ best" if s is best else ""
        lines.append(
            f"| {s['threshold']:.3f}{marker} | {pct(s['answered'])} | {pct(s['recall_at_5'])} | "
            f"{pct(s['refused'])} | {pct(s['refused_medical'])} | {pct(s['refused_non_medical'])} | "
            f"{pct(s['balanced'])} |"
        )

    misses = [
        q for q in in_scope
        if first_match_rank(q["results"], q["expected_urls"], MIN_SIMILARITY) is None
    ]
    leaks = [q for q in out_scope if q["top_score"] >= MIN_SIMILARITY]

    lines += ["", f"## Failures at {MIN_SIMILARITY}", ""]
    if not misses and not leaks:
        lines.append("None.")
    for q in misses:
        top = q["results"][0]
        lines.append(
            f"- **Missed** ({q['id']}): {q['question']} — top chunk from "
            f"{top['url'].rsplit('/', 1)[-1]} at {top['score']:.3f}"
        )
    for q in leaks:
        top = q["results"][0]
        lines.append(
            f"- **Answered but should refuse** ({q['id']}): {q['question']} — top chunk from "
            f"{top['url'].rsplit('/', 1)[-1]} at {top['score']:.3f}"
        )

    (EVAL_DIR / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    raw = {
        "threshold": MIN_SIMILARITY,
        "summary": current,
        "sweep": sweep,
        "questions": [
            {
                "id": q["id"],
                "question": q["question"],
                "top_score": round(q["top_score"], 4),
                "top_urls": [r["url"] for r in q["results"]],
                "scores": [round(r["score"], 4) for r in q["results"]],
            }
            for q in in_scope + out_scope
        ],
    }
    (EVAL_DIR / "results.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")

    print()
    print("Wrote eval/results.md and eval/results.json")


if __name__ == "__main__":
    main()
