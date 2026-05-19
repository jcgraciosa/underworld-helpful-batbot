#!/usr/bin/env python3
"""
compare_claude_vs_rag.py — Compare RAG bot answers vs Claude with no retrieval context.

Loads RAG answers from an existing eval.py output JSON, calls Claude directly on
the same questions with no retrieved context (training knowledge only), then compares
both sets of answers against the reference using semantic similarity.

Usage:
    python3 compare_claude_vs_rag.py --rag-results reranker_k10_full.json
    python3 compare_claude_vs_rag.py --rag-results reranker_k10_full.json --output comparison.json
    python3 compare_claude_vs_rag.py --rag-results reranker_k10_full.json --limit 10

Comparison metrics:
    - Semantic similarity to reference answer (cosine, via SentenceTransformer)
      for questions that have a 'reference' field.
    - Semantic similarity between the two answers (how different they are).
    - RAG RAGAS scores (faithfulness, answer_relevancy, etc.) from the input JSON.
"""

import json
import os
import argparse
from pathlib import Path

import numpy as np
import anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

CLAUDE_MODEL = "claude-sonnet-4-6"
SIM_MODEL = "all-MiniLM-L6-v2"  # same model used by RAGAS in eval.py

_st_model = None


def get_st_model() -> SentenceTransformer:
    global _st_model
    if _st_model is None:
        print(f"Loading sentence-transformer ({SIM_MODEL})...")
        _st_model = SentenceTransformer(SIM_MODEL)
    return _st_model


def semantic_sim(text1: str, text2: str) -> float:
    """Cosine similarity between two texts (normalized embeddings → dot product)."""
    if not text1 or not text2:
        return 0.0
    model = get_st_model()
    embs = model.encode([text1, text2], normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))


def call_claude_no_rag(question: str, client: anthropic.Anthropic) -> str:
    """Call Claude with no retrieved context — uses training knowledge only."""
    system_prompt = (
        "You are an expert assistant for Underworld3, a Python geodynamics modeling framework "
        "built on PETSc and petsc4py. Answer from your knowledge of Underworld3, PETSc, "
        "parallel computing, and finite element methods. "
        "Be concise and include code examples where applicable."
    )
    try:
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            temperature=0.2,
            system=system_prompt,
            messages=[{"role": "user", "content": question}],
        )
        return message.content[0].text
    except anthropic.APIError as e:
        return f"[API error: {e}]"
    except Exception as e:
        return f"[Error: {e}]"


def load_rag_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _sim_summary(label: str, entries: list[dict]) -> None:
    """Print a RAG vs Claude similarity block for a subset of entries."""
    with_ref = [e for e in entries if e.get("reference")]
    if not with_ref:
        return
    rag_avg = sum(e["rag_sim_to_ref"] for e in with_ref) / len(with_ref)
    claude_avg = sum(e["claude_sim_to_ref"] for e in with_ref) / len(with_ref)
    winner = "RAG" if rag_avg > claude_avg else "Claude (no-RAG)"
    print(f"\n{label} ({len(with_ref)} questions with references):")
    print(f"  RAG:    {rag_avg:.3f}   Claude: {claude_avg:.3f}   Winner: {winner} (+{abs(rag_avg - claude_avg):.3f})")
    print(f"  {'Question':<55} {'RAG':>6} {'Claude':>6} {'Delta':>6}")
    print(f"  {'-'*55} {'-'*6} {'-'*6} {'-'*6}")
    for e in with_ref:
        q = e["question"][:53] + ".." if len(e["question"]) > 55 else e["question"]
        rag_s = e["rag_sim_to_ref"]
        cl_s = e["claude_sim_to_ref"]
        delta = rag_s - cl_s
        sign = "+" if delta > 0 else ""
        print(f"  {q:<55} {rag_s:>6.3f} {cl_s:>6.3f} {sign}{delta:>5.3f}")


def print_summary(comparison: list[dict]) -> None:
    with_ref = [e for e in comparison if e.get("reference")]

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"Total questions: {len(comparison)}")
    print(f"Questions with reference answers: {len(with_ref)}")

    # --- Overall ---
    if with_ref:
        rag_avg = sum(e["rag_sim_to_ref"] for e in with_ref) / len(with_ref)
        claude_avg = sum(e["claude_sim_to_ref"] for e in with_ref) / len(with_ref)
        winner = "RAG" if rag_avg > claude_avg else "Claude (no-RAG)"
        print(f"\nSemantic similarity to reference answers (higher = closer to reference):")
        print(f"  RAG (best config):    {rag_avg:.3f}")
        print(f"  Claude (no-RAG):      {claude_avg:.3f}")
        print(f"  Winner:               {winner} (+{abs(rag_avg - claude_avg):.3f})")

    # --- By question type ---
    types_present = sorted({e.get("type", "unclassified") for e in comparison})
    if len(types_present) > 1 or (len(types_present) == 1 and types_present[0] != "unclassified"):
        print(f"\n--- Breakdown by question type ---")
        for qtype in types_present:
            group = [e for e in comparison if e.get("type") == qtype]
            _sim_summary(qtype, group)

    # --- Full per-question table ---
    if with_ref:
        print(f"\n--- Full per-question breakdown ---")
        _sim_summary("All questions", comparison)

    avg_cross = sum(e["rag_vs_claude_sim"] for e in comparison) / len(comparison)
    print(f"\nAvg similarity between RAG and Claude answers: {avg_cross:.3f}")
    print("(1.0 = identical answers, 0.0 = completely different)")

    # Show RAG RAGAS scores if available
    ragas_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    rag_scores = {}
    for k in ragas_keys:
        vals = [e["rag_ragas"].get(k) for e in comparison if e["rag_ragas"].get(k) is not None]
        if vals:
            rag_scores[k] = sum(vals) / len(vals)
    if rag_scores:
        print(f"\nRAG RAGAS scores (from input file):")
        for k, v in rag_scores.items():
            print(f"  {k:<25} {v:.3f}")
    print("=" * 70)


def run_comparison(rag_results_path: str, output_path: str | None, limit: int | None) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set in environment / .env")
        return

    client = anthropic.Anthropic(api_key=api_key)
    rag_results = load_rag_results(rag_results_path)

    if limit:
        rag_results = rag_results[:limit]

    print(f"Loaded {len(rag_results)} questions from {rag_results_path}")
    get_st_model()  # warm up

    ragas_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    comparison = []

    for i, row in enumerate(rag_results, 1):
        question = row["user_input"]
        rag_answer = row["response"]
        reference = row.get("reference") or ""

        print(f"[{i}/{len(rag_results)}] {question[:75]}")

        claude_answer = call_claude_no_rag(question, client)

        entry = {
            "question": question,
            "reference": reference,
            "type": row.get("type", "unclassified"),
            "rag_answer": rag_answer,
            "claude_answer": claude_answer,
            "rag_vs_claude_sim": semantic_sim(rag_answer, claude_answer),
            "rag_ragas": {k: row.get(k) for k in ragas_keys},
        }

        if reference:
            entry["rag_sim_to_ref"] = semantic_sim(rag_answer, reference)
            entry["claude_sim_to_ref"] = semantic_sim(claude_answer, reference)

        comparison.append(entry)

    print_summary(comparison)

    if output_path:
        Path(output_path).write_text(json.dumps(comparison, indent=2))
        print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare RAG bot answers vs no-context Claude on the same questions"
    )
    parser.add_argument(
        "--rag-results",
        required=True,
        metavar="FILE",
        help="Path to eval.py output JSON (e.g. reranker_k10_full.json)",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Save full per-question comparison to a JSON file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Only compare the first N questions (useful for quick tests)",
    )
    args = parser.parse_args()

    run_comparison(args.rag_results, args.output, args.limit)
