#!/usr/bin/env python3
"""
eval.py — Run RAGAS evaluation on HelpfulBatBot.

Calls bot internals directly (no HTTP server required).

Usage:
    python3 eval.py                           # uses eval_questions.yaml
    python3 eval.py my_questions.yaml         # custom question file
    python3 eval.py --output results.json     # save per-question results to JSON
    python3 eval.py --k 10                    # retrieve more chunks (default: 6)
    python3 eval.py --reranker                # use cross-encoder reranker
    python3 eval.py --reranker --n-candidates 20  # reranker with 20 candidates (default)

Retrieval config examples (testing sequence from notes.md):
    Step 1 — reranker only:
        python3 eval.py --reranker --output reranker.json
    Step 2 — embedding model upgrade (set env var, rebuild index first):
        BOT_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 BOT_FORCE_REBUILD=1 python3 eval.py --output bge.json
    Step 3 — both (if steps 1+2 each helped):
        BOT_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 python3 eval.py --reranker --output combined.json
    Step 5 — k tuning:
        python3 eval.py --k 3 --output k3.json
        python3 eval.py --k 10 --output k10.json

Question file format (YAML):
    - question: "How do I rebuild underworld3?"
      reference: "Run pixi run underworld-build."   # optional but enables more metrics

Requires (in addition to the bot's normal dependencies):
    pip install ragas datasets

RAGAS uses an LLM for scoring. Configure via environment variable:
    ANTHROPIC_API_KEY  — uses Claude (requires: pip install langchain-anthropic)
    OPENAI_API_KEY     — uses GPT-4o (RAGAS default)
"""

import sys
import json
import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# Import bot internals directly — no FastAPI server needed
from HelpfulBat_app import (
    ensure_index,
    retrieve,
    _agent_collect_chunks,
    call_llm_with_caching,
    build_system_prompt,
    format_context,
)


def load_questions(path: str) -> list:
    with open(path) as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        return data
    return data.get("questions", [])


def build_ragas_llm():
    """Configure RAGAS to use Claude if ANTHROPIC_API_KEY is set, otherwise use RAGAS default (OpenAI)."""
    import os
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from ragas.llms import LangchainLLMWrapper
            from langchain_anthropic import ChatAnthropic
            llm = LangchainLLMWrapper(ChatAnthropic(model="claude-sonnet-4-6"))
            print("RAGAS scorer: Claude (via langchain-anthropic)")
            return llm
        except ImportError:
            print("Note: langchain-anthropic not installed — install it to use Claude for scoring.")
            print("      Falling back to OpenAI (requires OPENAI_API_KEY).")
    print("RAGAS scorer: OpenAI default")
    return None


def build_ragas_embeddings():
    """Use local sentence-transformers embeddings so RAGAS doesn't require an OpenAI API key."""
    try:
        from ragas.embeddings import BaseRagasEmbeddings
        from sentence_transformers import SentenceTransformer

        class LocalEmbeddings(BaseRagasEmbeddings):
            def __init__(self, model_name: str):
                self._st = SentenceTransformer(model_name)
                self.model = model_name  # RAGAS expects this to be a string

            def embed_query(self, text: str) -> list:
                return self._st.encode(text, normalize_embeddings=True).tolist()

            def embed_documents(self, texts: list) -> list:
                return self._st.encode(texts, normalize_embeddings=True, batch_size=64).tolist()

            async def aembed_query(self, text: str) -> list:
                return self.embed_query(text)

            async def aembed_documents(self, texts: list) -> list:
                return self.embed_documents(texts)

        emb = LocalEmbeddings("all-MiniLM-L6-v2")
        print("RAGAS embeddings: local all-MiniLM-L6-v2")
        return emb
    except Exception as e:
        print(f"Warning: could not load local embeddings ({e}). RAGAS will fall back to OpenAI.")
        return None


def run_eval(questions_path: str, output_path: str | None, k: int = 6, use_reranker: bool = False, n_candidates: int = 20, use_hybrid: bool = False, use_hyde: bool = False, no_rag: bool = False, use_agent: bool = False):
    if not no_rag:
        print("Initialising index...")
        ensure_index()

    questions_data = load_questions(questions_path)
    if not questions_data:
        print(f"No questions found in {questions_path}")
        sys.exit(1)
    print(f"Loaded {len(questions_data)} questions from {questions_path}\n")

    rows = {"question": [], "answer": [], "contexts": [], "reference": []}
    types = []
    has_reference = False

    for i, item in enumerate(questions_data, 1):
        q = item["question"]
        ref = item.get("reference", "")
        types.append(item.get("type", "unclassified"))
        if ref:
            has_reference = True

        print(f"[{i}/{len(questions_data)}] {q[:80]}")

        if no_rag:
            ctx_docs = []
            context_text = ""
        elif use_agent:
            ctx_docs = _agent_collect_chunks(q, k=k, use_reranker=use_reranker, n_candidates=n_candidates, use_hybrid=use_hybrid, use_hyde=use_hyde)
            context_text = format_context(ctx_docs)
        else:
            ctx_docs = retrieve(q, k=k, use_reranker=use_reranker, n_candidates=n_candidates, use_hybrid=use_hybrid, use_hyde=use_hyde)
            context_text = format_context(ctx_docs)

        user_prompt = (
            f"Question: {q}\n\n"
            "Be concise — answer directly and completely, but avoid unnecessary explanation. "
            "Include code examples only when they add clarity. "
            "Include citations to specific files and line ranges."
        )
        answer = call_llm_with_caching(build_system_prompt(), user_prompt, context_text)

        rows["question"].append(q)
        rows["answer"].append(answer)
        rows["contexts"].append([doc.text for doc in ctx_docs])
        rows["reference"].append(ref)

    from ragas import evaluate
    from datasets import Dataset
    from ragas.metrics import AnswerRelevancy

    dataset = Dataset.from_dict(rows)

    ragas_llm = build_ragas_llm()
    ragas_emb = build_ragas_embeddings()

    if no_rag:
        # Without retrieved context, faithfulness and context metrics are not meaningful
        metrics = [AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb)]
        print("\nMetrics: answer_relevancy (faithfulness/context metrics require RAG)")
    else:
        from ragas.metrics import Faithfulness
        # ragas.metrics (classic API) works with LangchainLLMWrapper
        metrics = [Faithfulness(llm=ragas_llm), AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb)]

        if has_reference:
            from ragas.metrics import ContextPrecision, ContextRecall
            metrics += [ContextPrecision(llm=ragas_llm), ContextRecall(llm=ragas_llm)]
            print("\nMetrics: faithfulness, answer_relevancy, context_precision, context_recall")
        else:
            print("\nMetrics: faithfulness, answer_relevancy")
            print("(Add 'reference' to questions for context_precision and context_recall)")

    print("Running RAGAS evaluation...\n")
    result = evaluate(dataset, metrics=metrics)

    print("\n--- Results ---")
    print(result)

    # sim_to_ref: cosine similarity between answer and reference embeddings.
    # Uses all-MiniLM-L6-v2 (symmetric similarity model, not retrieval model)
    # so it's fair to compare across all configs regardless of retrieval budget.
    sim_scores = []
    if has_reference:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
            print("\nComputing sim_to_ref (cosine similarity to reference answer)...")
            st_model = SentenceTransformer("all-MiniLM-L6-v2")
            answers = rows["answer"]
            references = rows["reference"]
            for ans, ref in zip(answers, references):
                if not ref:
                    sim_scores.append(None)
                    continue
                embs = st_model.encode([ans, ref], normalize_embeddings=True)
                sim_scores.append(float(np.dot(embs[0], embs[1])))
            valid = [s for s in sim_scores if s is not None]
            print(f"sim_to_ref (mean over {len(valid)} questions with reference): {sum(valid)/len(valid):.3f}")
        except Exception as e:
            print(f"Warning: sim_to_ref skipped ({e})")
            sim_scores = [None] * len(rows["answer"])
    else:
        sim_scores = [None] * len(rows["answer"])

    if output_path:
        result_dict = result.to_pandas().to_dict(orient="records")
        for i, row in enumerate(result_dict):
            row["type"] = types[i]
            row["sim_to_ref"] = sim_scores[i]
        Path(output_path).write_text(json.dumps(result_dict, indent=2))
        print(f"\nSaved to {output_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HelpfulBatBot with RAGAS")
    parser.add_argument(
        "questions",
        nargs="?",
        default="eval_questions.yaml",
        help="YAML file of questions (default: eval_questions.yaml)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Save per-question results to a JSON file",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=6,
        help="Number of chunks to return per question (default: 6)",
    )
    parser.add_argument(
        "--reranker",
        action="store_true",
        help="Use cross-encoder reranker (cross-encoder/ms-marco-MiniLM-L-6-v2). "
             "Fetches --n-candidates chunks then reranks to top k.",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=20,
        dest="n_candidates",
        help="Candidates to fetch before reranking (only with --reranker, default: 20)",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Use hybrid BM25 + vector search with Reciprocal Rank Fusion (RRF).",
    )
    parser.add_argument(
        "--hyde",
        action="store_true",
        help="Use HyDE — embed a generated hypothetical answer instead of the raw question.",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        dest="no_rag",
        help="Skip retrieval entirely — call Claude directly with no context chunks. "
             "Use this to establish a plain-Claude baseline for comparison against the RAG pipeline. "
             "Only answer_relevancy is scored (faithfulness/context metrics require retrieved chunks).",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        dest="use_agent",
        help="Use tool-use agent RAG — Claude issues multiple search_docs calls to collect chunks, "
             "then generates the answer from all collected context.",
    )
    args = parser.parse_args()

    if args.no_rag:
        print("Mode: plain Claude (no RAG) — retrieval skipped")
    elif args.use_agent:
        print("Mode: agent RAG (tool-use) — Claude decides when/how to search")
        print(f"HyDE: {'ON' if args.hyde else 'OFF'}, max tool calls: 4")
    else:
        if args.reranker:
            print(f"Reranker: ON (fetching {args.n_candidates} candidates, reranking to top {args.k})")
        else:
            print(f"Reranker: OFF (retrieving top {args.k} directly)")
        print(f"Hybrid BM25+vector: {'ON' if args.hybrid else 'OFF'}")
        print(f"HyDE: {'ON' if args.hyde else 'OFF'}")

    run_eval(args.questions, args.output, k=args.k, use_reranker=args.reranker, n_candidates=args.n_candidates, use_hybrid=args.hybrid, use_hyde=args.hyde, no_rag=args.no_rag, use_agent=args.use_agent)
