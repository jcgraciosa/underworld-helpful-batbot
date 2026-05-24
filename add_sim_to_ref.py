#!/usr/bin/env python3
"""
add_sim_to_ref.py — Add sim_to_ref to existing eval JSON files without re-running the LLM.

Computes cosine similarity between each answer and its reference answer using
all-MiniLM-L6-v2 (symmetric similarity model). Updates the JSON files in place
and prints a per-file, per-class summary.

Usage:
    python3 add_sim_to_ref.py ablation_6_cell_chunking.json ablation_agent_6calls.json
    python3 add_sim_to_ref.py *.json          # all eval files
"""

import sys
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


def compute_sim_to_ref(answers, references, model):
    scores = []
    for ans, ref in zip(answers, references):
        if not ref:
            scores.append(None)
            continue
        embs = model.encode([ans, ref], normalize_embeddings=True)
        scores.append(float(np.dot(embs[0], embs[1])))
    return scores


def summarise(rows, label):
    scores = [r["sim_to_ref"] for r in rows if r.get("sim_to_ref") is not None]
    if not scores:
        print(f"  {label}: no references")
        return
    print(f"  {label} (n={len(scores)}): mean={sum(scores)/len(scores):.3f}  "
          f"min={min(scores):.3f}  max={max(scores):.3f}")


def process_file(path: Path, model):
    data = json.loads(path.read_text())

    # RAGAS renames "answer" → "response" and "contexts" → "retrieved_contexts" in output
    answers = [r.get("response", r.get("answer", "")) for r in data]
    references = [r.get("reference", "") for r in data]

    scores = compute_sim_to_ref(answers, references, model)
    for row, score in zip(data, scores):
        row["sim_to_ref"] = score

    path.write_text(json.dumps(data, indent=2))

    print(f"\n{path.name}:")
    summarise(data, "Overall")
    summarise([r for r in data if r.get("type") == "rag-specific"], "Class A")
    summarise([r for r in data if r.get("type") == "baseline"], "Class B")


def main():
    files = [Path(p) for p in sys.argv[1:]]
    if not files:
        print("Usage: python3 add_sim_to_ref.py <file1.json> [file2.json ...]")
        sys.exit(1)

    missing = [f for f in files if not f.exists()]
    if missing:
        print(f"Files not found: {', '.join(str(f) for f in missing)}")
        sys.exit(1)

    print(f"Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    for f in files:
        process_file(f, model)

    print("\nDone. Files updated in place.")


if __name__ == "__main__":
    main()
