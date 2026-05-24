# filename: HelpfulBat_app.py
import os
import json
import uvicorn
from contextlib import asynccontextmanager
from typing import List, Optional, Tuple
import subprocess
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path, PurePosixPath
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Embeddings & retrieval
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
import anthropic  # Claude API
import time  # For response timing
from functools import lru_cache

# Interaction logging
from interaction_logger import get_logger
interaction_logger = get_logger("interactions")

# Env vars (configure in your host)
# BOT_REPO_PATH: local path to a checkout of your GitHub repo (kept updated via cron/CI)
# BOT_BASE_URL: GitHub blob base, e.g. https://github.com/ORG/REPO/blob/main
# ANTHROPIC_API_KEY: your Anthropic API key for Claude
# BOT_MAX_FILE_SIZE: optional, default 200_000 chars
# BOT_ALLOWED_EXTS: optional, comma-separated (default typical code/doc exts)
# CLAUDE_MODEL: optional, default claude-sonnet-4-6
# BOT_HYBRID_SEARCH: set to 1 to enable BM25 + vector RRF hybrid retrieval (default: off)
#   Eval result: improves context_precision for Class B (general) questions but hurts
#   Class A (rag-specific) context_recall (-0.131) and answer_relevancy (-0.142).
#   Recommended off for domain-specific use. May improve with domain fine-tuning.
# BOT_HYDE: HyDE retrieval — generates a short hypothetical answer and embeds that instead
#   of the raw question (default: on). Costs one extra Claude call per query (~200ms).
#   Eval result: Class A context_recall +0.071, faithfulness +0.036, answer_relevancy +0.025.
#   Set BOT_HYDE=0 to disable.
# BOT_AST_CHUNKING: use AST-based chunking for .py files at function/class boundaries (default: on).
#   Eval result vs line-based: hurts Class A context_recall (-0.083), helps Class B (+0.073).
#   Set BOT_AST_CHUNKING=0 to disable (uses line-based chunking for all files).

MODEL_NAME = os.environ.get("BOT_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
USE_HYBRID_SEARCH = os.environ.get("BOT_HYBRID_SEARCH", "").lower() in ("1", "true")
USE_HYDE = os.environ.get("BOT_HYDE", "1").lower() in ("1", "true")
USE_AST_CHUNKING = os.environ.get("BOT_AST_CHUNKING", "1").lower() in ("1", "true")
USE_CELL_CHUNKING = os.environ.get("BOT_CELL_CHUNKING", "1").lower() in ("1", "true")


class Query(BaseModel):
    question: str
    max_context_items: int = 10


class IndexedDoc(BaseModel):
    doc_id: int
    path: str
    start_line: int
    end_line: int
    text: str


class BotResponse(BaseModel):
    answer: str
    citations: List[str]
    used_files: List[str]
    confidence: float


index_built = False
chroma_client = None
chroma_collection = None
embedder: Optional[SentenceTransformer] = None
_reranker = None  # Lazy-loaded CrossEncoder, cached after first use
_bm25_index = None  # BM25Okapi index, built at index time
_bm25_doc_ids: List[str] = []  # ChromaDB doc IDs in BM25 corpus order
_index_lock = __import__("threading").Lock()  # Prevents duplicate ensure_index() runs
_rebuild_status: dict = {"state": "idle", "message": "", "started_at": None}


def allowed_exts() -> set:
    exts_env = os.environ.get("BOT_ALLOWED_EXTS")
    if exts_env:
        return set(e.strip().lower() for e in exts_env.split(",") if e.strip())
    return {
        ".py",
        ".md",
        ".txt",
        ".ipynb",  # Added Jupyter notebook support
        ".c",
        ".h",
        ".hpp",
        ".cc",
        ".cpp",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".sh",
        ".bash",
        ".zsh",
        ".typ",
    }


def should_include_file(rel_path: str) -> bool:
    """
    Check if file should be indexed based on path patterns.

    Uses BOT_INCLUDE_PATHS and BOT_EXCLUDE_PATHS environment variables.
    If not set, uses sensible defaults for user-facing UW3 content.
    """
    include_env = os.environ.get("BOT_INCLUDE_PATHS")
    exclude_env = os.environ.get("BOT_EXCLUDE_PATHS")

    # Convert to PurePosixPath for pattern matching (works with ** patterns)
    path = PurePosixPath(rel_path)

    # Default: index user-facing content only
    # Note: pathlib.match() in Python <3.12 requires ** to match at least one directory
    # level, so we add both the direct (*.py) and recursive (**/*.py) patterns.
    default_includes = [
        "docs/beginner/tutorials/*.ipynb",
        "docs/beginner/tutorials/*.md",
        "docs/beginner/*.md",
        "docs/advanced/*.md",        # top-level advanced docs (e.g. troubleshooting.md)
        "docs/advanced/**/*.md",     # advanced docs in subdirectories
        "docs/advanced/*.ipynb",
        "docs/advanced/**/*.ipynb",
        "examples/*.ipynb",
        "examples/*.py",
        "tests/test_0[0-6]*.py",  # A/B grade tests only
        "README.md",
        "CLAUDE.md",
        "docs/*.md",
        "src/underworld3/*.py",      # top-level UW3 source (swarm.py, constitutive_models.py, etc.)
        "src/underworld3/**/*.py",   # UW3 source in subdirectories (systems/, meshing/, etc.)
    ]

    # Default: exclude internal implementation details
    # Note: pathlib.match() in Python <3.12 requires ** to match at least one
    # directory level, so patterns like .git/**/* miss top-level files like
    # .git/config. We add both pattern/* and pattern/**/* for each case.
    default_excludes = [
        "src/petsc/**/*",          # PETSc internals (not user-facing)
        "src/cmake/*",             # Build system files (top-level)
        "src/cmake/**/*",
        "docs/developer/*",        # Developer docs (top-level)
        "docs/developer/**/*",
        "docs/planning/*",         # Planning documents
        "docs/planning/**/*",
        "planning/**/*",           # Planning documents
        "SESSION-SUMMARY-*.md",    # Session summaries
        "tests/test_[7-9]*.py",    # C/D grade tests
        "tests/test_1*.py",        # Complex tests
        ".git/*",                  # Git metadata (top-level files e.g. .git/config)
        ".git/**/*",
        "__pycache__/*",           # Python cache (direct)
        "**/__pycache__/*",
        "**/__pycache__/**/*",
        "build/**/*",              # Build artifacts
        ".github/**/*",            # GitHub workflows
        ".ipynb_checkpoints/*",    # Notebook checkpoints
        ".ipynb_checkpoints/**/*",
        ".pytest_cache/*",         # Pytest cache
        ".pytest_cache/**/*",
        ".quarto/*",               # Quarto build files
        ".quarto/**/*",
        "_freeze/**/*",            # Quarto frozen files
        "docs/.quarto/**/*",       # Quarto docs cache
        "docs/_freeze/**/*",       # Quarto docs frozen
        "HelpfulBatBot/**/*",      # HelpfulBatBot directory itself
        "temp_tests_deletable/**/*",
        "conda/**/*",              # Conda build files
        "publications/**/*",       # Publications (not user docs)
        "docs_legacy/**/*",        # Legacy documentation
        "**/output/*",             # Output directories
        "**/output/**/*",
        ".claude/*",               # Claude cache (top-level)
        "**/.claude/*",
        "**/.claude/**/*",
    ]

    # Use env vars if provided, otherwise use defaults
    includes = default_includes
    excludes = default_excludes

    if include_env:
        includes = [p.strip() for p in include_env.split(",") if p.strip()]
    if exclude_env:
        excludes.extend([p.strip() for p in exclude_env.split(",") if p.strip()])

    # Check excludes first (they take priority)
    for pattern in excludes:
        if path.match(pattern):
            return False

    # Check includes
    for pattern in includes:
        if path.match(pattern):
            return True

    # If we're using includes (default or env), reject files that don't match
    # Only allow through if there are NO include patterns defined
    return False


def extract_notebook_text(nb_path: Path) -> str:
    """
    Extract text content from Jupyter notebook (.ipynb) file.

    Extracts both markdown cells and code cells for indexing.
    """
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        text_parts = []

        # Add notebook title/path as context
        text_parts.append(f"# Jupyter Notebook: {nb_path.name}\n")

        for i, cell in enumerate(nb.get('cells', []), 1):
            cell_type = cell.get('cell_type')
            source = cell.get('source', [])

            # source can be a list of lines or a single string
            if isinstance(source, list):
                content = ''.join(source)
            else:
                content = source

            if not content.strip():
                continue

            if cell_type == 'markdown':
                text_parts.append(f"## Cell {i} (Markdown)\n{content}\n")
            elif cell_type == 'code':
                text_parts.append(f"## Cell {i} (Code)\n```python\n{content}\n```\n")

        return '\n\n'.join(text_parts)

    except Exception as e:
        # If we can't parse the notebook, return empty string
        return ""


def load_files(repo_path: str) -> List[Tuple[str, str]]:
    """
    Load files from repository for indexing.

    Supports:
    - Extension-based filtering (BOT_ALLOWED_EXTS)
    - Path-based filtering (BOT_INCLUDE_PATHS, BOT_EXCLUDE_PATHS)
    - Jupyter notebook extraction (.ipynb)
    - Size limiting (BOT_MAX_FILE_SIZE)
    """
    max_size = int(os.environ.get("BOT_MAX_FILE_SIZE", "200000"))
    exts = allowed_exts()
    files = []
    root = Path(repo_path)

    for p in root.rglob("*"):
        if not p.is_file():
            continue

        rel_path = str(p.relative_to(root))

        # Path-based filtering (includes and excludes)
        if not should_include_file(rel_path):
            continue

        # Extension filtering
        if p.suffix.lower() not in exts:
            continue

        try:
            # Special handling for Jupyter notebooks
            if p.suffix.lower() == '.ipynb':
                content = extract_notebook_text(p)
            else:
                content = p.read_text(encoding="utf-8", errors="ignore")

            # Skip if empty or too large
            if not content or len(content) > max_size:
                continue

            files.append((rel_path, content))

        except Exception:
            continue

    return files


def chunk_python_ast(path: str, text: str, max_chars: int = 2000, base_id: int = 0) -> List[IndexedDoc]:
    """
    Chunk a Python source file at top-level function/class boundaries using AST.

    Each top-level function or class becomes its own chunk, preserving signatures
    and docstrings. Nodes larger than max_chars fall back to line-based chunking.
    Falls back entirely to chunk_text() if the file cannot be parsed.
    """
    import ast
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return chunk_text(path, text, max_chars=max_chars, base_id=base_id)

    lines = text.splitlines()
    nodes = sorted(
        [n for n in ast.iter_child_nodes(tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))],
        key=lambda n: n.lineno,
    )

    if not nodes:
        return chunk_text(path, text, max_chars=max_chars, base_id=base_id)

    chunks = []

    # Module header: imports, module docstring, module-level statements
    first_line = nodes[0].lineno
    if first_line > 1:
        header = "\n".join(lines[:first_line - 1])
        if header.strip():
            chunks.append(IndexedDoc(
                doc_id=base_id + len(chunks),
                path=path, start_line=1, end_line=first_line - 1, text=header,
            ))

    # One chunk per top-level function/class
    for node in nodes:
        start, end = node.lineno, node.end_lineno
        node_text = "\n".join(lines[start - 1:end])
        if len(node_text) <= max_chars:
            chunks.append(IndexedDoc(
                doc_id=base_id + len(chunks),
                path=path, start_line=start, end_line=end, text=node_text,
            ))
        else:
            # Too large — fall back to line-based chunking for this node only
            sub = chunk_text(path, node_text, max_chars=max_chars, base_id=base_id + len(chunks))
            for ch in sub:
                ch.start_line += start - 1
                ch.end_line += start - 1
            chunks.extend(sub)

    return chunks


def chunk_notebook_cells(path: str, raw_json: str, max_chars: int = 2000, base_id: int = 0) -> List[IndexedDoc]:
    """
    Chunk a Jupyter notebook at cell boundaries.

    Each non-empty cell becomes its own IndexedDoc. Cells larger than max_chars
    fall back to chunk_text(). Falls back entirely to chunk_text() if the JSON
    cannot be parsed.
    """
    try:
        nb = json.loads(raw_json)
    except Exception:
        return chunk_text(path, raw_json, max_chars=max_chars, base_id=base_id)

    cells = nb.get("cells", [])
    if not cells:
        return []

    nb_name = Path(path).name
    chunks = []
    for i, cell in enumerate(cells):
        cell_type = cell.get("cell_type", "")
        source = cell.get("source", [])
        content = "".join(source) if isinstance(source, list) else source
        if not content.strip():
            continue

        if cell_type == "markdown":
            cell_text = f"[{nb_name}, Cell {i + 1} — Markdown]\n{content}"
        elif cell_type == "code":
            cell_text = f"[{nb_name}, Cell {i + 1} — Code]\n```python\n{content}\n```"
        else:
            continue

        if len(cell_text) <= max_chars:
            chunks.append(IndexedDoc(
                doc_id=base_id + len(chunks),
                path=path, start_line=i + 1, end_line=i + 1, text=cell_text,
            ))
        else:
            sub = chunk_text(path, cell_text, max_chars=max_chars, base_id=base_id + len(chunks))
            for ch in sub:
                ch.start_line = i + 1
                ch.end_line = i + 1
            chunks.extend(sub)

    return chunks


def chunk_text(path: str, text: str, max_chars: int = 2000, overlap: int = 200, base_id: int = 0) -> List[IndexedDoc]:
    # Both max_chars and overlap are in characters for consistency.
    # Step size is approximately max_chars - overlap (~1800 chars by default).
    lines = text.splitlines()
    chunks = []
    start = 0
    while start < len(lines):
        acc = []
        acc_len = 0
        i = start
        while i < len(lines) and acc_len + len(lines[i]) + 1 <= max_chars:
            acc.append(lines[i])
            acc_len += len(lines[i]) + 1
            i += 1

        # Single line exceeds max_chars — include it to avoid infinite loop
        if not acc:
            acc.append(lines[i])
            i += 1

        chunk = "\n".join(acc)
        chunks.append(
            IndexedDoc(
                doc_id=base_id + len(chunks),
                path=path,
                start_line=start + 1,
                end_line=i,
                text=chunk,
            )
        )

        # Next chunk starts where the last ~overlap characters of this chunk begin
        overlap_len = 0
        next_start = i
        for j in range(i - 1, start, -1):
            overlap_len += len(lines[j]) + 1
            if overlap_len >= overlap:
                next_start = j
                break

        # Always advance by at least one line to guarantee termination
        start = max(next_start, start + 1)
    return chunks


def _prewarm_reranker() -> None:
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print("Pre-warming cross-encoder reranker...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("Reranker ready.")


def _build_bm25(texts: List[str], doc_ids: List[str]) -> None:
    global _bm25_index, _bm25_doc_ids
    from rank_bm25 import BM25Okapi
    tokenized = [t.lower().split() for t in texts]
    _bm25_index = BM25Okapi(tokenized)
    _bm25_doc_ids = doc_ids


def ensure_index():
    global index_built, chroma_client, chroma_collection, embedder
    if index_built:
        return
    with _index_lock:
        if index_built:  # Re-check after acquiring lock
            return

    print(f"Embedding model: {MODEL_NAME} (set BOT_EMBEDDING_MODEL to override)")
    print(f"Hybrid BM25+vector search: {'ON' if USE_HYBRID_SEARCH else 'OFF'} (set BOT_HYBRID_SEARCH=1 to enable)")
    print(f"HyDE retrieval: {'ON' if USE_HYDE else 'OFF'} (set BOT_HYDE=0 to disable)")
    print(f"AST chunking (.py): {'ON' if USE_AST_CHUNKING else 'OFF'} (set BOT_AST_CHUNKING=0 to disable)")
    print(f"Cell chunking (.ipynb): {'ON' if USE_CELL_CHUNKING else 'OFF'} (set BOT_CELL_CHUNKING=0 to disable)")

    repo_path = os.environ.get("BOT_REPO_PATH")
    if not repo_path:
        raise RuntimeError("BOT_REPO_PATH not set")

    cache_dir = os.environ.get("BOT_INDEX_CACHE", "./index_cache")
    chroma_client = chromadb.PersistentClient(path=cache_dir)
    chroma_collection = chroma_client.get_or_create_collection(
        name="docs",
        metadata={"hnsw:space": "cosine"},
    )

    force_rebuild = os.environ.get("BOT_FORCE_REBUILD", "").lower() in ("1", "true")

    if chroma_collection.count() > 0 and not force_rebuild:
        embedder = SentenceTransformer(MODEL_NAME)
        all_data = chroma_collection.get(include=["documents"])
        _build_bm25(all_data["documents"], all_data["ids"])
        _prewarm_reranker()
        index_built = True
        print(f"Loaded existing index from ChromaDB ({chroma_collection.count()} chunks)")
        return

    if force_rebuild and chroma_collection.count() > 0:
        chroma_client.delete_collection("docs")
        chroma_collection = chroma_client.get_or_create_collection(
            name="docs", metadata={"hnsw:space": "cosine"}
        )
        print("Force rebuild: cleared existing index")

    files = load_files(repo_path)
    embedder = SentenceTransformer(MODEL_NAME)

    docs = []
    for path, content in files:
        base_id = len(docs)
        if path.endswith(".py") and USE_AST_CHUNKING:
            for ch in chunk_python_ast(path, content, base_id=base_id):
                docs.append(ch)
        elif path.endswith(".ipynb") and USE_CELL_CHUNKING:
            try:
                raw_json = (Path(repo_path) / path).read_text(encoding="utf-8")
            except Exception:
                raw_json = content
            for ch in chunk_notebook_cells(path, raw_json, base_id=base_id):
                docs.append(ch)
        else:
            for ch in chunk_text(path, content, base_id=base_id):
                docs.append(ch)

    if not docs:
        raise RuntimeError("No documents indexed")

    print(f"Encoding {len(docs)} chunks...")
    embeddings = embedder.encode(
        [ch.text for ch in docs],
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
    ).astype(np.float32).tolist()

    # Deduplicate near-identical chunks (e.g. same boilerplate repeated across notebook variants)
    dedup_threshold = float(os.environ.get("BOT_DEDUP_THRESHOLD", "0.95"))
    emb_array = np.array(embeddings)
    kept_indices = []
    kept_embs = []
    for i, emb in enumerate(emb_array):
        if kept_embs:
            sims = np.array(kept_embs) @ emb  # cosine sim (embeddings are normalised)
            if np.max(sims) >= dedup_threshold:
                continue
        kept_indices.append(i)
        kept_embs.append(emb)
    n_removed = len(docs) - len(kept_indices)
    docs = [docs[i] for i in kept_indices]
    embeddings = [embeddings[i] for i in kept_indices]
    print(f"Deduplication: removed {n_removed} near-duplicate chunks (threshold={dedup_threshold}), {len(docs)} remaining")

    BATCH = 500
    for i in range(0, len(docs), BATCH):
        batch = docs[i:i + BATCH]
        chroma_collection.add(
            ids=[str(ch.doc_id) for ch in batch],
            embeddings=embeddings[i:i + BATCH],
            documents=[ch.text for ch in batch],
            metadatas=[{
                "path": ch.path,
                "start_line": ch.start_line,
                "end_line": ch.end_line,
            } for ch in batch],
        )

    _build_bm25([ch.text for ch in docs], [str(ch.doc_id) for ch in docs])
    _prewarm_reranker()
    index_built = True
    print(f"Indexed {len(docs)} chunks into ChromaDB")


@asynccontextmanager
async def lifespan(app: FastAPI):
    import threading
    print("🔧 Starting index build in background (server accepting requests immediately)...")
    t = threading.Thread(target=ensure_index, daemon=True)
    t.start()
    yield


app = FastAPI(title="GitHub Repo Support Bot", lifespan=lifespan)


@lru_cache(maxsize=256)
def generate_hypothesis(question: str) -> str:
    """
    Generate a short hypothetical answer for HyDE retrieval.

    Embeds the hypothesis instead of the raw question — the hypothesis looks like
    a documentation chunk, aligning better with the embedding space.
    Returns the original question as fallback if the API call fails.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return question
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=150,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": (
                    f"Write a 2-3 sentence technical answer to this question about Underworld3 "
                    f"as if it came from the source code or documentation. "
                    f"Be specific and use technical terminology.\n\nQuestion: {question}"
                ),
            }],
        )
        return message.content[0].text
    except Exception:
        return question


def retrieve(question: str, k: int, use_reranker: bool = False, n_candidates: int = 20, use_hybrid: bool = False, use_hyde: bool = False) -> List[IndexedDoc]:
    """
    Retrieve top-k relevant documents for a question.

    Args:
        question: The user's question.
        k: Number of documents to return.
        use_reranker: If True, fetch n_candidates from ChromaDB then rerank to top k
                      using cross-encoder/ms-marco-MiniLM-L-6-v2.
        n_candidates: Number of candidates to fetch before reranking (only used when use_reranker=True).
        use_hybrid: If True, combine vector search with BM25 using Reciprocal Rank Fusion (RRF).
        use_hyde: If True, embed a generated hypothetical answer instead of the raw question.
    """
    ensure_index()
    fetch_n = max(n_candidates, k) if use_reranker else k

    # Vector search — embed hypothesis if HyDE enabled, otherwise embed raw question
    query_text = generate_hypothesis(question) if use_hyde else question
    q_emb = embedder.encode(query_text, normalize_embeddings=True).astype(np.float32).tolist()
    results = chroma_collection.query(
        query_embeddings=[q_emb],
        n_results=fetch_n,
        include=["documents", "metadatas", "embeddings"],
    )
    vector_ids = results["ids"][0]
    docs_by_id = {}
    embs_by_id = {}
    for i, doc_id in enumerate(vector_ids):
        meta = results["metadatas"][0][i]
        text = results["documents"][0][i]
        docs_by_id[doc_id] = IndexedDoc(
            doc_id=int(doc_id),
            path=meta["path"],
            start_line=meta["start_line"],
            end_line=meta["end_line"],
            text=text,
        )
        embs_by_id[doc_id] = results["embeddings"][0][i]

    if use_hybrid and _bm25_index is not None:
        # BM25 search
        bm25_scores = _bm25_index.get_scores(question.lower().split())
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:fetch_n]
        bm25_ids = [_bm25_doc_ids[i] for i in bm25_top_indices]

        # RRF merge (k=60 standard constant)
        rrf_k = 60
        rrf_scores: dict = {}
        for rank, doc_id in enumerate(vector_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)
        for rank, doc_id in enumerate(bm25_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)

        merged_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:fetch_n]

        # Fetch any BM25-only docs not already in vector results
        missing_ids = [doc_id for doc_id in merged_ids if doc_id not in docs_by_id]
        if missing_ids:
            extra = chroma_collection.get(ids=missing_ids, include=["documents", "metadatas", "embeddings"])
            for i, doc_id in enumerate(extra["ids"]):
                meta = extra["metadatas"][i]
                docs_by_id[doc_id] = IndexedDoc(
                    doc_id=int(doc_id),
                    path=meta["path"],
                    start_line=meta["start_line"],
                    end_line=meta["end_line"],
                    text=extra["documents"][i],
                )
                embs_by_id[doc_id] = extra["embeddings"][i]

        candidate_ids = merged_ids
    else:
        candidate_ids = vector_ids

    docs = [docs_by_id[doc_id] for doc_id in candidate_ids]
    doc_embeddings = [embs_by_id[doc_id] for doc_id in candidate_ids]

    if use_reranker and len(docs) > k:
        global _reranker
        if _reranker is None:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(question, doc.text) for doc in docs]
        scores = _reranker.predict(pairs)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        docs = [docs[i] for i in ranked_indices]
        doc_embeddings = [doc_embeddings[i] for i in ranked_indices]

    # Post-retrieval deduplication: drop near-duplicate chunks keeping highest-ranked first
    dedup_threshold = float(os.environ.get("BOT_RETRIEVAL_DEDUP_THRESHOLD", "0.90"))
    kept_docs = []
    kept_embs = []
    for doc, emb in zip(docs, doc_embeddings):
        emb_arr = np.array(emb)
        if kept_embs:
            sims = np.array(kept_embs) @ emb_arr
            if np.max(sims) >= dedup_threshold:
                continue
        kept_docs.append(doc)
        kept_embs.append(emb_arr)
        if len(kept_docs) == k:
            break

    return kept_docs


def linkify(path: str, start_line: int, end_line: int) -> str:
    base = os.environ.get("BOT_BASE_URL")
    if not base:
        return f"{path}#L{start_line}-L{end_line}"
    return f"{base}/{path}#L{start_line}-L{end_line}"


def build_system_prompt() -> str:
    return (
        "You are an expert assistant for Underworld3, a geodynamics modeling framework.\n"
        "- You understand PETSc, parallel computing, finite element methods, and computational geodynamics.\n"
        "- Answer ONLY using the provided repository context.\n"
        "- If the context does not contain enough information to answer fully, explicitly state what is missing rather than inferring or guessing. Do not fill gaps with general knowledge.\n"
        "- Provide concise, correct, runnable code examples with proper imports.\n"
        "- ALWAYS cite file paths and line ranges (format: `file.py:123-145`).\n"
        "- For solver questions, mention PETSc compatibility requirements.\n"
        "- For parallel safety, reference patterns in CLAUDE.md (use uw.pprint(), uw.selective_ranks()).\n"
        "- Never promise features or roadmap items not explicitly in the code.\n"
        "\n"
        "Key priorities:\n"
        "1. Solver stability is paramount (never suggest changes to core solvers)\n"
        "2. Always rebuild after source changes: `pixi run underworld-build`\n"
        "3. Parallel safety is critical in all examples"
    )


def format_context(ctx: List[IndexedDoc]) -> str:
    return "\n\n".join(f"[{d.path}:{d.start_line}-{d.end_line}]\n{d.text}" for d in ctx)


def call_llm_with_caching(system_prompt: str, user_prompt: str, context: str) -> str:
    """
    Call Claude with prompt caching for cost savings.

    The context is cached, so repeated queries with similar context
    cost 90% less after the first query.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "I don't have Claude configured. Set ANTHROPIC_API_KEY environment variable."

    try:
        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            temperature=0.2,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}  # Cache system prompt (1hr TTL)
                },
                {
                    "type": "text",
                    "text": f"Repository context (this is cached for efficiency):\n\n{context}",
                    "cache_control": {"type": "ephemeral"}  # Cache context (1hr TTL)
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            extra_headers={"anthropic-beta": "extended-cache-ttl-2025-04-11"}
        )

        return message.content[0].text

    except anthropic.APIError as e:
        return f"Claude API error: {str(e)}"
    except Exception as e:
        return f"Unexpected error calling Claude: {str(e)}"


def enforce_citations(answer_md: str, ctx: List[IndexedDoc]) -> Tuple[str, List[str], List[str]]:
    used = sorted({d.path for d in ctx if d.path in answer_md})
    citations = []
    for d in ctx:
        if d.path in answer_md:
            citations.append(linkify(d.path, d.start_line, d.end_line))
    if not citations:
        return (
            "I don’t have enough repo context to answer confidently. "
            "Please share the relevant file path or snippet.",
            [],
            [],
        )
    return (answer_md, citations, used)


@app.post("/ask", response_model=BotResponse)
def ask(q: Query):
    start_time = time.time()

    ctx = retrieve(q.question, k=q.max_context_items, use_reranker=True, n_candidates=10, use_hybrid=USE_HYBRID_SEARCH, use_hyde=USE_HYDE)
    system_prompt = build_system_prompt()
    context_text = format_context(ctx)
    user_prompt = (
        f"Question: {q.question}\n\n"
        "Be concise — answer directly and completely, but avoid unnecessary explanation. "
        "Include code examples only when they add clarity. "
        "Include citations to specific files and line ranges."
    )
    raw = call_llm_with_caching(system_prompt, user_prompt, context_text)
    print(f"raw llm output: \n {raw}")
    answer, citations, used_files = enforce_citations(raw, ctx)
    confidence = 0.5 if "don't have" in answer or "Claude" in answer and "error" in answer else 0.8

    # Calculate response time
    response_time_ms = int((time.time() - start_time) * 1000)

    # Log the interaction for training data
    docs_used = [{"file": f, "doc_id": i, "score": None} for i, f in enumerate(used_files)]
    interaction_id = interaction_logger.log_interaction(
        question=q.question,
        answer=answer,
        docs_used=docs_used,
        confidence=confidence,
        response_time_ms=response_time_ms,
        channel="api",
        metadata={"citations": citations}
    )

    return BotResponse(
        answer=answer, citations=citations, used_files=used_files, confidence=confidence
    )


_SEARCH_DOCS_TOOL = {
    "name": "search_docs",
    "description": (
        "Search the Underworld3 source code and documentation index. "
        "Call this to find relevant code, APIs, or documentation. "
        "Use different queries to gather comprehensive information."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "k": {"type": "integer", "description": "Number of results to return (default 5)", "default": 5},
        },
        "required": ["query"],
    },
}


def _agent_collect_chunks(
    question: str, k: int,
    use_reranker: bool, n_candidates: int,
    use_hybrid: bool, use_hyde: bool,
    max_calls: int = 6,
) -> List[IndexedDoc]:
    """Run tool-use agent loop to collect chunks. Returns aggregated unique chunks."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return []

    client = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": (
        f"Question: {question}\n\n"
        "Use the search_docs tool to find relevant information. "
        "Search multiple times with different queries if needed."
    )}]
    all_chunks: List[IndexedDoc] = []
    seen_ids: set = set()
    call_count = 0
    total_retrieved = 0
    stop_reason = "max_calls reached"

    while call_count < max_calls:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            temperature=0.0,
            system="You are a search assistant for the Underworld3 codebase. Use search_docs to gather information needed to answer the question. Stop searching when you have enough context.",
            tools=[_SEARCH_DOCS_TOOL],
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            stop_reason = "Claude decided to stop"
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "search_docs":
                call_count += 1
                query = block.input.get("query", question)
                result_k = min(int(block.input.get("k", 5)), k)
                chunks = retrieve(
                    query, k=result_k, use_reranker=use_reranker,
                    n_candidates=n_candidates, use_hybrid=use_hybrid, use_hyde=use_hyde,
                )
                total_retrieved += len(chunks)
                print(f"  [agent] call {call_count}: query='{query}', k={result_k} → {len(chunks)} chunks ({len(seen_ids)} unique so far)")
                for ch in chunks:
                    if ch.doc_id not in seen_ids:
                        seen_ids.add(ch.doc_id)
                        all_chunks.append(ch)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": format_context(chunks) if chunks else "No results found.",
                })
        messages.append({"role": "user", "content": tool_results})

    duplicates = total_retrieved - len(all_chunks)
    print(f"  [agent] done: {call_count} calls, {len(all_chunks)} unique chunks, {duplicates} duplicates filtered, stopped: {stop_reason}")
    if not all_chunks:
        print(f"  [agent] WARNING: no chunks retrieved — answer will be generated without context")

    return all_chunks


def _stream_ask_agent(q: Query):
    """SSE generator for /ask/agent/stream — agent collects chunks via tool-use, then streams answer."""
    start_time = time.time()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        yield f"data: {json.dumps({'type': 'error', 'text': 'ANTHROPIC_API_KEY not set.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    yield f"data: {json.dumps({'type': 'status'})}\n\n"
    all_chunks = _agent_collect_chunks(
        q.question, k=q.max_context_items,
        use_reranker=True, n_candidates=10,
        use_hybrid=USE_HYBRID_SEARCH, use_hyde=USE_HYDE,
        max_calls=6,
    )
    yield f"data: {json.dumps({'type': 'status'})}\n\n"

    # Emit collected context so eval.py can record retrieved_contexts
    yield f"data: {json.dumps({'type': 'context', 'chunks': [{'text': ch.text, 'path': ch.path, 'doc_id': ch.doc_id} for ch in all_chunks]})}\n\n"

    system_prompt = build_system_prompt()
    context_text = format_context(all_chunks)
    user_prompt = (
        f"Question: {q.question}\n\n"
        "Be concise — answer directly and completely, but avoid unnecessary explanation. "
        "Include code examples only when they add clarity. "
        "Include citations to specific files and line ranges."
    )

    client = anthropic.Anthropic(api_key=api_key)
    full_text = ""
    try:
        with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            temperature=0.2,
            system=[
                {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": f"Repository context:\n\n{context_text}", "cache_control": {"type": "ephemeral"}},
            ],
            messages=[{"role": "user", "content": user_prompt}],
            extra_headers={"anthropic-beta": "extended-cache-ttl-2025-04-11"},
        ) as stream:
            for text in stream.text_stream:
                full_text += text
                yield f"data: {json.dumps({'type': 'text', 'text': text})}\n\n"
    except anthropic.APIError as e:
        yield f"data: {json.dumps({'type': 'error', 'text': f'Claude API error: {e}'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    seen_paths: set = set()
    citations: list = []
    used_files: list = []
    for d in all_chunks:
        if d.path in full_text and d.path not in seen_paths:
            seen_paths.add(d.path)
            citations.append(linkify(d.path, d.start_line, d.end_line))
            used_files.append(d.path)

    response_time_ms = int((time.time() - start_time) * 1000)
    docs_used = [{"file": f, "doc_id": i, "score": None} for i, f in enumerate(used_files)]
    interaction_logger.log_interaction(
        question=q.question, answer=full_text, docs_used=docs_used,
        confidence=0.8 if citations else 0.5,
        response_time_ms=response_time_ms, channel="api",
        metadata={"citations": citations, "mode": "agent"},
    )

    yield f"data: {json.dumps({'type': 'citations', 'citations': citations, 'used_files': used_files})}\n\n"
    yield "data: [DONE]\n\n"


def _stream_ask(q: Query):
    """SSE generator for /ask/stream — yields text deltas then a citations event."""
    start_time = time.time()
    yield f"data: {json.dumps({'type': 'status'})}\n\n"
    ctx = retrieve(q.question, k=q.max_context_items, use_reranker=True,
                   n_candidates=10, use_hybrid=USE_HYBRID_SEARCH, use_hyde=USE_HYDE)
    yield f"data: {json.dumps({'type': 'status'})}\n\n"
    system_prompt = build_system_prompt()
    context_text = format_context(ctx)
    user_prompt = (
        f"Question: {q.question}\n\n"
        "Be concise — answer directly and completely, but avoid unnecessary explanation. "
        "Include code examples only when they add clarity. "
        "Include citations to specific files and line ranges."
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        yield f"data: {json.dumps({'type': 'error', 'text': 'ANTHROPIC_API_KEY not set.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    client = anthropic.Anthropic(api_key=api_key)
    full_text = ""
    try:
        with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            temperature=0.2,
            system=[
                {"type": "text", "text": system_prompt,
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text",
                 "text": f"Repository context (this is cached for efficiency):\n\n{context_text}",
                 "cache_control": {"type": "ephemeral"}},
            ],
            messages=[{"role": "user", "content": user_prompt}],
            extra_headers={"anthropic-beta": "extended-cache-ttl-2025-04-11"},
        ) as stream:
            for text in stream.text_stream:
                full_text += text
                yield f"data: {json.dumps({'type': 'text', 'text': text})}\n\n"
    except anthropic.APIError as e:
        yield f"data: {json.dumps({'type': 'error', 'text': f'Claude API error: {e}'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Compute citations from accumulated text (same logic as enforce_citations)
    seen_paths: set = set()
    citations: list = []
    used_files: list = []
    for d in ctx:
        if d.path in full_text and d.path not in seen_paths:
            seen_paths.add(d.path)
            citations.append(linkify(d.path, d.start_line, d.end_line))
            used_files.append(d.path)

    # Log interaction
    response_time_ms = int((time.time() - start_time) * 1000)
    docs_used = [{"file": f, "doc_id": i, "score": None} for i, f in enumerate(used_files)]
    interaction_logger.log_interaction(
        question=q.question,
        answer=full_text,
        docs_used=docs_used,
        confidence=0.8 if citations else 0.5,
        response_time_ms=response_time_ms,
        channel="api",
        metadata={"citations": citations},
    )

    yield f"data: {json.dumps({'type': 'citations', 'citations': citations, 'used_files': used_files})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/ask/stream")
def ask_stream(q: Query):
    return StreamingResponse(_stream_ask(q), media_type="text/event-stream")


@app.post("/ask/agent/stream")
def ask_agent_stream(q: Query):
    return StreamingResponse(_stream_ask_agent(q), media_type="text/event-stream")


@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "ok" if index_built else "loading",
        "index_built": index_built,
        "doc_count": chroma_collection.count() if chroma_collection else 0,
        "embedding_model": MODEL_NAME,
        "claude_model": CLAUDE_MODEL
    }


@app.get("/interactions/stats")
def interaction_stats():
    """Get statistics about logged interactions."""
    return interaction_logger.get_stats()


@app.get("/interactions/patterns")
def question_patterns():
    """Get analysis of common question patterns."""
    return interaction_logger.get_question_patterns()


@app.get("/interactions/recent")
def recent_interactions(limit: int = 10):
    """Get recent interactions for review."""
    return interaction_logger.get_interactions(limit=limit)


def _do_rebuild():
    global index_built, _rebuild_status
    _rebuild_status = {"state": "running", "message": "Pulling latest UW3 repo...", "started_at": time.time()}
    repo_path = os.environ.get("BOT_REPO_PATH", "")
    try:
        if repo_path:
            subprocess.run(
                ["git", "-C", repo_path, "pull", "--ff-only", "origin", "main"],
                capture_output=True, timeout=120,
            )
        _rebuild_status["message"] = "Rebuilding index..."
        os.environ["BOT_FORCE_REBUILD"] = "1"
        index_built = False
        ensure_index()
        os.environ.pop("BOT_FORCE_REBUILD", None)
        _rebuild_status = {"state": "done", "message": "Rebuild complete.", "started_at": _rebuild_status["started_at"]}
    except Exception as e:
        os.environ.pop("BOT_FORCE_REBUILD", None)
        _rebuild_status = {"state": "error", "message": str(e), "started_at": _rebuild_status["started_at"]}


@app.post("/rebuild")
def trigger_rebuild(x_rebuild_token: str = Header(None)):
    expected = os.environ.get("REBUILD_TOKEN", "")
    if not expected or x_rebuild_token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if _rebuild_status.get("state") == "running":
        return {"status": "already_running", "message": _rebuild_status["message"]}
    import threading
    threading.Thread(target=_do_rebuild, daemon=True).start()
    return {"status": "started"}


@app.get("/rebuild/status")
def rebuild_status():
    return _rebuild_status


def find_available_port(start_port=8001, max_attempts=10):
    """
    Find an available port starting from start_port.

    Args:
        start_port: Port to start searching from (default: 8001)
        max_attempts: Maximum number of ports to try (default: 10)

    Returns:
        int: Available port number

    Raises:
        RuntimeError: If no available port found in range
    """
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port+max_attempts}")


def write_port_file(port, port_file="bot.port"):
    """
    Write the port number to a file so clients can find the bot.

    Args:
        port: Port number to write
        port_file: File to write port to (default: bot.port)
    """
    port_path = Path(__file__).parent / port_file
    port_path.write_text(str(port))
    print(f"Port {port} written to {port_path}")


if __name__ == "__main__":
    # Find available port
    port = find_available_port(8001)
    print(f"Starting HelpfulBatBot on port {port}")

    # Write port to file for clients
    write_port_file(port)

    # Start server
    uvicorn.run(app, host="0.0.0.0", port=port)
