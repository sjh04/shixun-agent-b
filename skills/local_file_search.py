from __future__ import annotations

import math
import re
from datetime import datetime

from skills import resolve_data_path
from skills.error_codes import (
    ERR_FILE_NOT_FOUND,
    ERR_FILE_TYPE,
    ERR_PARAM_MISSING,
    ERR_PARAM_RANGE,
    ERR_PARAM_TYPE,
    SkillError,
)


_STOP_WORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "可以", "被", "把", "让", "用", "对", "从", "而", "但", "或", "与",
    "以", "及", "为", "等", "将", "其", "所", "能", "如", "向", "使",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "and", "but", "or", "if", "while", "about", "up",
    "out", "it", "its", "this", "that", "these", "those",
}


def _tokenize(text: str) -> list[str]:
    lowered = text.casefold()
    tokens: list[str] = []

    for match in re.finditer(r"[一-鿿]+", lowered):
        seq = match.group()
        if len(seq) == 1:
            tokens.append(seq)
        else:
            for i in range(len(seq) - 1):
                tokens.append(seq[i:i + 2])

    non_chinese = re.sub(r"[一-鿿]+", " ", lowered)
    for token in re.split(r"[^a-z0-9]+", non_chinese):
        if token:
            tokens.append(token)

    return [t for t in tokens if t not in _STOP_WORDS]


def _snippet(text: str, terms: list[str], radius: int = 60) -> str:
    lowered = text.casefold()
    positions = [lowered.find(term.casefold()) for term in terms]
    positions = [position for position in positions if position >= 0]
    start = max(0, (min(positions) if positions else 0) - radius)
    end = min(len(text), start + radius * 2)
    prefix = "..." if start else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end].replace("\n", " ").strip() + suffix


def _compute_idf(doc_token_lists: list[list[str]]) -> dict[str, float]:
    N = len(doc_token_lists)
    all_terms: set[str] = set()
    for tokens in doc_token_lists:
        all_terms.update(tokens)
    idf: dict[str, float] = {}
    for term in all_terms:
        df = sum(1 for tokens in doc_token_lists if term in tokens)
        idf[term] = math.log((N + 1) / (df + 1)) + 1
    return idf


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    total = len(tokens)
    if total == 0:
        return {}
    tf: dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return {term: (cnt / total) * idf[term]
            for term, cnt in tf.items() if term in idf}


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    shared = a.keys() & b.keys()
    if not shared:
        return 0.0
    dot = sum(a[k] * b[k] for k in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def local_file_search(
    query: str,
    root_dir: str = "docs",
    file_types: list[str] | None = None,
    top_k: int = 5,
    *,
    data_root: str | None = None,
) -> dict:
    if not isinstance(query, str) or not query.strip():
        raise SkillError(ERR_PARAM_MISSING, "query must be a non-empty string", {"param": "query"})
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise SkillError(ERR_PARAM_TYPE, f"top_k must be an integer, got {type(top_k).__name__}", {"param": "top_k", "value": top_k})
    if top_k <= 0:
        raise SkillError(ERR_PARAM_RANGE, "top_k must be a positive integer", {"param": "top_k", "value": top_k})
    search_root, data_root_path = resolve_data_path(root_dir, data_root)
    if not search_root.is_dir():
        raise SkillError(ERR_FILE_NOT_FOUND, f"search directory not found: {root_dir}", {"root_dir": root_dir})
    extensions = file_types or ["txt", "md"]
    normalized_extensions = {f".{item.lower().lstrip('.')}" for item in extensions}
    if not normalized_extensions.issubset({".txt", ".md"}):
        raise SkillError(ERR_FILE_TYPE, "local_file_search only supports txt and md", {"file_types": file_types})

    query_terms = _tokenize(query)
    raw_terms = [term for term in re.split(r"\s+", query.strip()) if term]

    if not query_terms:
        return {"results": []}

    files: list[dict] = []
    for path in sorted(search_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in normalized_extensions:
            continue
        try:
            text = path.read_text(encoding="utf-8")
            stat = path.stat()
        except (UnicodeDecodeError, PermissionError, OSError):
            continue
        tokens = _tokenize(text)
        if not tokens:
            continue
        files.append({
            "path_obj": path,
            "text": text,
            "tokens": tokens,
            "rel_path": path.relative_to(data_root_path).as_posix(),
            "file_size_bytes": stat.st_size,
            "mtime": stat.st_mtime,
        })

    if not files:
        return {"results": []}

    all_doc_tokens = [f["tokens"] for f in files]

    idf = _compute_idf(all_doc_tokens)

    query_vec = _tfidf_vector(query_terms, idf)

    query_lower = query.strip().casefold()
    results: list[dict] = []
    for f in files:
        doc_vec = _tfidf_vector(f["tokens"], idf)
        raw_score = _cosine_similarity(query_vec, doc_vec)

        if query_lower and query_lower in f["text"].casefold():
            if raw_score == 0.0:
                raw_score = 0.01
            else:
                raw_score += 0.05

        if raw_score > 0:
            results.append({
                "path": f["rel_path"],
                "score": round(raw_score, 4),
                "_raw_score": raw_score,
                "snippet": _snippet(f["text"], raw_terms),
                "file_size_bytes": f["file_size_bytes"],
                "mtime": datetime.fromtimestamp(f["mtime"]).strftime("%Y-%m-%d %H:%M:%S"),
            })

    results.sort(key=lambda item: (-item["_raw_score"], item["path"]))
    for r in results:
        del r["_raw_score"]
    return {"results": results[:top_k]}