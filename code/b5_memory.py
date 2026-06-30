from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from common.io_utils import append_jsonl, read_json, read_text, read_yaml, write_json, write_text
from common.logging_utils import now_iso
from common.path_utils import resolve_cli_path, resolve_from_file


DEFAULTS = {
    "retrieval": {
        "mode": "hybrid",
        "top_k": 5,
        "candidate_limit": 30,
        "chunk_chars": 700,
        "chunk_overlap": 120,
        "rrf_k": 60,
        "recency_half_life_days": 30,
        "include_all_conversations": True,
        "include_global_when_query": True,
        "use_hyde": True,
        "use_rerank": True,
        "hashing_dim": 768,
        "vector_backend": "qwen_or_hashing",
    },
    "compression": {
        "enabled": True,
        "summary_chars": 700,
        "summary_sentences": 5,
        "min_chars_for_summary": 480,
    },
    "integration": {
        "enabled": True,
        "duplicate_similarity": 0.92,
        "conflict_similarity": 0.28,
    },
    "poison_gate": {
        "enabled": True,
        "action": "flag",
        "trusted_memory_types": ["global"],
        "conflict_threshold": 0.32,
    },
    "lifecycle": {
        "max_memories": 200,
        "protect_global": True,
        "importance_default": 5,
        "reflect_enabled": True,
        "reflect_every_n": 10,
        "reflect_min_cluster_size": 3,
    },
    "llm": {
        "enabled": False,
        "model_config": "../configs/model.yaml",
        "mode": "prompt_json",
        "max_new_tokens": 512,
    },
}

_QWEN_CACHE: dict[tuple[str, str], tuple[Any, Any, Any]] = {}


def _deep_merge(default: dict, override: dict | None) -> dict:
    merged = deepcopy(default)
    if not isinstance(override, dict):
        return merged
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_memory_config(config_path: str | Path) -> tuple[Path, dict]:
    path = Path(config_path).resolve()
    config = read_yaml(path)
    if not isinstance(config, dict) or not isinstance(config.get("memory"), dict):
        raise ValueError("memory.yaml must define a memory object")
    memory = _deep_merge(DEFAULTS, config.get("memory"))
    required = ["root_dir", "global_memory_dir", "conversation_memory_dir", "index_path", "max_memory_chars"]
    missing = [name for name in required if name not in memory]
    if missing:
        raise ValueError(f"memory.yaml missing: {', '.join(missing)}")
    max_chars = memory["max_memory_chars"]
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
        raise ValueError("max_memory_chars must be a positive integer")
    return path, memory


def _memory_paths(config_path: str | Path) -> dict[str, Any]:
    path, memory = _load_memory_config(config_path)
    root = resolve_from_file(memory["root_dir"], path)
    return {
        "config_path": path,
        "config": memory,
        "root": root,
        "global": root / memory["global_memory_dir"],
        "conversations": root / memory["conversation_memory_dir"],
        "index": root / memory["index_path"],
        "max_chars": memory["max_memory_chars"],
    }


def _read_index(index_path: Path) -> dict:
    if not index_path.exists():
        return {}
    index = read_json(index_path)
    if not isinstance(index, dict):
        raise ValueError("memory_index.json must be an object")
    return index


def _safe_conversation_id(conversation_id: str) -> str:
    if not isinstance(conversation_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", conversation_id):
        raise ValueError("conversation_id may only contain letters, numbers, dot, underscore, and hyphen")
    return conversation_id


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _age_days(metadata: dict, now: datetime) -> float:
    stamp = _parse_time(metadata.get("last_accessed_at")) or _parse_time(metadata.get("updated_at")) or _parse_time(metadata.get("created_at"))
    if stamp is None:
        return 0.0
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return max(0.0, (now - stamp.astimezone(timezone.utc)).total_seconds() / 86400)


def _recency_factor(metadata: dict, now: datetime, half_life_days: float) -> float:
    half_life = max(float(half_life_days), 1.0)
    return 0.5 ** (_age_days(metadata, now) / half_life)


def _importance_factor(metadata: dict, default: int) -> float:
    value = metadata.get("importance", default)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(default)
    return min(1.0, max(0.1, numeric / 10.0))


def _memory_document_path(paths: dict, metadata: dict) -> tuple[Path | None, str | None, dict | None]:
    relative_path = metadata.get("path")
    if not isinstance(relative_path, str):
        return None, None, {"type": "InvalidMetadata", "message": "memory path is missing"}
    document_path = (paths["root"] / relative_path).resolve()
    try:
        document_path.relative_to(paths["root"].resolve())
    except ValueError:
        return None, relative_path, {"type": "InvalidPath", "message": "memory path escapes root"}
    if not document_path.is_file():
        return None, relative_path, {"type": "FileNotFoundError", "message": f"memory file not found: {relative_path}"}
    return document_path, relative_path, None


def _load_memory_docs(paths: dict, index: dict, memory_ids: list[str]) -> tuple[list[dict], list[dict]]:
    docs = []
    errors = []
    for memory_id in memory_ids:
        metadata = index.get(memory_id)
        if not isinstance(metadata, dict):
            errors.append({"memory_id": memory_id, "type": "MemoryNotFound", "message": "memory_id does not exist"})
            continue
        document_path, relative_path, error = _memory_document_path(paths, metadata)
        if error:
            errors.append({"memory_id": memory_id, **error})
            continue
        content = read_text(document_path)
        docs.append(
            {
                "memory_id": memory_id,
                "metadata": metadata,
                "path": relative_path,
                "content": content,
                "search_content": _memory_search_text(metadata, content),
            }
        )
    return docs, errors


def _section(markdown: str, heading: str) -> str:
    pattern = rf"^## {re.escape(heading)}\s*$"
    lines = markdown.splitlines()
    start = None
    for index, line in enumerate(lines):
        if re.match(pattern, line):
            start = index + 1
            break
    if start is None:
        return ""
    end = len(lines)
    for index in range(start, len(lines)):
        if lines[index].startswith("## "):
            end = index
            break
    return "\n".join(lines[start:end]).strip()


def _memory_search_text(metadata: dict, markdown: str) -> str:
    parts = [
        str(metadata.get("title") or ""),
        str(metadata.get("summary") or ""),
    ]
    for heading in ("Final Answer", "Insight", "Previous Answer"):
        section = _section(markdown, heading)
        if section:
            parts.append(section)
    change_report = _section(markdown, "Change Report")
    if change_report:
        parts.append(change_report[:1200])
    text = "\n\n".join(part for part in parts if part.strip()).strip()
    return text or markdown


def _tokenize(text: str) -> list[str]:
    lowered = text.casefold()
    tokens = re.findall(r"[a-z0-9_]+", lowered)
    cjk = re.findall(r"[\u4e00-\u9fff]", lowered)
    tokens.extend(cjk)
    tokens.extend("".join(pair) for pair in zip(cjk, cjk[1:]))
    return [token for token in tokens if token.strip()]


def _sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[。！？!?；;])|\n+", text)
    return [item.strip() for item in raw if item and item.strip()]


def _hash_vector(tokens: list[str], dim: int) -> dict[int, float]:
    vector: dict[int, float] = defaultdict(float)
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % dim
        sign = 1.0 if (value >> 63) == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm:
        for index in list(vector):
            vector[index] /= norm
    return dict(vector)


def _cosine_dict(left: dict[int, float], right: dict[int, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(index, 0.0) for index, value in left.items())


def _text_similarity(left: str, right: str, dim: int = 512) -> float:
    return _cosine_dict(_hash_vector(_tokenize(left), dim), _hash_vector(_tokenize(right), dim))


def _chunk_text(text: str, chunk_chars: int, overlap: int) -> list[dict]:
    chunk_chars = max(160, int(chunk_chars))
    overlap = max(0, min(int(overlap), chunk_chars // 2))
    paragraphs = [item.strip() for item in re.split(r"\n{2,}", text) if item.strip()]
    chunks = []
    current = ""
    start = 0
    for paragraph in paragraphs or [text]:
        if len(paragraph) > chunk_chars:
            step = max(1, chunk_chars - overlap)
            for offset in range(0, len(paragraph), step):
                piece = paragraph[offset : offset + chunk_chars].strip()
                if piece:
                    chunks.append({"chunk_index": len(chunks), "content": piece, "start": offset, "end": offset + len(piece)})
            continue
        if current and len(current) + len(paragraph) + 2 > chunk_chars:
            chunks.append({"chunk_index": len(chunks), "content": current.strip(), "start": start, "end": start + len(current)})
            tail = current[-overlap:] if overlap else ""
            current = (tail + "\n\n" + paragraph).strip() if tail else paragraph
            start += max(0, len(current) - len(tail))
        else:
            current = (current + "\n\n" + paragraph).strip() if current else paragraph
    if current:
        chunks.append({"chunk_index": len(chunks), "content": current.strip(), "start": start, "end": start + len(current)})
    return chunks or [{"chunk_index": 0, "content": text[:chunk_chars], "start": 0, "end": min(len(text), chunk_chars)}]


def _bm25_scores(query: str, chunks: list[dict]) -> dict[int, float]:
    query_terms = _tokenize(query)
    if not query_terms or not chunks:
        return {}
    query_counts = Counter(query_terms)
    doc_tokens = [_tokenize(chunk["content"]) for chunk in chunks]
    doc_freq: Counter[str] = Counter()
    for tokens in doc_tokens:
        doc_freq.update(set(tokens))
    avg_len = sum(len(tokens) for tokens in doc_tokens) / max(1, len(doc_tokens))
    k1 = 1.5
    b = 0.75
    scores: dict[int, float] = {}
    for index, tokens in enumerate(doc_tokens):
        counts = Counter(tokens)
        length = len(tokens) or 1
        score = 0.0
        for term, q_count in query_counts.items():
            freq = counts.get(term, 0)
            if not freq:
                continue
            idf = math.log(1 + (len(chunks) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denom = freq + k1 * (1 - b + b * length / max(avg_len, 1))
            score += q_count * idf * ((freq * (k1 + 1)) / denom)
        if score:
            scores[index] = score
    return scores


def _rank_from_scores(scores: dict[int, float]) -> dict[int, int]:
    return {item: rank for rank, (item, _) in enumerate(sorted(scores.items(), key=lambda pair: (-pair[1], pair[0])), 1)}


def _qwen_paths(config_path: Path, memory_config: dict) -> tuple[Path | None, Path | None]:
    llm = memory_config.get("llm", {})
    setting = llm.get("model_config")
    if not isinstance(setting, str):
        return None, None
    model_config_path = resolve_from_file(setting, config_path)
    model_config = read_yaml(model_config_path)
    model = model_config.get("model", {}) if isinstance(model_config, dict) else {}
    model_setting = model.get("model_name_or_path")
    tokenizer_setting = model.get("tokenizer_name_or_path", model_setting)
    if not isinstance(model_setting, str) or not isinstance(tokenizer_setting, str):
        return None, None
    return resolve_from_file(model_setting, model_config_path), resolve_from_file(tokenizer_setting, model_config_path)


def _torch_dtype(torch_module: Any, configured: Any) -> Any:
    if configured in {None, "auto"}:
        return "auto"
    mapping = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }
    if configured not in mapping:
        raise ValueError(f"unsupported torch_dtype: {configured}")
    return mapping[configured]


def _load_qwen(config_path: Path, memory_config: dict) -> tuple[Any, Any, Any]:
    llm = memory_config.get("llm", {})
    model_config_path = resolve_from_file(llm.get("model_config", "../configs/model.yaml"), config_path)
    cache_key = (str(model_config_path), str(llm.get("mode", "prompt_json")))
    cached = _QWEN_CACHE.get(cache_key)
    if cached:
        return cached
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Qwen features require torch and transformers") from exc
    model_config = read_yaml(model_config_path)
    model = model_config.get("model", {}) if isinstance(model_config, dict) else {}
    model_path = resolve_from_file(model.get("model_name_or_path"), model_config_path)
    tokenizer_path = resolve_from_file(model.get("tokenizer_name_or_path", model.get("model_name_or_path")), model_config_path)
    local_only = bool(model.get("local_files_only", True))
    trust_remote_code = bool(model.get("trust_remote_code", True))
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=local_only,
        trust_remote_code=trust_remote_code,
    )
    loaded = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=local_only,
        trust_remote_code=trust_remote_code,
        dtype=_torch_dtype(torch, model.get("torch_dtype", "auto")),
        device_map=model.get("device_map", "auto"),
        max_memory=model.get("max_memory"),
    )
    loaded.eval()
    bundle = (torch, tokenizer, loaded)
    _QWEN_CACHE[cache_key] = bundle
    return bundle


def _qwen_generate(config_path: Path, memory_config: dict, prompt: str) -> str:
    if not memory_config.get("llm", {}).get("enabled", False):
        raise RuntimeError("Qwen generation is disabled")
    torch, tokenizer, model = _load_qwen(config_path, memory_config)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    )
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    input_length = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(memory_config.get("llm", {}).get("max_new_tokens", 512)),
            do_sample=False,
        )
    return tokenizer.decode(generated[0][input_length:], skip_special_tokens=True).strip()


def _qwen_embedding(config_path: Path, memory_config: dict, text: str) -> dict[int, float]:
    if not memory_config.get("llm", {}).get("enabled", False):
        raise RuntimeError("Qwen embedding is disabled")
    torch, tokenizer, model = _load_qwen(config_path, memory_config)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden = outputs.hidden_states[-1][0]
    mask = inputs["attention_mask"][0].unsqueeze(-1)
    pooled = (hidden * mask).sum(dim=0) / mask.sum().clamp(min=1)
    pooled = torch.nn.functional.normalize(pooled.float(), dim=0)
    values = pooled.detach().cpu().tolist()
    return {index: float(value) for index, value in enumerate(values) if abs(float(value)) > 1e-8}


def _vectorize(config_path: Path, memory_config: dict, text: str) -> tuple[dict[int, float], str]:
    retrieval = memory_config.get("retrieval", {})
    backend = retrieval.get("vector_backend", "qwen_or_hashing")
    if backend in {"qwen", "qwen_or_hashing"}:
        try:
            return _qwen_embedding(config_path, memory_config, text), "qwen"
        except Exception:
            if backend == "qwen":
                raise
    return _hash_vector(_tokenize(text), int(retrieval.get("hashing_dim", 768))), "hashing"


def _hyde_query(config_path: Path, memory_config: dict, query: str) -> tuple[str, str]:
    retrieval = memory_config.get("retrieval", {})
    if not retrieval.get("use_hyde", False):
        return query, "disabled"
    try:
        prompt = (
            "Write one concise hypothetical memory passage that would answer this query. "
            "Return only the passage.\n\n"
            f"Query: {query}"
        )
        generated = _qwen_generate(config_path, memory_config, prompt)
        if generated:
            return f"{query}\n{generated}", "qwen"
    except Exception:
        pass
    return query, "fallback"


def _extractive_summary(text: str, budget: int, query: str | None = None, max_sentences: int = 5) -> str:
    clean = text.strip()
    if len(clean) <= budget:
        return clean
    sentences = _sentences(clean)
    if not sentences:
        return clean[:budget].rstrip()
    query_tokens = set(_tokenize(query or ""))
    scored = []
    for index, sentence in enumerate(sentences):
        tokens = _tokenize(sentence)
        overlap = sum(1 for token in tokens if token in query_tokens)
        position_bonus = 1.0 / (index + 1)
        length_penalty = abs(len(sentence) - min(180, budget)) / max(budget, 1)
        score = overlap * 3 + position_bonus - length_penalty
        scored.append((score, index, sentence))
    selected = sorted(scored, key=lambda item: (-item[0], item[1]))[:max_sentences]
    selected.sort(key=lambda item: item[1])
    summary = "\n".join(item[2] for item in selected).strip()
    if len(summary) > budget:
        summary = summary[:budget].rstrip()
    return summary


def _compress_text(config_path: Path, memory_config: dict, text: str, budget: int, query: str | None = None) -> tuple[str, str]:
    compression = memory_config.get("compression", {})
    if not compression.get("enabled", True) or len(text) <= budget:
        return text[:budget], "none" if len(text) <= budget else "hard_truncate"
    try:
        prompt = (
            "Summarize the memory below for future retrieval. Preserve concrete facts, tool results, "
            "conflicts, and user preferences. Return plain text only.\n\n"
            f"Budget characters: {budget}\n"
            f"Query focus: {query or ''}\n\n"
            f"Memory:\n{text[:6000]}"
        )
        generated = _qwen_generate(config_path, memory_config, prompt)
        if generated:
            return generated[:budget].rstrip(), "qwen"
    except Exception:
        pass
    return (
        _extractive_summary(
            text,
            budget,
            query,
            int(compression.get("summary_sentences", 5)),
        ),
        "extractive",
    )


def _summarize_for_index(config_path: Path, memory_config: dict, answer: str) -> tuple[str, str]:
    budget = int(memory_config.get("compression", {}).get("summary_chars", 700))
    return _compress_text(config_path, memory_config, answer, min(300, budget), None)


def _estimate_importance(config_path: Path, memory_config: dict, answer: str, trace: dict) -> tuple[int, str]:
    try:
        prompt = (
            "Rate this memory importance from 1 to 10 for a local tool-using agent. "
            "Return only an integer.\n\n"
            f"Final answer:\n{answer[:3000]}"
        )
        generated = _qwen_generate(config_path, memory_config, prompt)
        match = re.search(r"\b([1-9]|10)\b", generated)
        if match:
            return int(match.group(1)), "qwen"
    except Exception:
        pass
    score = 4
    if len(answer) > 240:
        score += 1
    if trace.get("tool_rounds_used", 0):
        score += 1
    if any(term in answer.casefold() for term in ["error", "错误", "失败", "配置", "路径", "工具", "memory", "agent"]):
        score += 1
    return min(10, max(1, score)), "heuristic"


def _build_chunks(paths: dict, docs: list[dict]) -> list[dict]:
    retrieval = paths["config"].get("retrieval", {})
    chunks = []
    for doc in docs:
        for chunk in _chunk_text(
            doc.get("search_content") or doc["content"],
            int(retrieval.get("chunk_chars", 700)),
            int(retrieval.get("chunk_overlap", 120)),
        ):
            chunks.append(
                {
                    "memory_id": doc["memory_id"],
                    "metadata": doc["metadata"],
                    "path": doc["path"],
                    "chunk_index": chunk["chunk_index"],
                    "content": chunk["content"],
                    "full_content": doc["content"],
                }
            )
    return chunks


def _candidate_ids_for_load(index: dict, selected_memory_ids: list[str], use_global_memory: bool, query: str | None, config: dict) -> list[str]:
    ordered = []
    retrieval = config.get("retrieval", {})
    mode = retrieval.get("mode", "none")
    if query and mode != "none":
        if retrieval.get("include_global_when_query", True):
            ordered.extend(sorted(key for key, item in index.items() if item.get("memory_type") == "global"))
        if retrieval.get("include_all_conversations", True):
            ordered.extend(sorted(key for key, item in index.items() if item.get("memory_type") == "conversation"))
    else:
        if use_global_memory:
            ordered.extend(sorted(key for key, item in index.items() if item.get("memory_type") == "global"))
    ordered.extend(selected_memory_ids)
    return list(dict.fromkeys(ordered))


def _rank_memory_docs(paths: dict, docs: list[dict], query: str | None) -> tuple[list[dict], dict]:
    config = paths["config"]
    retrieval = config.get("retrieval", {})
    mode = str(retrieval.get("mode", "none"))
    if not query or mode == "none":
        return [
            {
                **doc,
                "rank": rank,
                "score": None,
                "factors": {"relevance": None, "recency": None, "importance": None},
                "chunk_index": None,
                "chunk_content": None,
                "retrieval_mode": "none",
            }
            for rank, doc in enumerate(docs, 1)
        ], {"mode": "none", "hyde": "disabled", "vector_backend": None}

    started = perf_counter()
    hyde_query, hyde_mode = _hyde_query(paths["config_path"], config, query)
    chunks = _build_chunks(paths, docs)
    bm25 = _bm25_scores(hyde_query if mode in {"keyword", "hybrid"} else query, chunks) if mode in {"keyword", "hybrid"} else {}
    vector_scores: dict[int, float] = {}
    vector_backend = None
    if mode in {"vector", "hybrid"}:
        query_vector, vector_backend = _vectorize(paths["config_path"], config, hyde_query)
        for index, chunk in enumerate(chunks):
            chunk_vector, used_backend = _vectorize(paths["config_path"], config, chunk["content"])
            vector_backend = vector_backend or used_backend
            score = _cosine_dict(query_vector, chunk_vector)
            if score:
                vector_scores[index] = score
    rankings = []
    if bm25:
        rankings.append(_rank_from_scores(bm25))
    if vector_scores:
        rankings.append(_rank_from_scores(vector_scores))
    if not rankings:
        rankings.append({index: index + 1 for index in range(len(chunks))})
    rrf_k = int(retrieval.get("rrf_k", 60))
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for chunk_index, rank in ranking.items():
            fused[chunk_index] += 1.0 / (rrf_k + rank)

    now = datetime.now(timezone.utc)
    half_life = float(retrieval.get("recency_half_life_days", 30))
    importance_default = int(config.get("lifecycle", {}).get("importance_default", 5))
    best_by_memory: dict[str, dict] = {}
    for chunk_index, relevance in fused.items():
        chunk = chunks[chunk_index]
        metadata = chunk["metadata"]
        recency = _recency_factor(metadata, now, half_life)
        importance = _importance_factor(metadata, importance_default)
        score = relevance * recency * importance
        record = {
            **chunk,
            "score": score,
            "factors": {
                "relevance": relevance,
                "recency": recency,
                "importance": importance,
            },
            "retrieval_mode": mode,
        }
        previous = best_by_memory.get(chunk["memory_id"])
        if previous is None or score > previous["score"]:
            best_by_memory[chunk["memory_id"]] = record
    ranked_chunks = sorted(best_by_memory.values(), key=lambda item: (-item["score"], item["memory_id"]))
    ranked_chunks = ranked_chunks[: int(retrieval.get("candidate_limit", 30))]

    rerank_mode = "disabled"
    if retrieval.get("use_rerank", False):
        ranked_chunks, rerank_mode = _rerank_with_qwen(paths["config_path"], config, query, ranked_chunks)

    doc_by_id = {doc["memory_id"]: doc for doc in docs}
    ranked_docs = []
    for rank, chunk in enumerate(ranked_chunks[: int(retrieval.get("top_k", 5))], 1):
        doc = doc_by_id[chunk["memory_id"]]
        ranked_docs.append(
            {
                **doc,
                "rank": rank,
                "score": round(float(chunk["score"]), 6),
                "factors": {key: round(float(value), 6) for key, value in chunk["factors"].items()},
                "chunk_index": chunk["chunk_index"],
                "chunk_content": chunk["content"],
                "retrieval_mode": mode,
            }
        )
    diagnostics = {
        "mode": mode,
        "hyde": hyde_mode,
        "rerank": rerank_mode,
        "vector_backend": vector_backend,
        "chunk_count": len(chunks),
        "latency_ms": round((perf_counter() - started) * 1000, 3),
    }
    return ranked_docs, diagnostics


def _rerank_with_qwen(config_path: Path, config: dict, query: str, ranked_chunks: list[dict]) -> tuple[list[dict], str]:
    if not ranked_chunks:
        return ranked_chunks, "empty"
    try:
        candidates = [
            {"memory_id": item["memory_id"], "chunk_index": item["chunk_index"], "content": item["content"][:700]}
            for item in ranked_chunks[:10]
        ]
        prompt = (
            "Rerank the memory candidates for the query. Return exactly one valid JSON array of memory_id strings. "
            "Do not output markdown, comments, or extra text.\n\n"
            f"Query: {query}\n"
            f"Candidates:\n{json.dumps(candidates, ensure_ascii=False)}"
        )
        generated = _qwen_generate(config_path, config, prompt)
        order = json.loads(generated)
        if isinstance(order, list):
            position = {str(memory_id): index for index, memory_id in enumerate(order)}
            return sorted(ranked_chunks, key=lambda item: (position.get(item["memory_id"], 999), -item["score"])), "qwen"
    except Exception:
        pass
    return ranked_chunks, "fallback"


def _format_memory_content(paths: dict, doc: dict, query: str | None, remaining: int) -> tuple[str, bool, str]:
    config = paths["config"]
    metadata = doc["metadata"]
    title = metadata.get("title", doc["memory_id"])
    summary = metadata.get("summary", "")
    source = doc.get("chunk_content") or doc["content"]
    if doc.get("retrieval_mode") != "none" and doc.get("chunk_content"):
        source = f"# {title}\n\nSummary: {summary}\n\nRelevant chunk:\n{doc['chunk_content']}"
    budget = max(0, remaining)
    if len(source) <= budget:
        return source, False, "none"
    compressed, method = _compress_text(paths["config_path"], config, source, budget, query)
    return compressed, True, method


def _update_access_metadata(index: dict, docs: list[dict], timestamp: str) -> None:
    for doc in docs:
        metadata = index.get(doc["memory_id"])
        if not isinstance(metadata, dict):
            continue
        metadata["last_accessed_at"] = timestamp
        metadata["access_count"] = int(metadata.get("access_count", 0) or 0) + 1


def load_memory(
    config_path: str,
    selected_memory_ids: list[str],
    use_global_memory: bool,
    query: str | None = None,
    outdir: str | None = None,
) -> dict:
    if not isinstance(selected_memory_ids, list) or not all(isinstance(item, str) for item in selected_memory_ids):
        raise ValueError("selected_memory_ids must be a list of strings")
    paths = _memory_paths(config_path)
    config = paths["config"]
    index = _read_index(paths["index"])
    ordered_ids = _candidate_ids_for_load(index, selected_memory_ids, use_global_memory, query, config)
    docs, errors = _load_memory_docs(paths, index, ordered_ids)
    if query and config.get("retrieval", {}).get("mode") != "none":
        missing_selected = [memory_id for memory_id in selected_memory_ids if memory_id not in index]
        for memory_id in missing_selected:
            if not any(error.get("memory_id") == memory_id for error in errors):
                errors.append({"memory_id": memory_id, "type": "MemoryNotFound", "message": "memory_id does not exist"})
    ranked_docs, diagnostics = _rank_memory_docs(paths, docs, query)

    selected = []
    remaining = int(paths["max_chars"])
    any_truncated = False
    for doc in ranked_docs:
        if remaining <= 0:
            any_truncated = True
            break
        content, compressed, compression_method = _format_memory_content(paths, doc, query, remaining)
        if not content:
            any_truncated = True
            continue
        truncated = compressed or len(content) < len(doc["content"])
        any_truncated = any_truncated or truncated
        selected.append(
            {
                "memory_id": doc["memory_id"],
                "memory_type": doc["metadata"].get("memory_type"),
                "title": doc["metadata"].get("title", doc["memory_id"]),
                "path": doc["path"],
                "content": content,
                "original_chars": len(doc["content"]),
                "included_chars": len(content),
                "truncated": truncated,
                "compressed": compressed,
                "compression_method": compression_method,
                "rank": doc.get("rank"),
                "score": doc.get("score"),
                "factors": doc.get("factors"),
                "retrieval_mode": doc.get("retrieval_mode"),
                "chunk_index": doc.get("chunk_index"),
                "importance": doc["metadata"].get("importance", config.get("lifecycle", {}).get("importance_default", 5)),
                "flagged": bool(doc["metadata"].get("flagged", False)),
            }
        )
        remaining -= len(content)
    if errors and selected:
        status = "partial"
    elif errors:
        status = "error"
    else:
        status = "success"
    result = {
        "status": status,
        "query": query,
        "selected_memory_docs": selected,
        "retrieval": diagnostics,
        "max_memory_chars": paths["max_chars"],
        "total_chars": sum(item["included_chars"] for item in selected),
        "truncated": any_truncated,
        "errors": errors,
    }
    timestamp = now_iso()
    if selected:
        _update_access_metadata(index, selected, timestamp)
        write_json(index, paths["index"])
    if outdir:
        output_dir = Path(outdir)
        write_json(result, output_dir / "selected_memory.json")
        append_jsonl(
            {
                "timestamp": timestamp,
                "operation": "load",
                "status": status,
                "query": query,
                "retrieval": diagnostics,
                "selected_ids": [item["memory_id"] for item in selected],
                "errors": errors,
            },
            output_dir / "memory_log.jsonl",
        )
    return result

def _detect_conflict(left: str, right: str, threshold: float) -> tuple[bool, str]:
    similarity = _text_similarity(left, right)
    if similarity < threshold:
        return False, "low_overlap"
    negators = {"不", "不是", "不会", "不能", "错误", "失败", "false", "not", "never", "cannot"}
    left_tokens = set(_tokenize(left))
    right_tokens = set(_tokenize(right))
    left_neg = any(term in left.casefold() for term in negators)
    right_neg = any(term in right.casefold() for term in negators)
    shared = len(left_tokens & right_tokens)
    if shared >= 3 and left_neg != right_neg:
        return True, "negation_mismatch"
    return False, "no_rule_conflict"


def _poison_gate(paths: dict, index: dict, save_type: str, answer: str) -> dict:
    config = paths["config"]
    gate = config.get("poison_gate", {})
    if not gate.get("enabled", False):
        return {"flagged": False, "action": "allow", "reason": "disabled", "matches": []}
    trusted_types = set(gate.get("trusted_memory_types", ["global"]))
    trusted_ids = [memory_id for memory_id, item in index.items() if item.get("memory_type") in trusted_types]
    trusted_docs, _ = _load_memory_docs(paths, index, trusted_ids)
    matches = []
    threshold = float(gate.get("conflict_threshold", 0.32))
    for doc in trusted_docs:
        conflict, reason = _detect_conflict(doc["content"], answer, threshold)
        if conflict:
            matches.append({"memory_id": doc["memory_id"], "reason": reason})
    if not matches:
        return {"flagged": False, "action": "allow", "reason": "no_conflict", "matches": []}
    action = gate.get("action", "flag")
    if save_type == "global":
        action = "flag"
    return {"flagged": True, "action": action, "reason": "trusted_memory_conflict", "matches": matches}


def _change_report(existing_markdown: str | None, answer: str, config: dict) -> dict:
    if not existing_markdown:
        return {"change_type": "new", "duplicate": False, "conflict": False, "similarity": None, "notes": []}
    previous_answer = _section(existing_markdown, "Final Answer") or existing_markdown[:2000]
    similarity = _text_similarity(previous_answer, answer)
    duplicate_threshold = float(config.get("integration", {}).get("duplicate_similarity", 0.92))
    conflict_threshold = float(config.get("integration", {}).get("conflict_similarity", 0.28))
    conflict, reason = _detect_conflict(previous_answer, answer, conflict_threshold)
    if similarity >= duplicate_threshold:
        change_type = "duplicate"
    elif conflict:
        change_type = "conflict"
    else:
        change_type = "supplement"
    return {
        "change_type": change_type,
        "duplicate": change_type == "duplicate",
        "conflict": conflict,
        "conflict_reason": reason,
        "similarity": round(similarity, 6),
        "notes": [
            "duplicate content discarded" if change_type == "duplicate" else "new content merged into memory document",
        ],
    }


def _memory_markdown(
    title: str,
    memory_id: str,
    conversation_id: str,
    timestamp: str,
    answer: str,
    messages: list,
    trace: dict,
    change_report: dict,
    previous_answer: str | None,
    poison: dict,
) -> str:
    previous_section = ""
    if previous_answer and change_report.get("change_type") in {"supplement", "conflict"}:
        previous_section = f"\n## Previous Answer\n\n{previous_answer}\n"
    return (
        f"# {title}\n\n"
        f"- memory_id: `{memory_id}`\n"
        f"- conversation_id: `{conversation_id}`\n"
        f"- created_or_updated_at: `{timestamp}`\n"
        f"- flagged: `{str(bool(poison.get('flagged'))).lower()}`\n\n"
        "## Change Report\n\n```json\n"
        f"{json.dumps(change_report, ensure_ascii=False, indent=2)}\n```\n"
        f"{previous_section}\n"
        "## Final Answer\n\n"
        f"{answer}\n\n"
        "## Messages\n\n```json\n"
        f"{json.dumps(messages, ensure_ascii=False, indent=2)}\n```\n\n"
        "## Trace\n\n```json\n"
        f"{json.dumps(trace, ensure_ascii=False, indent=2)}\n```\n"
    )


def _evict_if_needed(paths: dict, index: dict) -> list[dict]:
    lifecycle = paths["config"].get("lifecycle", {})
    max_memories = int(lifecycle.get("max_memories", 200))
    if max_memories <= 0 or len(index) <= max_memories:
        return []
    protect_global = bool(lifecycle.get("protect_global", True))
    now = datetime.now(timezone.utc)
    evictable = []
    for memory_id, metadata in index.items():
        if metadata.get("pinned"):
            continue
        if protect_global and metadata.get("memory_type") == "global":
            continue
        score = _importance_factor(metadata, int(lifecycle.get("importance_default", 5))) * _recency_factor(metadata, now, 30)
        evictable.append((score, memory_id, metadata))
    evicted = []
    for _, memory_id, metadata in sorted(evictable, key=lambda item: (item[0], item[1])):
        if len(index) <= max_memories:
            break
        relative_path = metadata.get("path")
        if isinstance(relative_path, str):
            document = (paths["root"] / relative_path).resolve()
            try:
                document.relative_to(paths["root"].resolve())
            except ValueError:
                document = None
            if document and document.exists():
                document.unlink()
        evicted.append({"memory_id": memory_id, "reason": "capacity", "score": round(float(_), 6)})
        index.pop(memory_id, None)
    return evicted


def _reflect_if_needed(paths: dict, index: dict, timestamp: str) -> dict | None:
    lifecycle = paths["config"].get("lifecycle", {})
    if not lifecycle.get("reflect_enabled", False):
        return None
    conversation_items = [item for item in index.values() if item.get("memory_type") == "conversation"]
    every_n = int(lifecycle.get("reflect_every_n", 10))
    if every_n <= 0 or len(conversation_items) < every_n or len(conversation_items) % every_n != 0:
        return None
    recent = sorted(conversation_items, key=lambda item: item.get("updated_at", ""))[-every_n:]
    summaries = [item.get("summary", "") for item in recent if item.get("summary")]
    if len(summaries) < int(lifecycle.get("reflect_min_cluster_size", 3)):
        return None
    insight_text, method = _compress_text(
        paths["config_path"],
        paths["config"],
        "\n".join(f"- {summary}" for summary in summaries),
        int(paths["config"].get("compression", {}).get("summary_chars", 700)),
        "cross conversation reflection",
    )
    digest = hashlib.blake2b(insight_text.encode("utf-8"), digest_size=6).hexdigest()
    memory_id = f"mem_global_reflect_{digest}"
    path = paths["global"] / f"reflect_{digest}.md"
    relative = f"global/reflect_{digest}.md"
    title = f"Reflection {digest}"
    markdown = (
        f"# {title}\n\n"
        f"- memory_id: `{memory_id}`\n"
        f"- created_or_updated_at: `{timestamp}`\n"
        f"- source_count: `{len(recent)}`\n"
        f"- compression_method: `{method}`\n\n"
        "## Insight\n\n"
        f"{insight_text}\n"
    )
    write_text(markdown, path)
    index[memory_id] = {
        "memory_id": memory_id,
        "memory_type": "global",
        "title": title,
        "summary": insight_text[:300],
        "path": relative,
        "conversation_id": None,
        "importance": 7,
        "created_at": timestamp,
        "updated_at": timestamp,
        "source_memory_ids": [item["memory_id"] for item in recent],
    }
    return {"memory_id": memory_id, "path": relative, "method": method}


def save_memory(
    config_path: str,
    conversation_id: str,
    save_type: str,
    messages_path: str,
    trace_path: str,
    answer_path: str,
    outdir: str | None = None,
) -> dict:
    conversation_id = _safe_conversation_id(conversation_id)
    if save_type not in {"conversation", "global"}:
        raise ValueError("save_type must be conversation or global")
    paths = _memory_paths(config_path)
    config = paths["config"]
    messages = read_json(messages_path)
    trace = read_json(trace_path)
    answer = read_text(answer_path).strip()
    if not isinstance(messages, list) or not isinstance(trace, dict):
        raise ValueError("messages must be an array and trace must be an object")
    timestamp = now_iso()
    memory_id = f"mem_{save_type}_{conversation_id}"
    target_dir = paths["conversations"] if save_type == "conversation" else paths["global"]
    relative_dir = "conversations" if save_type == "conversation" else "global"
    target_path = Path(target_dir) / f"{conversation_id}.md"
    relative_path = f"{relative_dir}/{conversation_id}.md"
    title = f"{save_type.title()} {conversation_id}"
    summary, summary_method = _summarize_for_index(paths["config_path"], config, answer)
    importance, importance_method = _estimate_importance(paths["config_path"], config, answer, trace)
    index = _read_index(paths["index"])
    existing = index.get(memory_id, {})
    existing_markdown = read_text(target_path) if target_path.exists() else None
    previous_answer = _section(existing_markdown, "Final Answer") if existing_markdown else None
    change_report = _change_report(existing_markdown, answer, config)
    poison = _poison_gate(paths, index, save_type, answer)
    if poison.get("action") == "block":
        result = {
            "status": "blocked",
            "memory_id": memory_id,
            "memory_type": save_type,
            "conversation_id": conversation_id,
            "title": title,
            "summary": summary,
            "path": relative_path,
            "change_report": change_report,
            "poison_gate": poison,
            "source_paths": {"messages": str(messages_path), "trace": str(trace_path), "answer": str(answer_path)},
        }
        if outdir:
            output_dir = Path(outdir)
            write_json(result, output_dir / "saved_memory.json")
            append_jsonl({"timestamp": timestamp, "operation": "save", "status": "blocked", "memory_id": memory_id}, output_dir / "memory_log.jsonl")
        return result

    if change_report["change_type"] == "duplicate" and existing_markdown:
        markdown = existing_markdown
    else:
        markdown = _memory_markdown(
            title,
            memory_id,
            conversation_id,
            timestamp,
            answer,
            messages,
            trace,
            change_report,
            previous_answer,
            poison,
        )
        write_text(markdown, target_path)
    created_at = existing.get("created_at", timestamp) if isinstance(existing, dict) else timestamp
    index[memory_id] = {
        "memory_id": memory_id,
        "memory_type": save_type,
        "title": title,
        "summary": summary,
        "summary_method": summary_method,
        "path": relative_path,
        "conversation_id": conversation_id,
        "importance": importance,
        "importance_method": importance_method,
        "flagged": bool(poison.get("flagged", False)),
        "poison_gate": poison,
        "change_report": change_report,
        "created_at": created_at,
        "updated_at": timestamp,
        "last_accessed_at": existing.get("last_accessed_at") if isinstance(existing, dict) else None,
        "access_count": int(existing.get("access_count", 0) or 0) if isinstance(existing, dict) else 0,
    }
    reflection = _reflect_if_needed(paths, index, timestamp)
    evicted = _evict_if_needed(paths, index)
    write_json(index, paths["index"])
    result = {
        "status": "success",
        "memory_id": memory_id,
        "memory_type": save_type,
        "conversation_id": conversation_id,
        "title": title,
        "summary": summary,
        "summary_method": summary_method,
        "importance": importance,
        "importance_method": importance_method,
        "path": relative_path,
        "index_path": Path(paths["index"]).name,
        "created_at": created_at,
        "updated_at": timestamp,
        "change_report": change_report,
        "poison_gate": poison,
        "reflection": reflection,
        "evicted": evicted,
        "source_paths": {
            "messages": str(messages_path),
            "trace": str(trace_path),
            "answer": str(answer_path),
        },
    }
    if outdir:
        output_dir = Path(outdir)
        write_json(result, output_dir / "saved_memory.json")
        append_jsonl(
            {
                "timestamp": timestamp,
                "operation": "save",
                "status": "success",
                "memory_id": memory_id,
                "change_type": change_report["change_type"],
                "flagged": poison.get("flagged", False),
                "reflection": reflection,
                "evicted": evicted,
            },
            output_dir / "memory_log.jsonl",
        )
    return result


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select or save local memory documents.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--select_memory_ids", nargs="*")
    parser.add_argument("--use_global_memory", type=parse_bool)
    parser.add_argument("--query")
    parser.add_argument("--save_type", choices=["conversation", "global"])
    parser.add_argument("--save_input_path")
    parser.add_argument("--outdir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path = resolve_cli_path(args.config)
        outdir = resolve_cli_path(args.outdir)
        if args.save_type or args.save_input_path:
            if not args.save_type or not args.save_input_path:
                raise ValueError("--save_type and --save_input_path must be provided together")
            input_path = resolve_cli_path(args.save_input_path)
            payload = read_json(input_path)
            if payload.get("save_type") != args.save_type:
                raise ValueError("CLI save_type must match memory_save_input.json")
            base = input_path.parent
            save_memory(
                str(config_path),
                payload["conversation_id"],
                args.save_type,
                str((base / payload["messages_path"]).resolve()),
                str((base / payload["trace_path"]).resolve()),
                str((base / payload["answer_path"]).resolve()),
                str(outdir),
            )
            print(outdir / "saved_memory.json")
        else:
            if args.select_memory_ids is None and args.use_global_memory is None:
                raise ValueError("select mode requires --select_memory_ids or --use_global_memory")
            load_memory(
                str(config_path),
                args.select_memory_ids or [],
                bool(args.use_global_memory),
                args.query,
                str(outdir),
            )
            print(outdir / "selected_memory.json")
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
