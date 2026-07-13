from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
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


# 默认配置只放可被 memory.yaml 覆盖的策略参数，便于在不同项目中复用同一套逻辑。
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
        "use_chunk": True,
        "use_three_factor": True,
        # 归一化加权和融合（而非朴素乘法）：RRF 相关度取值范围窄（约 1/(k+rank)），
        # 直接乘 recency/importance 会让相关度被完全淹没，检索退化为"按重要性排序"。
        # three_factor_top_m：时近性/重要性只在相关度前 M 的候选内参与重排，
        # 防止低相关但"新且重要"的记忆从深位跃升。
        "three_factor_weights": {"relevance": 0.8, "recency": 0.1, "importance": 0.1},
        "three_factor_top_m": 10,
        "use_hyde": True,
        "use_rerank": True,
        "hashing_dim": 768,
        "vector_backend": "qwen_or_hashing",
        "cache_embeddings": True,
        "retrieval_cache_path": "memory_retrieval_cache.sqlite3",
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
        "use_llm_judge": True,  # llm.enabled 时用 Qwen judge 做三分类，失败回退规则
    },
    "poison_gate": {
        "enabled": True,
        "action": "flag",
        "trusted_memory_types": ["global"],
        "conflict_threshold": 0.32,
        "use_llm_judge": True,  # llm.enabled 时用 Qwen 做 NLI 式核验，失败回退规则
    },
    "lifecycle": {
        "max_memories": 200,
        "protect_global": True,
        "importance_default": 5,
        "update_access_on_load": True,
        "reflect_enabled": True,
        "reflect_every_n": 10,
        "reflect_min_cluster_size": 3,
        "reflect_similarity_threshold": 0.35,
    },
    "llm": {
        "enabled": True,
        "model_config": "../configs/model.yaml",
        "mode": "prompt_json",
        "max_new_tokens": 512,
        # 按用途拆分生成预算。检索热路径上的 HyDE/rerank 是自回归解码的主要延迟来源，
        # 给"一句话假想段落"配 512 token 纯属浪费（实测 HyDE ~4s/查询）。
        # 未列出的用途回退到 max_new_tokens；greedy 解码遇 EOS 会提前停，上限只防跑飞。
        "token_budgets": {
            "hyde": 48,       # 检索热路径：一句假想段落足够
            "rerank": 160,    # 检索热路径：只需吐 <=10 个 memory_id 的 JSON 数组
            "importance": 8,  # 只需一个 1-10 整数
            "judge": 96,      # 短 JSON 判定（duplicate/supplement/conflict、NLI）
            "compress": 512,  # 摘要/压缩，保质量
            "reflect": 512,   # 跨会话反思综合，保质量
        },
    },
}

# Qwen 模型加载成本很高，按模型配置和调用模式缓存已加载的 torch/tokenizer/model 三元组。
_QWEN_CACHE: dict[tuple[str, str], tuple[Any, Any, Any]] = {}


def _deep_merge(default: dict, override: dict | None) -> dict:
    """递归合并默认配置和用户覆盖配置，保留未显式覆盖的默认项。"""
    # 递归合并配置：只覆盖用户显式设置的叶子节点，未配置项保留 DEFAULTS。
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
    """读取并校验 memory.yaml，返回配置路径和合并后的 memory 配置。"""
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
    """根据 memory 配置解析所有运行时路径和字符预算。"""
    path, memory = _load_memory_config(config_path)
    # 所有相对路径都以配置文件位置为锚点解析，避免受当前工作目录影响。
    root = resolve_from_file(memory["root_dir"], path)
    return {
        "config_path": path,
        "config": memory,
        "root": root,
        "global": root / memory["global_memory_dir"],
        "conversations": root / memory["conversation_memory_dir"],
        "index": root / memory["index_path"],
        "retrieval_cache": root / memory.get("retrieval", {}).get("retrieval_cache_path", "memory_retrieval_cache.sqlite3"),
        "max_chars": memory["max_memory_chars"],
    }


def _read_index(index_path: Path) -> dict:
    """读取 memory 索引文件；不存在时返回空索引。"""
    if not index_path.exists():
        return {}
    index = read_json(index_path)
    if not isinstance(index, dict):
        raise ValueError("memory_index.json must be an object")
    return index


def _open_retrieval_cache(paths: dict) -> sqlite3.Connection:
    """打开并初始化 SQLite 检索缓存，包含文档、chunk 和向量表。"""
    cache_path = Path(paths["retrieval_cache"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(cache_path)
    # WAL 允许读写并行；NORMAL 同步级别适合可重建的缓存数据，减少频繁刷盘开销。
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    # documents 记录失效依据，chunks 保存检索文本，vectors 按后端独立缓存向量。
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            memory_id TEXT PRIMARY KEY,
            source_signature TEXT NOT NULL,
            chunk_signature TEXT NOT NULL,
            path TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            memory_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            PRIMARY KEY (memory_id, chunk_index)
        );
        CREATE TABLE IF NOT EXISTS vectors (
            cache_key TEXT PRIMARY KEY,
            memory_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            backend TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            path TEXT NOT NULL,
            vector TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_vectors_memory_backend
            ON vectors(memory_id, backend);
        CREATE TABLE IF NOT EXISTS query_vectors (
            cache_key TEXT PRIMARY KEY,
            backend TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            vector TEXT NOT NULL
        );
        """
    )
    return connection


def _chunk_config_signature(retrieval: dict) -> str:
    """为切块相关配置生成签名，用于判断 chunk 缓存是否失效。"""
    # 切块参数改变会影响 BM25 和向量排序，因此必须让已有 chunk 整体失效。
    payload = {
        "use_chunk": bool(retrieval.get("use_chunk", True)),
        "chunk_chars": int(retrieval.get("chunk_chars", 700)),
        "chunk_overlap": int(retrieval.get("chunk_overlap", 120)),
        "search_text_version": 2,
    }
    return _content_hash(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _document_source_signature(document_path: Path, metadata: dict) -> str:
    """为单个记忆文档生成源签名，覆盖文件状态和影响检索的元数据。"""
    stat = document_path.stat()
    # 标题和摘要也属于检索文本，不能只根据 Markdown 文件状态判断缓存是否有效。
    payload = {
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
        "title": metadata.get("title"),
        "summary": metadata.get("summary"),
        "retrieval_summary": metadata.get("retrieval_summary"),
        "retrieval_terms": metadata.get("retrieval_terms"),
        "updated_at": metadata.get("updated_at"),
    }
    return _content_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


def _safe_conversation_id(conversation_id: str) -> str:
    """校验 conversation_id 是否只包含可安全组成文件名的字符。"""
    if not isinstance(conversation_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", conversation_id):
        raise ValueError("conversation_id may only contain letters, numbers, dot, underscore, and hyphen")
    return conversation_id


def _parse_time(value: Any) -> datetime | None:
    """解析 ISO 时间字符串，无法解析时返回 None。"""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _age_days(metadata: dict, now: datetime) -> float:
    """根据访问、更新或创建时间估算一条记忆距离现在的天数。"""
    stamp = _parse_time(metadata.get("last_accessed_at")) or _parse_time(metadata.get("updated_at")) or _parse_time(metadata.get("created_at"))
    if stamp is None:
        return 0.0
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return max(0.0, (now - stamp.astimezone(timezone.utc)).total_seconds() / 86400)


def _recency_factor(metadata: dict, now: datetime, half_life_days: float) -> float:
    """把记忆年龄转换为半衰期衰减后的时近性分数。"""
    half_life = max(float(half_life_days), 1.0)
    return 0.5 ** (_age_days(metadata, now) / half_life)


def _importance_factor(metadata: dict, default: int) -> float:
    """把 1-10 的重要性元数据归一化为 0.1-1.0 的权重。"""
    value = metadata.get("importance", default)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(default)
    return min(1.0, max(0.1, numeric / 10.0))


def _memory_document_path(paths: dict, metadata: dict) -> tuple[Path | None, str | None, dict | None]:
    """从索引元数据解析记忆文档路径，并阻止路径逃逸 memory root。"""
    relative_path = metadata.get("path")
    if not isinstance(relative_path, str):
        return None, None, {"type": "InvalidMetadata", "message": "memory path is missing"}
    document_path = (paths["root"] / relative_path).resolve()
    try:
        # 防止索引中的 path 使用 ../ 逃出 memory root 后读取任意文件。
        document_path.relative_to(paths["root"].resolve())
    except ValueError:
        return None, relative_path, {"type": "InvalidPath", "message": "memory path escapes root"}
    if not document_path.is_file():
        return None, relative_path, {"type": "FileNotFoundError", "message": f"memory file not found: {relative_path}"}
    return document_path, relative_path, None


def _load_memory_docs(
    paths: dict,
    index: dict,
    memory_ids: list[str],
    load_content: bool = True,
) -> tuple[list[dict], list[dict]]:
    """按 id 从索引加载记忆元数据和可选正文，同时收集缺失或非法路径错误。"""
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
        doc = {
            "memory_id": memory_id,
            "metadata": metadata,
            "path": relative_path,
            "document_path": document_path,
        }
        if load_content:
            content = read_text(document_path)
            doc.update({"content": content, "search_content": _memory_search_text(metadata, content)})
        docs.append(doc)
    return docs, errors


def _hydrate_memory_docs(docs: list[dict]) -> list[dict]:
    """排序完成后只读取最终入选记忆的完整正文。"""
    hydrated = []
    for doc in docs:
        if "content" in doc:
            hydrated.append(doc)
            continue
        content = read_text(Path(doc["document_path"]))
        hydrated.append(
            {
                **doc,
                "content": content,
                "search_content": _memory_search_text(doc["metadata"], content),
            }
        )
    return hydrated


def _section(markdown: str, heading: str) -> str:
    """提取 Markdown 中指定二级标题下的内容。"""
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
    """构造用于检索的精简文本，优先保留标题、摘要和关键章节。"""
    # 检索文本优先使用标题、摘要和关键正文段落，避免整份 trace/messages 稀释相关性。
    parts = [
        str(metadata.get("title") or ""),
        str(metadata.get("summary") or ""),
        str(metadata.get("retrieval_summary") or ""),
    ]
    terms = metadata.get("retrieval_terms")
    if isinstance(terms, dict):
        for key in ("tags", "entities", "files", "errors", "commands", "tool_names"):
            value = terms.get(key)
            if isinstance(value, list):
                parts.append(" ".join(str(item) for item in value if isinstance(item, str)))
    for heading in ("Final Answer", "Insight", "Previous Answer"):
        section = _section(markdown, heading)
        if section:
            parts.append(section)
    change_report = _section(markdown, "Change Report")
    if change_report:
        parts.append(change_report[:1200])
    text = "\n\n".join(part for part in parts if part.strip()).strip()
    return text or markdown


def _iter_strings(value: Any, limit: int = 300) -> list[str]:
    """从嵌套 list/dict 中抽取字符串，限制数量避免 trace 过大拖慢保存。"""
    items: list[str] = []

    def visit(node: Any) -> None:
        if len(items) >= limit:
            return
        if isinstance(node, str):
            if node.strip():
                items.append(node.strip())
            return
        if isinstance(node, dict):
            for child in node.values():
                visit(child)
            return
        if isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return items


def _collect_named_values(value: Any, names: set[str], limit: int = 80) -> list[str]:
    """从嵌套 dict/list 中收集指定字段名下的字符串值。"""
    items: list[str] = []

    def add(node: Any) -> None:
        if len(items) >= limit:
            return
        if isinstance(node, str) and node.strip():
            items.append(node.strip())
        elif isinstance(node, list):
            for child in node:
                add(child)

    def visit(node: Any) -> None:
        if len(items) >= limit:
            return
        if isinstance(node, dict):
            for key, child in node.items():
                if isinstance(key, str) and key in names:
                    add(child)
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return items


def _dedupe_limited(values: list[str], limit: int, max_chars: int = 160) -> list[str]:
    """按出现顺序去重并截断字段，保证检索 metadata 小而稳定。"""
    selected = []
    seen = set()
    for value in values:
        cleaned = re.sub(r"\s+", " ", str(value)).strip()
        if not cleaned:
            continue
        cleaned = cleaned[:max_chars]
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        selected.append(cleaned)
        if len(selected) >= limit:
            break
    return selected


def _build_retrieval_metadata(answer: str, messages: list, trace: dict) -> dict:
    """从保存输入中抽取检索友好的结构化 signals，供后续 BM25/向量召回使用。"""
    combined = "\n".join([answer, *_iter_strings(messages), *_iter_strings(trace)])
    file_pattern = r"\b[\w./-]+\.(?:py|ya?ml|json|md|txt|csv|sqlite3|db|html|css|js|ts|tsx|jsx|sh|log)\b"
    files = _dedupe_limited(re.findall(file_pattern, combined), 20)
    errors = _dedupe_limited(
        re.findall(
            r"\b[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)\b|CUDA out of memory|out of memory|OOM",
            combined,
            flags=re.IGNORECASE,
        ),
        16,
    )
    commands = _dedupe_limited(
        re.findall(r"\b(?:python3?|pip|conda|git|pytest|bash|sh|npm|node)\s+[^\n`。；;]{1,140}", combined),
        12,
    )
    identifiers = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", combined)
    entities = _dedupe_limited(
        [
            item
            for item in identifiers
            if "_" in item
            or any(char.isdigit() for char in item)
            or any(char.isupper() for char in item[1:])
        ],
        32,
        80,
    )
    tool_names = _dedupe_limited(
        [
            item
            for item in _collect_named_values(trace, {"tool", "tool_name", "tool_names", "name"})
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{2,}", item)
        ],
        16,
        80,
    )
    tags = _dedupe_limited([*errors, *entities[:12], *files[:8]], 30, 80)
    retrieval_summary = "; ".join([*tags, *commands[:4]])[:700]
    return {
        "retrieval_summary": retrieval_summary,
        "retrieval_terms": {
            "tags": tags,
            "entities": entities,
            "files": files,
            "errors": errors,
            "commands": commands,
            "tool_names": tool_names,
        },
    }


def _tokenize(text: str) -> list[str]:
    """把中英文混合文本切成检索 token，中文额外加入单字和相邻双字。"""
    lowered = text.casefold()
    tokens = re.findall(r"[a-z0-9_]+", lowered)
    # 中文没有天然空格分词，这里加入单字和相邻双字，兼顾召回率和简单实现。
    cjk = re.findall(r"[\u4e00-\u9fff]", lowered)
    tokens.extend(cjk)
    tokens.extend("".join(pair) for pair in zip(cjk, cjk[1:]))
    return [token for token in tokens if token.strip()]


def _sentences(text: str) -> list[str]:
    """按常见中英文句末符号和换行切分句子。"""
    raw = re.split(r"(?<=[。！？!?；;])|\n+", text)
    return [item.strip() for item in raw if item and item.strip()]


def _hash_vector(tokens: list[str], dim: int) -> dict[int, float]:
    """用 signed hashing trick 把 token 序列转换成归一化稀疏向量。"""
    # 使用 signed hashing trick 构造稀疏向量，作为无模型环境下的稳定向量后端。
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
    """计算两个稀疏向量字典的余弦相似度点积。"""
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(index, 0.0) for index, value in left.items())


def _text_similarity(left: str, right: str, dim: int = 512) -> float:
    """基于 hashing 向量快速估计两段文本的相似度。"""
    return _cosine_dict(_hash_vector(_tokenize(left), dim), _hash_vector(_tokenize(right), dim))


def _chunk_text(text: str, chunk_chars: int, overlap: int) -> list[dict]:
    """按字符预算和 overlap 把长文本切成可检索 chunk。"""
    chunk_chars = max(160, int(chunk_chars))
    overlap = max(0, min(int(overlap), chunk_chars // 2))
    paragraphs = [item.strip() for item in re.split(r"\n{2,}", text) if item.strip()]
    chunks = []
    current = ""
    start = 0
    for paragraph in paragraphs or [text]:
        if len(paragraph) > chunk_chars:
            # 单段过长时直接滑窗切分，避免一个长段突破候选长度上限。
            step = max(1, chunk_chars - overlap)
            for offset in range(0, len(paragraph), step):
                piece = paragraph[offset : offset + chunk_chars].strip()
                if piece:
                    chunks.append({"chunk_index": len(chunks), "content": piece, "start": offset, "end": offset + len(piece)})
            continue
        if current and len(current) + len(paragraph) + 2 > chunk_chars:
            chunks.append({"chunk_index": len(chunks), "content": current.strip(), "start": start, "end": start + len(current)})
            # 保留尾部 overlap，减少答案刚好落在分块边界时的召回损失。
            tail = current[-overlap:] if overlap else ""
            current = (tail + "\n\n" + paragraph).strip() if tail else paragraph
            start += max(0, len(current) - len(tail))
        else:
            current = (current + "\n\n" + paragraph).strip() if current else paragraph
    if current:
        chunks.append({"chunk_index": len(chunks), "content": current.strip(), "start": start, "end": start + len(current)})
    return chunks or [{"chunk_index": 0, "content": text[:chunk_chars], "start": 0, "end": min(len(text), chunk_chars)}]


def _bm25_scores(query: str, chunks: list[dict]) -> dict[int, float]:
    """为查询和 chunk 列表计算轻量 BM25 关键词分数。"""
    # 轻量 BM25 用于关键词召回；分数只在当前候选 chunks 内比较。
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
    """把分数字典转换为从 1 开始的排名字典。"""
    return {item: rank for rank, (item, _) in enumerate(sorted(scores.items(), key=lambda pair: (-pair[1], pair[0])), 1)}


def _minmax_scores(scores: dict[int, float]) -> dict[int, float]:
    """对分数字典做 min-max 归一化，空输入返回空字典。"""
    if not scores:
        return {}
    low = min(scores.values())
    high = max(scores.values())
    span = high - low
    return {key: (value - low) / span if span > 0 else 1.0 for key, value in scores.items()}


def _qwen_paths(config_path: Path, memory_config: dict) -> tuple[Path | None, Path | None]:
    """从 memory 配置间接解析 Qwen 模型和 tokenizer 的本地路径。"""
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
    """把配置中的 dtype 字符串映射为 torch dtype 或 auto。"""
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
    """加载并缓存 Qwen 运行所需的 torch、tokenizer 和模型实例。"""
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
    # 默认 local_files_only=True，避免检索/保存记忆时意外联网下载模型。
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


def _token_budget(memory_config: dict, purpose: str) -> int:
    """按用途读取 LLM 生成 token 预算，配置非法时回退默认值。"""
    llm = memory_config.get("llm", {})
    default = int(llm.get("max_new_tokens", 512))
    budgets = llm.get("token_budgets") or {}
    value = budgets.get(purpose, default)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _qwen_generate(config_path: Path, memory_config: dict, prompt: str, max_new_tokens: int | None = None) -> str:
    """使用本地 Qwen 以 greedy 方式生成文本，返回去除特殊符号后的结果。"""
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
    limit = int(max_new_tokens) if max_new_tokens is not None else int(memory_config.get("llm", {}).get("max_new_tokens", 512))
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max(1, limit),
            do_sample=False,
        )
    return tokenizer.decode(generated[0][input_length:], skip_special_tokens=True).strip()


def _qwen_embedding(config_path: Path, memory_config: dict, text: str) -> dict[int, float]:
    """用 Qwen 最后一层 hidden state 的 mean pooling 生成稀疏字典形式向量。"""
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
    # 用 attention mask 做 mean pooling，忽略 padding 后再归一化为余弦相似度友好的向量。
    pooled = (hidden * mask).sum(dim=0) / mask.sum().clamp(min=1)
    pooled = torch.nn.functional.normalize(pooled.float(), dim=0)
    values = pooled.detach().cpu().tolist()
    return {index: float(value) for index, value in enumerate(values) if abs(float(value)) > 1e-8}


def _vectorize(config_path: Path, memory_config: dict, text: str) -> tuple[dict[int, float], str]:
    """按配置选择 Qwen 或 hashing 后端对文本向量化，并返回实际使用的后端名。"""
    retrieval = memory_config.get("retrieval", {})
    backend = retrieval.get("vector_backend", "qwen_or_hashing")
    if backend in {"qwen", "qwen_or_hashing"}:
        try:
            return _qwen_embedding(config_path, memory_config, text), "qwen"
        except Exception:
            if backend == "qwen":
                raise
    # qwen_or_hashing 模式下，模型不可用时自动退回 hashing，保证 memory 功能不中断。
    return _hash_vector(_tokenize(text), int(retrieval.get("hashing_dim", 768))), "hashing"


def _content_hash(text: str) -> str:
    """生成短 blake2b 内容哈希，用于缓存键和变更检测。"""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _preferred_vector_backend(memory_config: dict) -> str:
    """根据配置和 LLM 开关推导检索缓存应优先使用的向量后端。"""
    retrieval = memory_config.get("retrieval", {})
    backend = str(retrieval.get("vector_backend", "qwen_or_hashing"))
    if backend == "qwen_or_hashing":
        return "qwen" if memory_config.get("llm", {}).get("enabled", False) else "hashing"
    return backend


def _serialize_vector(vector: dict[int, float]) -> list[list[float]]:
    """把稀疏向量字典转换为稳定排序的 JSON 可序列化列表。"""
    return [[int(index), float(value)] for index, value in sorted(vector.items())]


def _deserialize_vector(raw: Any) -> dict[int, float]:
    """把 JSON 缓存中的向量列表还原为稀疏字典，跳过异常项。"""
    if not isinstance(raw, list):
        return {}
    vector = {}
    for item in raw:
        if not isinstance(item, list) or len(item) != 2:
            continue
        try:
            index = int(item[0])
            value = float(item[1])
        except (TypeError, ValueError):
            continue
        vector[index] = value
    return vector


def _chunk_cache_key(chunk: dict, backend: str, source_hash: str) -> str:
    """构造单个 chunk 在指定后端和内容版本下的向量缓存键。"""
    return f"{chunk['memory_id']}:{chunk['chunk_index']}:{backend}:{source_hash}"


def _query_vector_cache_key(backend: str, source_hash: str) -> str:
    """构造查询文本在指定后端和内容版本下的向量缓存键。"""
    return f"query:{backend}:{source_hash}"


def _cached_chunk_vectorize(
    paths: dict,
    cache: dict,
    chunk: dict,
    diagnostics: dict,
    backend_override: str | None = None,
) -> tuple[dict[int, float], str]:
    """读取或生成 chunk 向量，并同步更新本轮缓存诊断计数。"""
    config = paths["config"]
    retrieval = config.get("retrieval", {})
    if not retrieval.get("cache_embeddings", True):
        # 关闭缓存时仍记录诊断信息，方便定位检索耗时来源。
        vector, backend = _vectorize_with_backend(paths["config_path"], config, chunk["content"], backend_override)
        diagnostics["embedding_cache_disabled"] = diagnostics.get("embedding_cache_disabled", 0) + 1
        return vector, backend

    source_hash = _content_hash(chunk["content"])
    preferred_backend = backend_override or _preferred_vector_backend(config)
    vectors = cache.setdefault("vectors", {})
    preferred_key = _chunk_cache_key(chunk, preferred_backend, source_hash)
    cached = vectors.get(preferred_key)
    if isinstance(cached, dict):
        vector = _deserialize_vector(cached.get("vector"))
        if vector:
            # 缓存键包含内容 hash，记忆正文变化后会自然失效。
            diagnostics["embedding_cache_hits"] = diagnostics.get("embedding_cache_hits", 0) + 1
            return vector, str(cached.get("backend", preferred_backend))

    vector, backend = _vectorize_with_backend(paths["config_path"], config, chunk["content"], backend_override)
    key = _chunk_cache_key(chunk, backend, source_hash)
    vectors[key] = {
        "memory_id": chunk["memory_id"],
        "chunk_index": chunk["chunk_index"],
        "backend": backend,
        "source_hash": source_hash,
        "path": chunk["path"],
        "vector": _serialize_vector(vector),
    }
    cache.setdefault("_dirty_keys", set()).add(key)
    diagnostics["embedding_cache_misses"] = diagnostics.get("embedding_cache_misses", 0) + 1
    return vector, backend


def _cached_query_vectorize(
    paths: dict,
    connection: sqlite3.Connection,
    text: str,
    diagnostics: dict,
) -> tuple[dict[int, float], str]:
    """读取或生成 query 向量；query cache 不依赖 memory 内容，memory 更新无需失效。"""
    config = paths["config"]
    retrieval = config.get("retrieval", {})
    if not retrieval.get("cache_embeddings", True):
        vector, backend = _vectorize(paths["config_path"], config, text)
        diagnostics["query_embedding_cache_disabled"] = diagnostics.get("query_embedding_cache_disabled", 0) + 1
        return vector, backend

    source_hash = _content_hash(text)
    preferred_backend = _preferred_vector_backend(config)
    preferred_key = _query_vector_cache_key(preferred_backend, source_hash)
    cached = connection.execute(
        "SELECT backend, vector FROM query_vectors WHERE cache_key = ?",
        (preferred_key,),
    ).fetchone()
    if cached:
        try:
            vector = _deserialize_vector(json.loads(cached[1]))
        except json.JSONDecodeError:
            vector = {}
        if vector:
            diagnostics["query_embedding_cache_hits"] = diagnostics.get("query_embedding_cache_hits", 0) + 1
            return vector, str(cached[0])

    vector, backend = _vectorize(paths["config_path"], config, text)
    key = _query_vector_cache_key(backend, source_hash)
    connection.execute(
        """
        INSERT OR REPLACE INTO query_vectors(cache_key, backend, source_hash, vector)
        VALUES (?, ?, ?, ?)
        """,
        (key, backend, source_hash, json.dumps(_serialize_vector(vector), separators=(",", ":"))),
    )
    connection.commit()
    diagnostics["query_embedding_cache_misses"] = diagnostics.get("query_embedding_cache_misses", 0) + 1
    return vector, backend


def _vectorize_with_backend(
    config_path: Path,
    memory_config: dict,
    text: str,
    backend_override: str | None,
) -> tuple[dict[int, float], str]:
    """在指定后端存在时强制使用该后端，否则按全局配置自动选择。"""
    if backend_override == "hashing":
        dim = int(memory_config.get("retrieval", {}).get("hashing_dim", 768))
        return _hash_vector(_tokenize(text), dim), "hashing"
    if backend_override == "qwen":
        return _qwen_embedding(config_path, memory_config, text), "qwen"
    return _vectorize(config_path, memory_config, text)


def _hyde_query(config_path: Path, memory_config: dict, query: str) -> tuple[str, str]:
    """按 HyDE 策略扩展查询，返回增强后的查询文本和实际模式。"""
    retrieval = memory_config.get("retrieval", {})
    if not retrieval.get("use_hyde", False):
        return query, "disabled"
    try:
        # HyDE 生成一段假想答案，把短查询扩展成更接近 memory 文档的检索文本。
        prompt = (
            "Write one concise hypothetical memory passage that would answer this query. "
            "Return only the passage.\n\n"
            f"Query: {query}"
        )
        generated = _qwen_generate(config_path, memory_config, prompt, _token_budget(memory_config, "hyde"))
        if generated:
            return f"{query}\n{generated}", "qwen"
    except Exception:
        pass
    # LLM 不可用或输出为空时回退原始查询，不让增强检索影响基本功能。
    return query, "fallback"


def _extractive_summary(text: str, budget: int, query: str | None = None, max_sentences: int = 5) -> str:
    """用可解释的抽取式策略在字符预算内生成摘要。"""
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
        # 兼顾查询覆盖、靠前句子和句长，作为 LLM 摘要失败时的可解释兜底。
        score = overlap * 3 + position_bonus - length_penalty
        scored.append((score, index, sentence))
    selected = sorted(scored, key=lambda item: (-item[0], item[1]))[:max_sentences]
    selected.sort(key=lambda item: item[1])
    summary = "\n".join(item[2] for item in selected).strip()
    if len(summary) > budget:
        summary = summary[:budget].rstrip()
    return summary


def _compress_text(config_path: Path, memory_config: dict, text: str, budget: int, query: str | None = None) -> tuple[str, str]:
    """在字符预算内压缩文本，优先 LLM 摘要，失败时抽取式兜底。"""
    compression = memory_config.get("compression", {})
    if not compression.get("enabled", True) or len(text) <= budget:
        return text[:budget], "none" if len(text) <= budget else "hard_truncate"
    try:
        prompt = (
            "Summarize the memory below for future retrieval. Preserve concrete facts, decisions, "
            "tool results, conflicts, and user preferences; copy identifiers, numbers, file names, "
            "parameters, and error codes verbatim. Write in the same language as the memory. "
            "Stay within the character budget. Return plain text only.\n\n"
            f"Budget characters: {budget}\n"
            f"Query focus: {query or ''}\n\n"
            f"Memory:\n{text[:6000]}"
        )
        generated = _qwen_generate(config_path, memory_config, prompt, _token_budget(memory_config, "compress"))
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
    """为索引元数据生成短摘要，限制在摘要预算和 300 字符以内。"""
    budget = int(memory_config.get("compression", {}).get("summary_chars", 700))
    return _compress_text(config_path, memory_config, answer, min(300, budget), None)


def _estimate_importance(config_path: Path, memory_config: dict, answer: str, trace: dict) -> tuple[int, str]:
    """估计记忆长期价值，优先 LLM 打分，失败时使用启发式规则。"""
    try:
        prompt = (
            "Rate this memory importance from 1 to 10 for a local tool-using agent. "
            "Return only an integer.\n\n"
            f"Final answer:\n{answer[:3000]}"
        )
        generated = _qwen_generate(config_path, memory_config, prompt, _token_budget(memory_config, "importance"))
        match = re.search(r"\b([1-9]|10)\b", generated)
        if match:
            return int(match.group(1)), "qwen"
    except Exception:
        pass
    # 无 LLM 时用长度、工具调用和问题关键词粗略估计长期记忆价值。
    score = 4
    if len(answer) > 240:
        score += 1
    if trace.get("tool_rounds_used", 0):
        score += 1
    if any(term in answer.casefold() for term in ["error", "错误", "失败", "配置", "路径", "工具", "memory", "agent"]):
        score += 1
    return min(10, max(1, score)), "heuristic"


def _build_chunks(paths: dict, docs: list[dict]) -> list[dict]:
    """把已加载正文的记忆文档转换成检索 chunk 列表。"""
    retrieval = paths["config"].get("retrieval", {})
    if not retrieval.get("use_chunk", True):
        return [
            {
                "memory_id": doc["memory_id"],
                "metadata": doc["metadata"],
                "path": doc["path"],
                "chunk_index": 0,
                "content": doc.get("search_content") or doc["content"],
                "full_content": doc["content"],
            }
            for doc in docs
        ]
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


def _load_or_build_chunks(
    paths: dict,
    docs: list[dict],
    connection: sqlite3.Connection,
) -> tuple[list[dict], dict]:
    """从持久化缓存读取 chunk，仅在文档或切块配置变化时重建。"""
    retrieval = paths["config"].get("retrieval", {})
    chunk_signature = _chunk_config_signature(retrieval)
    chunks = []
    hits = 0
    misses = 0
    for doc in docs:
        source_signature = _document_source_signature(Path(doc["document_path"]), doc["metadata"])
        cached_doc = connection.execute(
            "SELECT source_signature, chunk_signature, path FROM documents WHERE memory_id = ?",
            (doc["memory_id"],),
        ).fetchone()
        cached_rows = []
        # 两类签名及路径都一致时，可跳过 Markdown 读取、搜索文本提取和重新切块。
        if cached_doc == (source_signature, chunk_signature, doc["path"]):
            cached_rows = connection.execute(
                "SELECT chunk_index, content, source_hash FROM chunks WHERE memory_id = ? ORDER BY chunk_index",
                (doc["memory_id"],),
            ).fetchall()
        if cached_rows:
            hits += 1
            doc_chunks = [
                {"chunk_index": row[0], "content": row[1], "source_hash": row[2]}
                for row in cached_rows
            ]
        else:
            misses += 1
            # 缓存未命中时才读取完整正文，并沿用原有搜索文本和切块算法重建数据。
            content = read_text(Path(doc["document_path"]))
            search_content = _memory_search_text(doc["metadata"], content)
            if retrieval.get("use_chunk", True):
                raw_chunks = _chunk_text(
                    search_content,
                    int(retrieval.get("chunk_chars", 700)),
                    int(retrieval.get("chunk_overlap", 120)),
                )
            else:
                raw_chunks = [{"chunk_index": 0, "content": search_content}]
            doc_chunks = [
                {
                    "chunk_index": int(item["chunk_index"]),
                    "content": item["content"],
                    "source_hash": _content_hash(item["content"]),
                }
                for item in raw_chunks
            ]
            connection.execute("DELETE FROM chunks WHERE memory_id = ?", (doc["memory_id"],))
            connection.executemany(
                "INSERT INTO chunks(memory_id, chunk_index, content, source_hash) VALUES (?, ?, ?, ?)",
                [
                    (doc["memory_id"], item["chunk_index"], item["content"], item["source_hash"])
                    for item in doc_chunks
                ],
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO documents(memory_id, source_signature, chunk_signature, path)
                VALUES (?, ?, ?, ?)
                """,
                (doc["memory_id"], source_signature, chunk_signature, doc["path"]),
            )
            # 仅删除无法对应新 chunk 的旧向量；内容 hash 未变的向量仍可继续复用。
            connection.execute(
                """
                DELETE FROM vectors
                WHERE memory_id = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM chunks
                      WHERE chunks.memory_id = vectors.memory_id
                        AND chunks.chunk_index = vectors.chunk_index
                        AND chunks.source_hash = vectors.source_hash
                  )
                """,
                (doc["memory_id"],),
            )
        for item in doc_chunks:
            chunks.append(
                {
                    "memory_id": doc["memory_id"],
                    "metadata": doc["metadata"],
                    "path": doc["path"],
                    "chunk_index": item["chunk_index"],
                    "content": item["content"],
                    "source_hash": item["source_hash"],
                }
            )
    connection.commit()
    return chunks, {"hits": hits, "misses": misses}


def _read_sqlite_vectors(
    connection: sqlite3.Connection,
    chunks: list[dict],
    backend: str,
) -> dict:
    """按候选 memory_id 批量读取 SQLite 向量缓存。"""
    memory_ids = list(dict.fromkeys(chunk["memory_id"] for chunk in chunks))
    if not memory_ids:
        return {"version": 2, "vectors": {}}
    rows = []
    # SQLite 默认变量数有限，大语料按批查询避免候选超过上限。
    for offset in range(0, len(memory_ids), 900):
        batch = memory_ids[offset : offset + 900]
        placeholders = ",".join("?" for _ in batch)
        rows.extend(
            connection.execute(
                f"""
                SELECT cache_key, memory_id, chunk_index, backend, source_hash, path, vector
                FROM vectors
                WHERE backend = ? AND memory_id IN ({placeholders})
                """,
                [backend, *batch],
            ).fetchall()
        )
    vectors = {}
    for cache_key, memory_id, chunk_index, used_backend, source_hash, path, raw_vector in rows:
        try:
            serialized = json.loads(raw_vector)
        except json.JSONDecodeError:
            continue
        vectors[cache_key] = {
            "memory_id": memory_id,
            "chunk_index": chunk_index,
            "backend": used_backend,
            "source_hash": source_hash,
            "path": path,
            "vector": serialized,
        }
    return {"version": 2, "vectors": vectors}


def _write_sqlite_vectors(connection: sqlite3.Connection, cache: dict) -> None:
    """把本轮新生成或更新的 chunk 向量写回 SQLite 缓存。"""
    rows = []
    dirty_keys = cache.get("_dirty_keys")
    for cache_key, item in cache.get("vectors", {}).items():
        if dirty_keys is not None and cache_key not in dirty_keys:
            continue
        rows.append(
            (
                cache_key,
                item["memory_id"],
                int(item["chunk_index"]),
                item["backend"],
                item["source_hash"],
                item["path"],
                json.dumps(item["vector"], separators=(",", ":")),
            )
        )
    if not rows:
        return
    # 批量 upsert 比逐条提交显著减少事务和磁盘同步次数。
    connection.executemany(
        """
        INSERT OR REPLACE INTO vectors
            (cache_key, memory_id, chunk_index, backend, source_hash, path, vector)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    connection.commit()


def _warm_retrieval_cache(paths: dict, index: dict, memory_ids: list[str]) -> dict:
    """保存记忆时提前完成切块和向量化，把计算成本移出检索热路径。"""
    docs, errors = _load_memory_docs(paths, index, memory_ids, load_content=False)
    if errors or not docs:
        return {"status": "skipped", "errors": errors}
    connection = _open_retrieval_cache(paths)
    try:
        chunks, chunk_diagnostics = _load_or_build_chunks(paths, docs, connection)
        retrieval = paths["config"].get("retrieval", {})
        mode = str(retrieval.get("mode", "none"))
        vector_diagnostics = {"embedding_cache_hits": 0, "embedding_cache_misses": 0, "embedding_cache_disabled": 0}
        backend = None
        if mode in {"vector", "hybrid"} and retrieval.get("cache_embeddings", True) and chunks:
            # 写入阶段使用与检索阶段相同的后端，保证预热向量可以直接命中。
            backend = _preferred_vector_backend(paths["config"])
            cache = _read_sqlite_vectors(connection, chunks, backend)
            for chunk in chunks:
                _cached_chunk_vectorize(paths, cache, chunk, vector_diagnostics, backend)
            if vector_diagnostics["embedding_cache_misses"]:
                _write_sqlite_vectors(connection, cache)
        return {
            "status": "success",
            "chunk_cache": chunk_diagnostics,
            "vector_backend": backend,
            "embedding_cache": vector_diagnostics,
        }
    finally:
        connection.close()


def _candidate_ids_for_load(index: dict, selected_memory_ids: list[str], use_global_memory: bool, query: str | None, config: dict) -> list[str]:
    """生成 load 阶段的候选 id 列表；显式选择的记忆总会追加到候选末尾。"""
    ordered = []
    retrieval = config.get("retrieval", {})
    mode = retrieval.get("mode", "none")
    if query and mode != "none":
        # 有查询时扩大候选池，由后续排序决定真正进入上下文的记忆。
        if retrieval.get("include_global_when_query", True):
            ordered.extend(sorted(key for key, item in index.items() if item.get("memory_type") == "global"))
        if retrieval.get("include_all_conversations", True):
            ordered.extend(sorted(key for key, item in index.items() if item.get("memory_type") == "conversation"))
    else:
        if use_global_memory:
            ordered.extend(sorted(key for key, item in index.items() if item.get("memory_type") == "global"))
    ordered.extend(selected_memory_ids)
    # dict.fromkeys 保序去重，确保显式选择的 id 不会重复加载。
    return list(dict.fromkeys(ordered))


def _rank_memory_docs(paths: dict, docs: list[dict], query: str | None) -> tuple[list[dict], dict]:
    """对候选记忆排序，并返回可写入日志的检索诊断信息。"""
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
    # 热路径只从 SQLite 读取当前候选的 chunk/向量，不再解析整份 JSON 缓存。
    connection = _open_retrieval_cache(paths)
    chunks, chunk_cache_diagnostics = _load_or_build_chunks(paths, docs, connection)
    bm25 = _bm25_scores(hyde_query if mode in {"keyword", "hybrid"} else query, chunks) if mode in {"keyword", "hybrid"} else {}
    vector_scores: dict[int, float] = {}
    vector_backend = None
    cache = {"version": 2, "vectors": {}}
    cache_diagnostics = {
        "embedding_cache_hits": 0,
        "embedding_cache_misses": 0,
        "embedding_cache_disabled": 0,
        "query_embedding_cache_hits": 0,
        "query_embedding_cache_misses": 0,
        "query_embedding_cache_disabled": 0,
    }
    if mode in {"vector", "hybrid"}:
        query_vector, vector_backend = _cached_query_vectorize(paths, connection, hyde_query, cache_diagnostics)
        cache = _read_sqlite_vectors(connection, chunks, vector_backend)
        for index, chunk in enumerate(chunks):
            chunk_vector, used_backend = _cached_chunk_vectorize(
                paths,
                cache,
                chunk,
                cache_diagnostics,
                vector_backend,
            )
            vector_backend = vector_backend or used_backend
            score = _cosine_dict(query_vector, chunk_vector)
            if score:
                vector_scores[index] = score
        if config.get("retrieval", {}).get("cache_embeddings", True) and cache_diagnostics["embedding_cache_misses"]:
            _write_sqlite_vectors(connection, cache)
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
            # RRF 融合关键词和向量两路排名，避免直接比较两种不可比的原始分数。
            fused[chunk_index] += 1.0 / (rrf_k + rank)

    now = datetime.now(timezone.utc)
    half_life = float(retrieval.get("recency_half_life_days", 30))
    importance_default = int(config.get("lifecycle", {}).get("importance_default", 5))
    use_three_factor = bool(retrieval.get("use_three_factor", True))
    weights = retrieval.get("three_factor_weights") or {}
    w_rel = float(weights.get("relevance", 0.8))
    w_rec = float(weights.get("recency", 0.1))
    w_imp = float(weights.get("importance", 0.1))
    top_m = int(retrieval.get("three_factor_top_m", 10))
    # 三因子的相关度分量用各路原始分数的 min-max 归一化（保留绝对差距），
    # 不用 RRF 融合分：rank 倒数曲线过平，会让任何 recency/importance 权重都能翻转 top-1。
    bm25_norm = _minmax_scores(bm25)
    vector_norm = _minmax_scores(vector_scores)
    fused_norm = _minmax_scores(dict(fused))
    # 只让 RRF 前 top_m 的 chunk 参与 recency/importance 加权；其余候选按相关度保守排序。
    rescoring_pool = {
        chunk_index
        for chunk_index, _ in sorted(fused.items(), key=lambda pair: -pair[1])[: max(0, top_m)]
    }
    best_by_memory: dict[str, dict] = {}
    for chunk_index, relevance in fused.items():
        chunk = chunks[chunk_index]
        metadata = chunk["metadata"]
        if bm25_norm or vector_norm:
            relevance_norm = max(bm25_norm.get(chunk_index, 0.0), vector_norm.get(chunk_index, 0.0))
        else:
            relevance_norm = fused_norm.get(chunk_index, 0.0)
        recency = _recency_factor(metadata, now, half_life)
        importance = _importance_factor(metadata, importance_default)
        if use_three_factor and chunk_index in rescoring_pool:
            score = w_rel * relevance_norm + w_rec * recency + w_imp * importance
        elif use_three_factor:
            score = w_rel * relevance_norm
        else:
            score = fused_norm.get(chunk_index, 0.0)
        record = {
            **chunk,
            "score": score,
            "factors": {
                "relevance": relevance,
                "relevance_norm": relevance_norm,
                "recency": recency,
                "importance": importance,
            },
            "retrieval_mode": mode,
        }
        previous = best_by_memory.get(chunk["memory_id"])
        if previous is None or score > previous["score"]:
            # 一个 memory 可能切成多个 chunk，最终只保留该 memory 的最佳命中块。
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
    vector_count = connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
    connection.close()
    diagnostics = {
        "mode": mode,
        "hyde": hyde_mode,
        "rerank": rerank_mode,
        "vector_backend": vector_backend,
        "use_chunk": bool(retrieval.get("use_chunk", True)),
        "use_three_factor": use_three_factor,
        "chunk_count": len(chunks),
        "chunk_cache": chunk_cache_diagnostics,
        "embedding_cache": {
            "path": str(paths["retrieval_cache"]),
            "size": vector_count,
            "hits": cache_diagnostics["embedding_cache_hits"],
            "misses": cache_diagnostics["embedding_cache_misses"],
            "disabled": cache_diagnostics["embedding_cache_disabled"],
        },
        "query_embedding_cache": {
            "path": str(paths["retrieval_cache"]),
            "hits": cache_diagnostics["query_embedding_cache_hits"],
            "misses": cache_diagnostics["query_embedding_cache_misses"],
            "disabled": cache_diagnostics["query_embedding_cache_disabled"],
        },
        "latency_ms": round((perf_counter() - started) * 1000, 3),
    }
    return ranked_docs, diagnostics


def _rerank_with_qwen(config_path: Path, config: dict, query: str, ranked_chunks: list[dict]) -> tuple[list[dict], str]:
    """用 Qwen 对初排候选做小规模重排，失败时保持原排序。"""
    if not ranked_chunks:
        return ranked_chunks, "empty"
    try:
        # 只把前 10 个候选交给 LLM 重排，控制 prompt 长度和生成延迟。
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
        generated = _qwen_generate(config_path, config, prompt, _token_budget(config, "rerank"))
        order = json.loads(generated)
        if isinstance(order, list):
            position = {str(memory_id): index for index, memory_id in enumerate(order)}
            return sorted(ranked_chunks, key=lambda item: (position.get(item["memory_id"], 999), -item["score"])), "qwen"
    except Exception:
        pass
    # rerank 失败不改变原排序，调用方可通过 mode 诊断看到 fallback。
    return ranked_chunks, "fallback"


def _format_memory_content(paths: dict, doc: dict, query: str | None, remaining: int) -> tuple[str, bool, str]:
    """把一条入选记忆压到剩余字符预算内，返回内容、是否压缩和压缩方法。"""
    config = paths["config"]
    metadata = doc["metadata"]
    title = metadata.get("title", doc["memory_id"])
    summary = metadata.get("summary", "")
    source = doc.get("chunk_content") or doc["content"]
    if doc.get("retrieval_mode") != "none" and doc.get("chunk_content"):
        # 检索模式下优先塞入命中片段，并附带标题/摘要作为最小上下文。
        source = f"# {title}\n\nSummary: {summary}\n\nRelevant chunk:\n{doc['chunk_content']}"
    budget = max(0, remaining)
    if len(source) <= budget:
        return source, False, "none"
    compressed, method = _compress_text(paths["config_path"], config, source, budget, query)
    return compressed, True, method


def _runtime_versions() -> dict:
    """记录影响复现实验和性能诊断的关键运行时版本。"""
    import platform
    from importlib import metadata

    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("numpy", "torch", "transformers", "PyYAML"):
        try:
            versions[package] = metadata.version(package)
        except Exception:
            versions[package] = None
    return versions


def _config_snapshot(config: dict) -> dict:
    """输出日志用配置快照，避免把模型路径等细节散落到每条日志里。"""
    return {
        "max_memory_chars": config.get("max_memory_chars"),
        "retrieval": config.get("retrieval"),
        "compression": config.get("compression"),
        "integration": config.get("integration"),
        "poison_gate": config.get("poison_gate"),
        "lifecycle": config.get("lifecycle"),
        "llm_enabled": bool(config.get("llm", {}).get("enabled", False)),
    }


def _update_access_metadata(index: dict, docs: list[dict], timestamp: str) -> None:
    """在索引层更新访问元数据；调用方负责最终 write_json 落盘。"""
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
    """加载可注入 agent 上下文的记忆，并在需要时执行检索、压缩和访问元数据更新。"""
    if not isinstance(selected_memory_ids, list) or not all(isinstance(item, str) for item in selected_memory_ids):
        raise ValueError("selected_memory_ids must be a list of strings")
    paths = _memory_paths(config_path)
    config = paths["config"]
    index = _read_index(paths["index"])
    # load 的主流程：确定候选 id -> 读取文档 -> 排序 -> 按预算压缩/截断。
    ordered_ids = _candidate_ids_for_load(index, selected_memory_ids, use_global_memory, query, config)
    retrieval_enabled = bool(query and config.get("retrieval", {}).get("mode") != "none")
    # 检索开启时先只校验路径并读取元数据，正文延迟到 top-k 已确定之后。
    docs, errors = _load_memory_docs(paths, index, ordered_ids, load_content=not retrieval_enabled)
    if query and config.get("retrieval", {}).get("mode") != "none":
        missing_selected = [memory_id for memory_id in selected_memory_ids if memory_id not in index]
        for memory_id in missing_selected:
            if not any(error.get("memory_id") == memory_id for error in errors):
                errors.append({"memory_id": memory_id, "type": "MemoryNotFound", "message": "memory_id does not exist"})
    ranked_docs, diagnostics = _rank_memory_docs(paths, docs, query)
    # 排名完成后最多读取 top_k 份正文，避免未命中候选产生无效文件 I/O。
    ranked_docs = _hydrate_memory_docs(ranked_docs)

    selected = []
    remaining = int(paths["max_chars"])
    any_truncated = False
    for doc in ranked_docs:
        if remaining <= 0:
            any_truncated = True
            break
        # 每加入一条记忆都消耗全局字符预算，保证返回内容不会超过 max_memory_chars。
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
    if selected and config.get("lifecycle", {}).get("update_access_on_load", True):
        # 读取也会更新访问时间和次数，供 recency 和生命周期淘汰使用。
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
                "config_snapshot": _config_snapshot(config),
                "versions": _runtime_versions(),
            },
            output_dir / "memory_log.jsonl",
        )
    return result

def _extract_json_object(text: str) -> dict | None:
    """从 LLM 输出中提取第一个 JSON object，解析失败时返回 None。"""
    # LLM 有时会包一层解释文字，这里尽量抽出第一段 JSON object。
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _llm_change_judge(config_path: Path, config: dict, previous_answer: str, answer: str) -> dict | None:
    """用 LLM 判断新旧记忆关系；异常和格式错误交给调用方规则兜底。"""
    prompt = (
        "You are auditing an agent memory update. Compare the OLD memory content with the NEW content "
        "and classify their relation as exactly one of: "
        "duplicate (same facts restated, nothing new), "
        "supplement (adds new compatible information), "
        "conflict (a key fact in NEW contradicts OLD). "
        'Return only JSON: {"change_type": "duplicate|supplement|conflict", "reason": "short reason"}\n\n'
        f"OLD:\n{previous_answer[:2000]}\n\nNEW:\n{answer[:2000]}"
    )
    generated = _qwen_generate(config_path, config, prompt, _token_budget(config, "judge"))
    verdict = _extract_json_object(generated)
    if not verdict or verdict.get("change_type") not in {"duplicate", "supplement", "conflict"}:
        return None
    return verdict


def _llm_nli_conflict(config_path: Path, config: dict, trusted_text: str, answer: str) -> dict | None:
    """用 NLI 风格提示检查新记忆是否与可信记忆矛盾。"""
    prompt = (
        "You are a consistency checker for an agent memory system. "
        "Decide whether the NEW memory contradicts the TRUSTED memory on any key fact. "
        "Extra information that does not contradict is NOT a conflict. "
        'Return only JSON: {"conflict": true or false, "reason": "short reason"}\n\n'
        f"TRUSTED:\n{trusted_text[:1500]}\n\nNEW:\n{answer[:1500]}"
    )
    generated = _qwen_generate(config_path, config, prompt, _token_budget(config, "judge"))
    verdict = _extract_json_object(generated)
    if not verdict or not isinstance(verdict.get("conflict"), bool):
        return None
    return verdict


def _detect_conflict(left: str, right: str, threshold: float) -> tuple[bool, str]:
    """规则冲突检测：先要求文本足够相似，再检查明显的否定翻转。"""
    similarity = _text_similarity(left, right)
    if similarity < threshold:
        return False, "low_overlap"
    negators = {"不", "不是", "不会", "不能", "错误", "失败", "false", "not", "never", "cannot"}
    left_tokens = set(_tokenize(left))
    right_tokens = set(_tokenize(right))
    left_neg = any(term in left.casefold() for term in negators)
    right_neg = any(term in right.casefold() for term in negators)
    shared = len(left_tokens & right_tokens)
    # 简单规则只捕捉高重叠文本中的否定翻转，复杂冲突交给 LLM judge。
    if shared >= 3 and left_neg != right_neg:
        return True, "negation_mismatch"
    return False, "no_rule_conflict"


def _poison_gate(paths: dict, index: dict, save_type: str, answer: str) -> dict:
    """保存前的防污染门：用可信长期记忆拦截或标记明显矛盾的新内容。"""
    config = paths["config"]
    gate = config.get("poison_gate", {})
    if not gate.get("enabled", False):
        return {"flagged": False, "action": "allow", "reason": "disabled", "method": "disabled", "matches": []}
    trusted_types = set(gate.get("trusted_memory_types", ["global"]))
    # 只用可信类型的记忆做防污染基准，默认 global 代表长期稳定事实。
    trusted_ids = [memory_id for memory_id, item in index.items() if item.get("memory_type") in trusted_types]
    trusted_docs, _ = _load_memory_docs(paths, index, trusted_ids)
    matches = []
    threshold = float(gate.get("conflict_threshold", 0.32))
    use_judge = bool(gate.get("use_llm_judge", True))
    method = "rule"
    for doc in trusted_docs:
        conflict, reason = _detect_conflict(doc["content"], answer, threshold)
        if use_judge:
            try:
                verdict = _llm_nli_conflict(paths["config_path"], config, doc["search_content"], answer)
            except Exception:
                verdict = None
            if verdict is not None:
                conflict = verdict["conflict"]
                reason = f"qwen_nli: {str(verdict.get('reason') or '')[:200]}"
                method = "qwen_nli"
        if conflict:
            matches.append({"memory_id": doc["memory_id"], "reason": reason})
    if not matches:
        return {"flagged": False, "action": "allow", "reason": "no_conflict", "method": method, "matches": []}
    action = gate.get("action", "flag")
    if save_type == "global":
        # 新 global 即使冲突也只标记不阻断，避免自动流程误删/误挡人工维护的长期记忆。
        action = "flag"
    return {"flagged": True, "action": action, "reason": "trusted_memory_conflict", "method": method, "matches": matches}


def _change_report(config_path: Path, existing_markdown: str | None, answer: str, config: dict) -> dict:
    """生成保存时的变更报告，描述本次内容是新增、重复、补充还是冲突。"""
    if not existing_markdown:
        return {"change_type": "new", "duplicate": False, "conflict": False, "similarity": None, "method": "rule", "notes": []}
    previous_answer = _section(existing_markdown, "Final Answer") or existing_markdown[:2000]
    similarity = _text_similarity(previous_answer, answer)
    integration = config.get("integration", {})
    duplicate_threshold = float(integration.get("duplicate_similarity", 0.92))
    conflict_threshold = float(integration.get("conflict_similarity", 0.28))
    conflict, reason = _detect_conflict(previous_answer, answer, conflict_threshold)
    if similarity >= duplicate_threshold:
        change_type = "duplicate"
    elif conflict:
        change_type = "conflict"
    else:
        change_type = "supplement"
    method = "rule"
    judge_reason = None
    rule_change_type = change_type
    if integration.get("use_llm_judge", True):
        try:
            # 规则先给出稳定兜底，LLM 只在可用时覆盖 duplicate/supplement/conflict 判断。
            verdict = _llm_change_judge(config_path, config, previous_answer, answer)
        except Exception:
            verdict = None
        if verdict:
            change_type = verdict["change_type"]
            conflict = change_type == "conflict"
            method = "qwen_judge"
            judge_reason = str(verdict.get("reason") or "")[:300]
    return {
        "change_type": change_type,
        "duplicate": change_type == "duplicate",
        "conflict": conflict,
        "conflict_reason": reason,
        "similarity": round(similarity, 6),
        "method": method,
        "rule_change_type": rule_change_type,
        "judge_reason": judge_reason,
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
    """把记忆正文、诊断信息和原始上下文统一写成可读 Markdown 文档。"""
    previous_section = ""
    if previous_answer and change_report.get("change_type") in {"supplement", "conflict"}:
        # 发生补充或冲突时保留上一版答案，便于后续人工追溯差异。
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
    """当索引超过容量上限时淘汰低价值记忆，并同步删除对应 Markdown 文件。"""
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
        # 淘汰优先级同时考虑重要性和时近性：低重要且久未访问的记忆先出局。
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


def _reflection_text(item: dict) -> str:
    """提取可用于反思聚类的短文本，避免把完整 trace/messages 放进聚类。"""
    return "\n".join(
        str(value)
        for value in (item.get("title"), item.get("summary"))
        if isinstance(value, str) and value.strip()
    ).strip()


def _cluster_reflection_sources(paths: dict, candidates: list[dict], min_size: int) -> tuple[list[dict], dict]:
    """从最近 conversation 记忆中找出一个语义相近的簇，作为反思输入。"""
    lifecycle = paths["config"].get("lifecycle", {})
    threshold = float(lifecycle.get("reflect_similarity_threshold", 0.35))
    vectors = []
    backend = None
    for item in candidates:
        text = _reflection_text(item)
        if not text:
            continue
        try:
            vector, used_backend = _vectorize(paths["config_path"], paths["config"], text)
        except Exception:
            # 反思聚类不能因为模型失败而中断，退回 hashing 仍能提供粗粒度相似度。
            vector, used_backend = _hash_vector(_tokenize(text), int(paths["config"].get("retrieval", {}).get("hashing_dim", 768))), "hashing"
        backend = backend or used_backend
        vectors.append({"item": item, "text": text, "vector": vector})
    if len(vectors) < min_size:
        return [], {
            "method": "embedding_greedy",
            "backend": backend,
            "candidate_count": len(vectors),
            "cluster_size": 0,
            "threshold": threshold,
            "reason": "not_enough_vectorized_candidates",
        }

    clusters = []
    assigned: set[int] = set()
    for seed_index, seed in enumerate(vectors):
        if seed_index in assigned:
            continue
        # 贪心聚类足够轻量：按阈值吸收相似记忆，并用当前簇均值持续更新中心。
        cluster_indices = [seed_index]
        assigned.add(seed_index)
        centroid = dict(seed["vector"])
        for index, candidate in enumerate(vectors):
            if index in assigned:
                continue
            if _cosine_dict(centroid, candidate["vector"]) >= threshold:
                cluster_indices.append(index)
                assigned.add(index)
                centroid = _mean_vectors([vectors[item]["vector"] for item in cluster_indices])
        clusters.append(cluster_indices)

    importance_default = int(lifecycle.get("importance_default", 5))
    now = datetime.now(timezone.utc)

    def cluster_score(indices: list[int]) -> tuple[int, float, float]:
        items = [vectors[index]["item"] for index in indices]
        avg_importance = sum(_importance_factor(item, importance_default) for item in items) / len(items)
        avg_recency = sum(_recency_factor(item, now, 30) for item in items) / len(items)
        return len(indices), avg_importance, avg_recency

    best = max(clusters, key=cluster_score)
    selected = [vectors[index]["item"] for index in best]
    diagnostics = {
        "method": "embedding_greedy",
        "backend": backend,
        "candidate_count": len(vectors),
        "cluster_count": len(clusters),
        "cluster_size": len(selected),
        "threshold": threshold,
        "source_memory_ids": [item["memory_id"] for item in selected],
    }
    if len(selected) < min_size:
        diagnostics["reason"] = "largest_cluster_below_min_size"
        return [], diagnostics
    return selected, diagnostics


def _generate_reflection_insight(paths: dict, summaries: list[str], budget: int) -> tuple[str, str]:
    """把同簇记忆综合成一条 global insight；LLM 不可用时使用可解释摘要兜底。"""
    joined = "\n".join(f"- {summary}" for summary in summaries if summary.strip())
    try:
        prompt = (
            "Synthesize the following related conversation memories into one higher-level long-term memory insight. "
            "Preserve concrete reusable facts and avoid listing every source verbatim. Return plain text only.\n\n"
            f"Budget characters: {budget}\n\n"
            f"Related memories:\n{joined[:6000]}"
        )
        generated = _qwen_generate(paths["config_path"], paths["config"], prompt, _token_budget(paths["config"], "reflect"))
        if generated:
            return generated[:budget].rstrip(), "qwen_reflection"
    except Exception:
        pass
    if len(joined) <= budget:
        return joined, "cluster_join"
    return (
        _extractive_summary(
            joined,
            budget,
            "cross conversation reflection",
            int(paths["config"].get("compression", {}).get("summary_sentences", 5)),
        ),
        "extractive_reflection",
    )


def _mean_vectors(vectors: list[dict[int, float]]) -> dict[int, float]:
    """计算归一化均值向量，用作贪心聚类的动态中心。"""
    merged: dict[int, float] = defaultdict(float)
    for vector in vectors:
        for index, value in vector.items():
            merged[index] += value
    count = max(1, len(vectors))
    for index in list(merged):
        merged[index] /= count
    norm = math.sqrt(sum(value * value for value in merged.values()))
    if norm:
        for index in list(merged):
            merged[index] /= norm
    return dict(merged)


def _reflect_if_needed(paths: dict, index: dict, timestamp: str) -> dict | None:
    """按固定间隔触发跨会话反思，并把稳定洞察写成 global memory。"""
    lifecycle = paths["config"].get("lifecycle", {})
    if not lifecycle.get("reflect_enabled", False):
        return None
    conversation_items = [item for item in index.values() if item.get("memory_type") == "conversation"]
    every_n = int(lifecycle.get("reflect_every_n", 10))
    if every_n <= 0 or len(conversation_items) < every_n or len(conversation_items) % every_n != 0:
        return None
    # 只在 conversation 记忆数量达到固定间隔时触发，避免每次保存都做聚类和生成。
    recent = sorted(conversation_items, key=lambda item: item.get("updated_at", ""))[-every_n:]
    min_cluster_size = int(lifecycle.get("reflect_min_cluster_size", 3))
    clustered, cluster_diagnostics = _cluster_reflection_sources(paths, recent, min_cluster_size)
    if len(clustered) < min_cluster_size:
        return None
    summaries = [_reflection_text(item) for item in clustered if _reflection_text(item)]
    insight_text, method = _generate_reflection_insight(
        paths,
        summaries,
        int(paths["config"].get("compression", {}).get("summary_chars", 700)),
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
        f"- source_count: `{len(clustered)}`\n"
        f"- compression_method: `{method}`\n\n"
        "## Cluster\n\n```json\n"
        f"{json.dumps(cluster_diagnostics, ensure_ascii=False, indent=2)}\n```\n\n"
        "## Insight\n\n"
        f"{insight_text}\n"
    )
    write_text(markdown, path)
    index[memory_id] = {
        "memory_id": memory_id,
        "memory_type": "global",
        "title": title,
        "summary": insight_text[:300],
        "retrieval_summary": insight_text[:700],
        "retrieval_terms": {
            "tags": _dedupe_limited(_tokenize(insight_text), 30, 80),
            "entities": [],
            "files": [],
            "errors": [],
            "commands": [],
            "tool_names": [],
        },
        "path": relative,
        "conversation_id": None,
        "importance": 7,
        "created_at": timestamp,
        "updated_at": timestamp,
        "source_memory_ids": [item["memory_id"] for item in clustered],
        "reflection_cluster": cluster_diagnostics,
    }
    return {"memory_id": memory_id, "path": relative, "method": method, "cluster": cluster_diagnostics}


def save_memory(
    config_path: str,
    conversation_id: str,
    save_type: str,
    messages_path: str,
    trace_path: str,
    answer_path: str,
    outdir: str | None = None,
) -> dict:
    """保存一条 conversation/global 记忆，并更新索引、缓存和生命周期诊断。"""
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
    # conversation/global 使用同一套文档格式，但落到不同目录并写入不同 memory_type。
    target_dir = paths["conversations"] if save_type == "conversation" else paths["global"]
    relative_dir = "conversations" if save_type == "conversation" else "global"
    target_path = Path(target_dir) / f"{conversation_id}.md"
    relative_path = f"{relative_dir}/{conversation_id}.md"
    title = f"{save_type.title()} {conversation_id}"
    summary, summary_method = _summarize_for_index(paths["config_path"], config, answer)
    importance, importance_method = _estimate_importance(paths["config_path"], config, answer, trace)
    retrieval_metadata = _build_retrieval_metadata(answer, messages, trace)
    index = _read_index(paths["index"])
    existing = index.get(memory_id, {})
    existing_markdown = read_text(target_path) if target_path.exists() else None
    previous_answer = _section(existing_markdown, "Final Answer") if existing_markdown else None
    change_report = _change_report(paths["config_path"], existing_markdown, answer, config)
    poison = _poison_gate(paths, index, save_type, answer)
    if poison.get("action") == "block":
        # block 只返回诊断结果，不写 markdown 和索引，避免污染已存在记忆。
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
        # 重复内容不重写文档正文，但仍会刷新索引中的更新时间和诊断信息。
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
        **retrieval_metadata,
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
    try:
        retrieval_cache = _warm_retrieval_cache(paths, index, [memory_id])
    except Exception as exc:
        # 缓存预热失败不能阻止记忆落盘；下次检索会按同一逻辑自动补建。
        retrieval_cache = {"status": "fallback", "error": f"{type(exc).__name__}: {exc}"}
    reflection = _reflect_if_needed(paths, index, timestamp)
    if reflection:
        try:
            reflection["retrieval_cache"] = _warm_retrieval_cache(paths, index, [reflection["memory_id"]])
        except Exception as exc:
            # reflection 是派生记忆，缓存失败同样只记录诊断，不影响主保存流程。
            reflection["retrieval_cache"] = {"status": "fallback", "error": f"{type(exc).__name__}: {exc}"}
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
        "retrieval_cache": retrieval_cache,
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
                "config_snapshot": _config_snapshot(config),
                "versions": _runtime_versions(),
            },
            output_dir / "memory_log.jsonl",
        )
    return result


def parse_bool(value: str) -> bool:
    """解析 CLI 布尔字符串，非法值抛出 argparse 友好的错误。"""
    lowered = value.lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_parser() -> argparse.ArgumentParser:
    """构建 b5_memory.py 的命令行参数解析器。"""
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
    """命令行入口：根据参数选择加载记忆或保存记忆。"""
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
