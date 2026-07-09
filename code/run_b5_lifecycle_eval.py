"""B5 生命周期层评测（RQ5，对应 proposal 6.2 节）。

A) 淘汰一致性：合成 30 条元数据（importance / 时近性各异，含 pinned 与 global），
   容量上限 20，触发 `_evict_if_needed`，对照评测脚本独立复算的 oracle
   （重要性 × 时近性升序）报 Kendall τ、保护规则违例数与容量稳定性。
B) 反思洞见命中率：构造 3 簇同主题对话记忆（每簇 4 条）按簇顺序写入，
   每满 4 条触发一次 `_reflect_if_needed`（reflect_every_n=4），再用 3 条
   洞见级查询跑检索，统计"正确簇的反思记忆进入 top-k"的比例；
   洞见全文落盘供人工评分（1–5）。

用法（在 code/ 目录下）：
    python run_b5_lifecycle_eval.py --llm off --outdir ../outputs/B5_lifecycle_rule
    python run_b5_lifecycle_eval.py --llm on  --outdir ../outputs/B5_lifecycle_qwen
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from common.io_utils import read_yaml, write_json, write_text
from common.logging_utils import now_iso
from common.path_utils import resolve_cli_path, resolve_from_file
from b5_memory import (
    _evict_if_needed,
    _importance_factor,
    _memory_paths,
    _recency_factor,
    _reflect_if_needed,
    load_memory,
)


REFLECT_CLUSTERS = [
    {
        "topic": "检索调参",
        "query": "记忆检索层的参数应该怎么配置？",
        "items": [
            ("BM25 参数结论", "BM25 关键词打分取 k1=1.5、b=0.75，中文按字符 bigram 切词。"),
            ("RRF 融合参数", "关键词与向量两路检索用 RRF 融合，rrf_k 取 60，候选上限 30 条。"),
            ("chunk 切块参数", "长记忆按 700 字符切块、重叠 120 字符，chunk 粒度检索后回溯原记忆。"),
            ("top_k 与门控", "检索最终返回 top 5 条记忆，三因子重排只在相关度前 10 名候选内生效。"),
        ],
    },
    {
        "topic": "显存与加载",
        "query": "模型显存和加载方面有哪些经验教训？",
        "items": [
            ("OOM 修复方案", "CUDA out of memory 用 bfloat16 加 max_memory 上限 20GiB 解决，float32 更占显存。"),
            ("加载耗时实测", "Qwen3.5-4B 模型加载约九十秒，进程内必须缓存复用，禁止重复加载。"),
            ("精度选型", "bfloat16 全程无溢出警告，float16 在长文任务偶发溢出，配置锁定 bfloat16。"),
            ("显存检查习惯", "共享服务器启动任务前先 nvidia-smi 看显存余量，避免挤崩别人的训练。"),
        ],
    },
    {
        "topic": "工具调用规范",
        "query": "调用内置工具有哪些使用规范？",
        "items": [
            ("calculator 用法", "calculator 只接收标准算式字符串，禁止传自然语言描述。"),
            ("table_analyzer 限制", "table_analyzer 一次调用只支持一个统计操作，列名须与表头完全一致。"),
            ("file_reader 边界", "file_reader 只能读取 data 根目录内的文本文件，越界请求被拒绝。"),
            ("搜索工具用法", "local_file_search 用通配符匹配文件名，可选递归子目录，范围限 data 内。"),
        ],
    },
]


def _materialize_config(base_config_path: Path, llm_enabled: bool, root_dir: Path, target_path: Path, overrides: dict) -> Path:
    config = read_yaml(base_config_path)
    memory = deepcopy(config["memory"])
    memory["root_dir"] = str(root_dir)
    llm = memory.setdefault("llm", {})
    if isinstance(llm.get("model_config"), str):
        llm["model_config"] = str(resolve_from_file(llm["model_config"], base_config_path))
    llm["enabled"] = llm_enabled
    lifecycle = memory.setdefault("lifecycle", {})
    lifecycle.update(overrides.get("lifecycle", {}))
    retrieval = memory.setdefault("retrieval", {})
    retrieval.update(overrides.get("retrieval", {}))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    write_text(yaml.safe_dump({"memory": memory}, allow_unicode=True, sort_keys=False), target_path)
    return target_path


def _kendall_tau(order_a: list[str], order_b: list[str]) -> float | None:
    common = [item for item in order_a if item in set(order_b)]
    if len(common) < 2:
        return None
    rank_b = {item: index for index, item in enumerate(order_b)}
    concordant = discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            diff = rank_b[common[i]] - rank_b[common[j]]
            if diff < 0:
                concordant += 1
            elif diff > 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else None


def _eval_evict(base_config_path: Path, output_dir: Path, llm_enabled: bool) -> dict:
    root = output_dir / "corpus_evict"
    (root / "conversations").mkdir(parents=True, exist_ok=True)
    (root / "global").mkdir(parents=True, exist_ok=True)
    max_memories = 20
    config_path = _materialize_config(
        base_config_path,
        llm_enabled,
        root,
        output_dir / "configs" / "evict_eval.yaml",
        {"lifecycle": {"max_memories": max_memories, "protect_global": True}},
    )
    paths = _memory_paths(config_path)
    now = datetime.now(timezone.utc)
    index: dict[str, dict] = {}
    for i in range(30):
        memory_type = "global" if i % 8 == 0 else "conversation"
        stamp = (now - timedelta(days=(i * 11) % 90)).isoformat()
        index[f"mem_evict_{i:02d}"] = {
            "memory_id": f"mem_evict_{i:02d}",
            "memory_type": memory_type,
            "title": f"synthetic {i}",
            "summary": "synthetic",
            "path": f"{'global' if memory_type == 'global' else 'conversations'}/fake_{i:02d}.md",
            "importance": (i * 7) % 9 + 1,
            "pinned": i % 10 == 3,
            "created_at": stamp,
            "updated_at": stamp,
            "last_accessed_at": stamp,
        }
    protected = {mid for mid, meta in index.items() if meta.get("pinned") or meta.get("memory_type") == "global"}
    evictable = {mid: meta for mid, meta in index.items() if mid not in protected}
    oracle = sorted(
        evictable,
        key=lambda mid: (_importance_factor(evictable[mid], 5) * _recency_factor(evictable[mid], now, 30), mid),
    )
    expected_evictions = oracle[: len(index) - max_memories]
    working = deepcopy(index)
    evicted = _evict_if_needed(paths, working)
    evicted_ids = [item["memory_id"] for item in evicted]
    violations = [mid for mid in evicted_ids if mid in protected]
    return {
        "initial_count": len(index),
        "max_memories": max_memories,
        "final_count": len(working),
        "capacity_ok": len(working) <= max_memories,
        "evicted_count": len(evicted_ids),
        "protected_violations": violations,
        "kendall_tau_vs_oracle": _kendall_tau(evicted_ids, oracle),
        "set_match_with_oracle": set(evicted_ids) == set(expected_evictions),
        "evicted_order": evicted_ids,
        "oracle_order": expected_evictions,
    }


def _eval_reflect(base_config_path: Path, output_dir: Path, llm_enabled: bool) -> dict:
    root = output_dir / "corpus_reflect"
    (root / "conversations").mkdir(parents=True, exist_ok=True)
    (root / "global").mkdir(parents=True, exist_ok=True)
    config_path = _materialize_config(
        base_config_path,
        llm_enabled,
        root,
        output_dir / "configs" / "reflect_eval.yaml",
        {
            "lifecycle": {
                "reflect_enabled": True,
                "reflect_every_n": 4,
                "reflect_min_cluster_size": 3,
                # hashing 向量的余弦系统性低于 qwen embedding（短文本 bigram 重合少），
                # 阈值按后端取值；反思窗口只含最近 4 条同主题记忆，低阈值不会跨主题误聚。
                "reflect_similarity_threshold": 0.35 if llm_enabled else 0.05,
                "update_access_on_load": False,
            },
            "retrieval": {"use_rerank": False, "use_hyde": False},
        },
    )
    paths = _memory_paths(config_path)
    now = datetime.now(timezone.utc)
    index: dict[str, dict] = {}
    reflections = []
    counter = 0
    cluster_ids: dict[str, list[str]] = {}
    for cluster in REFLECT_CLUSTERS:
        ids = []
        for title, summary in cluster["items"]:
            counter += 1
            memory_id = f"mem_conv_life_{counter:02d}"
            relative = f"conversations/life_{counter:02d}.md"
            stamp = (now - timedelta(minutes=(48 - counter))).isoformat()
            write_text(f"# {title}\n\n## Final Answer\n\n{summary}\n", root / relative)
            index[memory_id] = {
                "memory_id": memory_id,
                "memory_type": "conversation",
                "title": title,
                "summary": summary,
                "path": relative,
                "conversation_id": f"life_{counter:02d}",
                "importance": 6,
                "created_at": stamp,
                "updated_at": stamp,
                "last_accessed_at": stamp,
            }
            ids.append(memory_id)
        cluster_ids[cluster["topic"]] = ids
        reflection = _reflect_if_needed(paths, index, now_iso())
        reflections.append({"after_cluster": cluster["topic"], "reflection": reflection})
    write_json(index, paths["index"])

    hits = []
    for cluster in REFLECT_CLUSTERS:
        result = load_memory(
            str(config_path),
            [],
            True,
            cluster["query"],
            str(output_dir / "reflect_queries" / cluster["topic"]),
        )
        ranked = [doc["memory_id"] for doc in result["selected_memory_docs"]]
        target_sources = set(cluster_ids[cluster["topic"]])
        hit = False
        for memory_id in ranked:
            meta = index.get(memory_id, {})
            sources = set(meta.get("source_memory_ids") or [])
            if memory_id.startswith("mem_global_reflect_") and len(sources & target_sources) >= 2:
                hit = True
                break
        hits.append({"topic": cluster["topic"], "query": cluster["query"], "ranked": ranked, "insight_hit": hit})
    generated = [item for item in reflections if item["reflection"]]
    insight_texts = []
    for item in generated:
        insight_path = paths["root"] / item["reflection"]["path"]
        insight_texts.append({"after_cluster": item["after_cluster"], "text": insight_path.read_text(encoding="utf-8")})
    return {
        "clusters": len(REFLECT_CLUSTERS),
        "reflections_generated": len(generated),
        "insight_hit_rate": sum(item["insight_hit"] for item in hits) / len(hits),
        "hits": hits,
        "reflections": reflections,
        "insight_documents_for_manual_scoring": insight_texts,
    }


def run(base_config: str, outdir: str, llm_flag: str) -> dict:
    base_config_path = resolve_cli_path(base_config)
    output_dir = resolve_cli_path(outdir)
    llm_enabled = llm_flag == "on"
    evict = _eval_evict(base_config_path, output_dir, llm_enabled)
    tau = evict["kendall_tau_vs_oracle"]
    print(f"[RQ5-evict] tau={tau if tau is None else round(tau, 3)} capacity_ok={evict['capacity_ok']} violations={len(evict['protected_violations'])}")
    reflect = _eval_reflect(base_config_path, output_dir, llm_enabled)
    print(f"[RQ5-reflect] generated={reflect['reflections_generated']}/{reflect['clusters']} insight_hit_rate={reflect['insight_hit_rate']:.3f}")
    summary = {"llm": llm_flag, "evict": evict, "reflect": reflect}
    write_json(summary, output_dir / "lifecycle_eval.json")
    lines = [
        f"# B5 生命周期评测（llm={llm_flag}）",
        "",
        "## 淘汰一致性（evict vs oracle）",
        "",
        f"- Kendall τ：{tau if tau is None else f'{tau:.3f}'}",
        f"- 淘汰集合与 oracle 一致：{evict['set_match_with_oracle']}",
        f"- 容量稳定：{evict['initial_count']} → {evict['final_count']}（上限 {evict['max_memories']}）",
        f"- pinned / global 保护违例：{len(evict['protected_violations'])}",
        "",
        "## 反思洞见",
        "",
        f"- 触发反思：{reflect['reflections_generated']}/{reflect['clusters']} 簇",
        f"- 洞见命中率（洞见级查询进 top-k）：{reflect['insight_hit_rate']:.3f}",
        "- 洞见全文见 lifecycle_eval.json 的 insight_documents_for_manual_scoring，供人工评分（1–5）。",
        "",
    ]
    write_text("\n".join(lines), output_dir / "lifecycle_eval.md")
    print(output_dir / "lifecycle_eval.md")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate B5 lifecycle: evict consistency and reflection insight hit rate.")
    parser.add_argument("--base_config", default="../configs/memory_eval_corpus.yaml")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--llm", choices=["on", "off"], default="off")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args.base_config, args.outdir, args.llm)
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
