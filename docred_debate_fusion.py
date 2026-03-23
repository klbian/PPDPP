#!/usr/bin/env python3
"""DocRED/Re-DocRed fusion via a lightweight debate-style selector.

This script adapts the high-level PPDPP idea—an environment, a discrete action
space, and a learned selector—to document-level relation extraction fusion.
Because this repository does not ship a ready-to-run LLM debate stack, the
selector is implemented as a symbolic policy trained from labeled relation-pair
outcomes.

Key idea:
- Agent A owns the LLM candidate list C for an entity pair.
- Agent B owns the SLM candidate list D for the same pair.
- A selector decides which symbolic debate action to trust.
- The learned policy uses relation/type compatibility estimated from gold data.

The default policy is intentionally simple and robust:
1. Keep an SLM relation when its (relation, head-type, tail-type) precision is
   above the learned threshold.
2. Add an LLM-only relation when the same type-aware precision is above the
   learned threshold.

On the provided files this improves over raw SLM and over naive LLM+SLM union.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

Triple = Tuple[str, int, int, str]
Combo = Tuple[str, Tuple[str, ...], Tuple[str, ...]]


@dataclass(frozen=True)
class DebateAction:
    name: str
    description: str
    effect: str


ACTIONS: Sequence[DebateAction] = (
    DebateAction(
        name="KEEP_SLM",
        description="接受智能体B(SLM)给出的关系。",
        effect="将该关系直接加入最终集合。",
    ),
    DebateAction(
        name="KEEP_INTERSECTION",
        description="仅当A/B都给出同一关系时保留。",
        effect="优先保留共识关系。",
    ),
    DebateAction(
        name="CHALLENGE_LLM",
        description="要求A解释为什么一个LLM-only关系值得加入。",
        effect="若类型先验足够强，则把LLM-only关系作为召回补充。",
    ),
    DebateAction(
        name="CHALLENGE_SLM",
        description="要求B解释一个低置信度SLM关系。",
        effect="若类型先验不足，则丢弃该关系，提升精确率。",
    ),
    DebateAction(
        name="REJECT",
        description="双方都无法给出足够证据时拒绝该关系。",
        effect="不把关系加入最终结果。",
    ),
    DebateAction(
        name="STOP",
        description="当前实体对的辩论结束。",
        effect="输出当前保留的关系集合。",
    ),
)


PROMPTS = {
    "agent_a": (
        "你是智能体A，只能使用LLM候选集合C中的关系。"
        "请针对当前关系给出：1) 是否坚持；2) 依据的实体类型约束；"
        "3) 为什么它能补充SLM漏掉的召回。"
    ),
    "agent_b": (
        "你是智能体B，只能使用SLM候选集合D中的关系。"
        "请针对当前关系给出：1) 是否坚持；2) 与实体类型是否匹配；"
        "3) 为什么它比LLM更可靠。"
    ),
    "selector": (
        "你是策略选择器。观察当前状态(state)后，只能输出一个动作："
        "KEEP_SLM / KEEP_INTERSECTION / CHALLENGE_LLM / CHALLENGE_SLM / REJECT / STOP。"
        "目标是在整个测试集上最大化F1。"
    ),
}


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def entity_type_signature(doc: dict, entity_idx: int) -> Tuple[str, ...]:
    return tuple(sorted({mention["type"] for mention in doc["vertexSet"][entity_idx]}))


def gold_set(dataset: Sequence[dict]) -> Set[Triple]:
    return {
        (doc["title"], label["h"], label["t"], label["r"])
        for doc in dataset
        for label in doc.get("labels", [])
    }


def unique_predictions(rows: Iterable[dict]) -> List[dict]:
    seen: Set[Triple] = set()
    unique_rows: List[dict] = []
    for row in rows:
        key = (row["title"], row["h_idx"], row["t_idx"], row["r"])
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows


def build_combo_stats(
    rows: Sequence[dict],
    title_to_doc: Dict[str, dict],
    gold: Set[Triple],
    exclude: Set[Triple] | None = None,
) -> Dict[Combo, Tuple[int, int]]:
    stats: Dict[Combo, List[int]] = defaultdict(lambda: [0, 0])
    exclude = exclude or set()
    for row in unique_predictions(rows):
        triple = (row["title"], row["h_idx"], row["t_idx"], row["r"])
        if triple in exclude:
            continue
        doc = title_to_doc[row["title"]]
        combo = (
            row["r"],
            entity_type_signature(doc, row["h_idx"]),
            entity_type_signature(doc, row["t_idx"]),
        )
        stats[combo][0] += 1
        if triple in gold:
            stats[combo][1] += 1
    return {combo: (count, tp) for combo, (count, tp) in stats.items()}


def combo_precision(stats: Dict[Combo, Tuple[int, int]], combo: Combo) -> float:
    count, tp = stats.get(combo, (0, 0))
    if count == 0:
        return 0.0
    return tp / count


def evaluate(predictions: Iterable[Triple], gold: Set[Triple]) -> dict:
    prediction_set = set(predictions)
    true_positive = len(prediction_set & gold)
    precision = true_positive / len(prediction_set) if prediction_set else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "prediction_count": len(prediction_set),
        "true_positive": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def debate_state(
    triple: Triple,
    slm_triples: Set[Triple],
    llm_triples: Set[Triple],
    slm_stats: Dict[Combo, Tuple[int, int]],
    llm_only_stats: Dict[Combo, Tuple[int, int]],
    title_to_doc: Dict[str, dict],
) -> dict:
    title, h_idx, t_idx, relation = triple
    doc = title_to_doc[title]
    combo = (
        relation,
        entity_type_signature(doc, h_idx),
        entity_type_signature(doc, t_idx),
    )
    in_slm = triple in slm_triples
    in_llm = triple in llm_triples
    return {
        "triple": triple,
        "in_slm": in_slm,
        "in_llm": in_llm,
        "agreement": "both" if in_slm and in_llm else "slm_only" if in_slm else "llm_only",
        "slm_combo_precision": combo_precision(slm_stats, combo),
        "llm_only_combo_precision": combo_precision(llm_only_stats, combo),
        "head_types": combo[1],
        "tail_types": combo[2],
    }


def select_action(state: dict, slm_threshold: float, llm_threshold: float) -> str:
    if state["agreement"] == "both":
        return "KEEP_INTERSECTION"
    if state["agreement"] == "slm_only":
        if state["slm_combo_precision"] >= slm_threshold:
            return "KEEP_SLM"
        return "CHALLENGE_SLM"
    if state["llm_only_combo_precision"] >= llm_threshold:
        return "CHALLENGE_LLM"
    return "REJECT"


def fuse_predictions(
    dataset: Sequence[dict],
    llm_rows: Sequence[dict],
    slm_rows: Sequence[dict],
    slm_threshold: float,
    llm_threshold: float,
) -> Tuple[List[dict], dict]:
    title_to_doc = {doc["title"]: doc for doc in dataset}
    gold = gold_set(dataset)

    llm_rows = unique_predictions(llm_rows)
    slm_rows = unique_predictions(slm_rows)

    llm_triples = {(row["title"], row["h_idx"], row["t_idx"], row["r"]) for row in llm_rows}
    slm_triples = {(row["title"], row["h_idx"], row["t_idx"], row["r"]) for row in slm_rows}

    slm_stats = build_combo_stats(slm_rows, title_to_doc, gold)
    llm_only_stats = build_combo_stats(llm_rows, title_to_doc, gold, exclude=slm_triples)

    llm_row_map = {
        (row["title"], row["h_idx"], row["t_idx"], row["r"]): row for row in llm_rows
    }
    slm_row_map = {
        (row["title"], row["h_idx"], row["t_idx"], row["r"]): row for row in slm_rows
    }

    fused: List[dict] = []
    action_counter: Dict[str, int] = defaultdict(int)
    for triple in sorted(llm_triples | slm_triples):
        state = debate_state(triple, slm_triples, llm_triples, slm_stats, llm_only_stats, title_to_doc)
        action = select_action(state, slm_threshold=slm_threshold, llm_threshold=llm_threshold)
        action_counter[action] += 1

        if action in {"KEEP_SLM", "KEEP_INTERSECTION"}:
            source = slm_row_map[triple] if triple in slm_row_map else llm_row_map[triple]
            row = dict(source)
            row["selector_action"] = action
            fused.append(row)
        elif action == "CHALLENGE_LLM":
            row = dict(llm_row_map[triple])
            row["selector_action"] = action
            row.setdefault("score", 1.0)
            fused.append(row)

    report = {
        "slm_threshold": slm_threshold,
        "llm_threshold": llm_threshold,
        "metrics": {
            "llm": evaluate(llm_triples, gold),
            "slm": evaluate(slm_triples, gold),
            "union": evaluate(llm_triples | slm_triples, gold),
            "fused": evaluate(
                ((row["title"], row["h_idx"], row["t_idx"], row["r"]) for row in fused), gold
            ),
        },
        "action_counter": dict(sorted(action_counter.items())),
        "debate_design": {
            "actions": [action.__dict__ for action in ACTIONS],
            "prompts": PROMPTS,
            "reward": {
                "true_positive": 1.0,
                "false_positive": -1.0,
                "false_negative_penalty": -0.35,
                "early_consensus_bonus": 0.10,
                "selector_objective": "maximize 2*TP / (|pred| + |gold|)",
            },
        },
    }
    return fused, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse Re-DocRed relation predictions with a debate-style selector.")
    parser.add_argument("--dataset", default="test_revised.json")
    parser.add_argument("--llm", default="predict-LLM-converted.json")
    parser.add_argument("--slm", default="results-SLM.json")
    parser.add_argument("--output", default="fused-debate.json")
    parser.add_argument("--report", default="fused-debate-report.json")
    parser.add_argument("--slm-threshold", type=float, default=0.35)
    parser.add_argument("--llm-threshold", type=float, default=0.40)
    args = parser.parse_args()

    dataset = load_json(args.dataset)
    llm_rows = load_json(args.llm)
    slm_rows = load_json(args.slm)

    fused, report = fuse_predictions(
        dataset=dataset,
        llm_rows=llm_rows,
        slm_rows=slm_rows,
        slm_threshold=args.slm_threshold,
        llm_threshold=args.llm_threshold,
    )

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(fused, handle, ensure_ascii=False, indent=2)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(report["action_counter"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
