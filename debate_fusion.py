import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from debate_prompt import build_agent_a_prompt, build_agent_b_prompt, build_selector_prompt


ACTION_DESCRIPTIONS: Dict[str, str] = {
    "none": "Reject both candidate lists and output an empty relation set.",
    "inter": "Keep only the intersection C ∩ D.",
    "slm": "Keep the full SLM candidate set D.",
    "llm": "Keep the full LLM candidate set C.",
    "union": "Keep C ∪ D.",
    "slm_top1": "Keep only the highest-scoring SLM relation.",
    "slm_top2": "Keep the top-2 SLM relations.",
    "slm_top3": "Keep the top-3 SLM relations.",
    "llm_top1": "Keep only the first LLM relation.",
    "llm_top2": "Keep the first two LLM relations.",
    "slm_cal": "Keep only SLM relations whose score exceeds a learned relation-specific threshold.",
    "white": "Keep only LLM relations that fall into learned high-precision LLM-only type patterns.",
    "slm_cal_plus_white": "Take calibrated SLM relations and add high-precision LLM-only relations.",
    "slm_plus_white": "Take the full SLM set and add high-precision LLM-only relations.",
    "slm_plus_llm_top1": "Take the SLM set and add the top LLM relation.",
    "inter_plus_llm_top1": "Take the overlap and then add the top LLM relation.",
}


@dataclass
class PairCase:
    title: str
    head_idx: int
    tail_idx: int
    head_name: str
    tail_name: str
    head_type: str
    tail_type: str
    llm_candidates: List[Dict[str, object]]
    slm_candidates: List[Dict[str, object]]
    gold_relations: Set[str]
    overlap_size: int
    agent_a_prompt: str
    agent_b_prompt: str
    selector_prompt: str

    @property
    def pair_key(self) -> Tuple[str, int, int]:
        return (self.title, self.head_idx, self.tail_idx)


class DebateActionSpace:
    def __init__(self, relation_thresholds: Dict[str, float], llm_whitelist: Set[Tuple[str, str, str]]):
        self.relation_thresholds = relation_thresholds
        self.llm_whitelist = llm_whitelist

    def apply(self, case: PairCase, action_name: str) -> Set[str]:
        llm_rels = [item["r"] for item in case.llm_candidates]
        slm_rels = [item["r"] for item in case.slm_candidates]
        llm_set = set(llm_rels)
        slm_set = set(slm_rels)
        inter = llm_set & slm_set
        slm_cal = {
            item["r"]
            for item in case.slm_candidates
            if float(item.get("score", 0.0)) >= self.relation_thresholds.get(item["r"], float("inf"))
        }
        white = {
            item["r"]
            for item in case.llm_candidates
            if (item["r"], case.head_type, case.tail_type) in self.llm_whitelist
        }

        candidates = {
            "none": set(),
            "inter": inter,
            "slm": slm_set,
            "llm": llm_set,
            "union": llm_set | slm_set,
            "slm_top1": set(slm_rels[:1]),
            "slm_top2": set(slm_rels[:2]),
            "slm_top3": set(slm_rels[:3]),
            "llm_top1": set(llm_rels[:1]),
            "llm_top2": set(llm_rels[:2]),
            "slm_cal": slm_cal,
            "white": white,
            "slm_cal_plus_white": slm_cal | white,
            "slm_plus_white": slm_set | white,
            "slm_plus_llm_top1": slm_set | set(llm_rels[:1]),
            "inter_plus_llm_top1": inter | set(llm_rels[:1]),
        }
        if action_name not in candidates:
            raise KeyError(f"Unsupported action: {action_name}")
        return candidates[action_name]

    @property
    def actions(self) -> List[str]:
        return list(ACTION_DESCRIPTIONS.keys())


class RelationDebateEnv:
    def __init__(
        self,
        gold_path: str,
        llm_path: str,
        slm_path: str,
        rel_info_path: str,
    ):
        self.docs = self._load_json(gold_path)
        self.rel_info = self._load_json(rel_info_path)
        self.doc_map = {doc["title"]: doc for doc in self.docs}
        self.gold_set = {
            (doc["title"], label["h"], label["t"], label["r"])
            for doc in self.docs
            for label in doc.get("labels", [])
        }
        self.pair_gold: Dict[Tuple[str, int, int], Set[str]] = defaultdict(set)
        for title, h_idx, t_idx, relation in self.gold_set:
            self.pair_gold[(title, h_idx, t_idx)].add(relation)

        self.llm_by_pair = self._index_predictions(llm_path)
        self.slm_by_pair = self._index_predictions(slm_path)
        self.relation_thresholds = self._learn_relation_thresholds()
        self.llm_whitelist = self._learn_llm_whitelist()
        self.action_space = DebateActionSpace(self.relation_thresholds, self.llm_whitelist)
        self.cases = self._build_cases()
        self.current_case: Optional[PairCase] = None

    @staticmethod
    def _load_json(path: str):
        with open(path, "r", encoding="utf-8") as infile:
            return json.load(infile)

    def _index_predictions(self, path: str) -> Dict[Tuple[str, int, int], List[Dict[str, object]]]:
        pair_map: Dict[Tuple[str, int, int], List[Dict[str, object]]] = defaultdict(list)
        for row in self._load_json(path):
            title = row["title"]
            head_idx = row.get("h_idx", row.get("h"))
            tail_idx = row.get("t_idx", row.get("t"))
            item = {
                "title": title,
                "h_idx": head_idx,
                "t_idx": tail_idx,
                "r": row["r"],
                "score": float(row.get("score", 0.0) or 0.0),
                "evidence": row.get("evidence", []),
                "relation_text": self.rel_info.get(row["r"], row["r"]),
            }
            pair_map[(title, head_idx, tail_idx)].append(item)
        for key in pair_map:
            source = pair_map[key]
            source.sort(key=lambda item: (-float(item.get("score", 0.0)), item["r"]))
        return pair_map

    def _learn_relation_thresholds(self) -> Dict[str, float]:
        by_relation: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
        for pair, items in self.slm_by_pair.items():
            for item in items:
                key = (pair[0], pair[1], pair[2], item["r"])
                by_relation[item["r"]].append((float(item.get("score", 0.0)), 1 if key in self.gold_set else 0))

        thresholds: Dict[str, float] = {}
        for relation, rows in by_relation.items():
            rows.sort(key=lambda entry: (-entry[0], -entry[1]))
            total_positive = sum(label for _, label in rows)
            true_positive = 0
            false_positive = 0
            best_f1 = -1.0
            best_threshold = float("inf")
            cursor = 0
            while cursor < len(rows):
                score = rows[cursor][0]
                while cursor < len(rows) and rows[cursor][0] == score:
                    if rows[cursor][1] == 1:
                        true_positive += 1
                    else:
                        false_positive += 1
                    cursor += 1
                false_negative = total_positive - true_positive
                precision, recall, f1_value = self._metrics_from_counts(true_positive, false_positive, false_negative)
                if f1_value > best_f1:
                    best_f1 = f1_value
                    best_threshold = score
            thresholds[relation] = best_threshold
        return thresholds

    def _learn_llm_whitelist(self) -> Set[Tuple[str, str, str]]:
        slm_keys = {
            (pair[0], pair[1], pair[2], item["r"])
            for pair, items in self.slm_by_pair.items()
            for item in items
        }
        stats: Dict[Tuple[str, str, str], List[int]] = defaultdict(lambda: [0, 0])
        for pair, items in self.llm_by_pair.items():
            doc = self.doc_map[pair[0]]
            head_type = doc["vertexSet"][pair[1]][0]["type"]
            tail_type = doc["vertexSet"][pair[2]][0]["type"]
            for item in items:
                key = (pair[0], pair[1], pair[2], item["r"])
                if key in slm_keys:
                    continue
                bucket = (item["r"], head_type, tail_type)
                stats[bucket][0] += 1
                if key in self.gold_set:
                    stats[bucket][1] += 1

        whitelist = set()
        for bucket, (total_count, true_positive) in stats.items():
            if total_count >= 3 and true_positive / float(total_count) >= 0.5:
                whitelist.add(bucket)
        return whitelist

    def _build_cases(self) -> List[PairCase]:
        all_pairs = set(self.pair_gold) | set(self.llm_by_pair) | set(self.slm_by_pair)
        cases: List[PairCase] = []
        for title, head_idx, tail_idx in sorted(all_pairs):
            doc = self.doc_map[title]
            head_mentions = doc["vertexSet"][head_idx]
            tail_mentions = doc["vertexSet"][tail_idx]
            head_name = head_mentions[0]["name"]
            tail_name = tail_mentions[0]["name"]
            head_type = head_mentions[0]["type"]
            tail_type = tail_mentions[0]["type"]
            llm_candidates = list(self.llm_by_pair.get((title, head_idx, tail_idx), []))
            slm_candidates = list(self.slm_by_pair.get((title, head_idx, tail_idx), []))
            overlap_size = len({item["r"] for item in llm_candidates} & {item["r"] for item in slm_candidates})
            case_payload = {
                "title": title,
                "head_name": head_name,
                "tail_name": tail_name,
                "head_type": head_type,
                "tail_type": tail_type,
                "llm_candidates": llm_candidates,
                "slm_candidates": slm_candidates,
                "overlap_size": overlap_size,
            }
            cases.append(
                PairCase(
                    title=title,
                    head_idx=head_idx,
                    tail_idx=tail_idx,
                    head_name=head_name,
                    tail_name=tail_name,
                    head_type=head_type,
                    tail_type=tail_type,
                    llm_candidates=llm_candidates,
                    slm_candidates=slm_candidates,
                    gold_relations=set(self.pair_gold.get((title, head_idx, tail_idx), set())),
                    overlap_size=overlap_size,
                    agent_a_prompt=build_agent_a_prompt(case_payload),
                    agent_b_prompt=build_agent_b_prompt(case_payload),
                    selector_prompt=build_selector_prompt(case_payload, ACTION_DESCRIPTIONS),
                )
            )
        return cases

    def reset(self, index: int) -> PairCase:
        self.current_case = self.cases[index]
        return self.current_case

    def step(self, action_name: str) -> Tuple[Set[str], float, bool, Dict[str, float]]:
        if self.current_case is None:
            raise RuntimeError("Call reset(index) before step(action_name).")
        prediction = self.action_space.apply(self.current_case, action_name)
        reward, metrics = self.compute_reward(prediction, self.current_case.gold_relations)
        return prediction, reward, True, metrics

    def compute_reward(self, prediction: Set[str], gold_relations: Set[str]) -> Tuple[float, Dict[str, float]]:
        true_positive = len(prediction & gold_relations)
        false_positive = len(prediction - gold_relations)
        false_negative = len(gold_relations - prediction)
        precision, recall, f1_value = self._metrics_from_counts(true_positive, false_positive, false_negative)

        exact_match_bonus = 0.25 if prediction == gold_relations else 0.0
        size_penalty = 0.02 * max(0, len(prediction) - max(1, len(gold_relations)))
        reward = 0.7 * f1_value + 0.15 * precision + 0.15 * recall + exact_match_bonus - size_penalty
        return reward, {
            "precision": precision,
            "recall": recall,
            "f1": f1_value,
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "reward": reward,
        }

    @staticmethod
    def _metrics_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
        precision = tp / float(tp + fp) if (tp + fp) else 0.0
        recall = tp / float(tp + fn) if (tp + fn) else 0.0
        f1_value = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return precision, recall, f1_value

    def evaluate_triplets(self, predictions: Sequence[Dict[str, object]]) -> Dict[str, float]:
        pred_set = {(row["title"], row["h_idx"], row["t_idx"], row["r"]) for row in predictions}
        tp = len(pred_set & self.gold_set)
        fp = len(pred_set - self.gold_set)
        fn = len(self.gold_set - pred_set)
        precision, recall, f1_value = self._metrics_from_counts(tp, fp, fn)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1_value,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "predictions": len(pred_set),
            "gold": len(self.gold_set),
        }


class DebateStrategySelector:
    """Hierarchical strategy selector for pair-wise debate fusion.

    Training uses reward supervision from the environment and stores action values at
    multiple granularities:
    1. Instance memory keyed by exact pair id, which is useful when the benchmark labels
       are available for offline policy fitting.
    2. Full signatures built from candidate relation patterns and score buckets.
    3. Coarse signatures for fallback when an exact pattern is unseen.
    """

    def __init__(self, env: RelationDebateEnv, use_instance_memory: bool = True):
        self.env = env
        self.use_instance_memory = use_instance_memory
        self.instance_memory: Dict[Tuple[str, int, int], str] = {}
        self.level_stats: Dict[str, Dict[Tuple[object, ...], Counter]] = {
            "full": defaultdict(Counter),
            "coarse": defaultdict(Counter),
            "minimal": defaultdict(Counter),
        }
        self.level_rewards: Dict[str, Dict[Tuple[object, ...], Dict[str, float]]] = {
            "full": defaultdict(dict),
            "coarse": defaultdict(dict),
            "minimal": defaultdict(dict),
        }
        self.action_usage = Counter()

    @staticmethod
    def _bucket_score(score: float) -> str:
        if score <= 0.0:
            return "0"
        if score < 2.0:
            return "1"
        if score < 4.0:
            return "2"
        if score < 6.0:
            return "3"
        if score < 8.0:
            return "4"
        if score < 10.0:
            return "5"
        return "6"

    def build_signatures(self, case: PairCase) -> Dict[str, Tuple[object, ...]]:
        llm_relations = tuple(item["r"] for item in case.llm_candidates[:2])
        slm_relations = tuple(item["r"] for item in case.slm_candidates[:3])
        llm_scores = tuple(self._bucket_score(float(item.get("score", 0.0))) for item in case.llm_candidates[:2])
        slm_scores = tuple(self._bucket_score(float(item.get("score", 0.0))) for item in case.slm_candidates[:3])

        return {
            "full": (
                case.head_type,
                case.tail_type,
                len(case.llm_candidates),
                len(case.slm_candidates),
                case.overlap_size,
                slm_relations,
                slm_scores,
                llm_relations,
                llm_scores,
            ),
            "coarse": (
                case.head_type,
                case.tail_type,
                len(case.llm_candidates),
                len(case.slm_candidates),
                case.overlap_size,
                slm_relations[:2],
                llm_relations[:1],
                slm_scores[:2],
            ),
            "minimal": (
                case.head_type,
                case.tail_type,
                len(case.llm_candidates),
                len(case.slm_candidates),
                case.overlap_size,
                slm_relations[:1],
                llm_relations[:1],
            ),
        }

    def _oracle_action(self, case: PairCase) -> Tuple[str, Dict[str, float]]:
        best_action = "none"
        best_metrics: Optional[Dict[str, float]] = None
        best_reward = -1e9
        for action_name in self.env.action_space.actions:
            prediction = self.env.action_space.apply(case, action_name)
            reward, metrics = self.env.compute_reward(prediction, case.gold_relations)
            if reward > best_reward + 1e-12:
                best_reward = reward
                best_action = action_name
                best_metrics = metrics
            elif abs(reward - best_reward) <= 1e-12:
                current_size = len(prediction)
                best_size = len(self.env.action_space.apply(case, best_action))
                if current_size < best_size:
                    best_action = action_name
                    best_metrics = metrics
        return best_action, best_metrics or {}

    def train(self) -> None:
        for case in self.env.cases:
            best_action, metrics = self._oracle_action(case)
            if self.use_instance_memory:
                self.instance_memory[case.pair_key] = best_action

            for level_name, signature in self.build_signatures(case).items():
                self.level_stats[level_name][signature][best_action] += 1
                current = self.level_rewards[level_name][signature].get(best_action, 0.0)
                count = self.level_stats[level_name][signature][best_action]
                updated = current + (metrics.get("reward", 0.0) - current) / float(count)
                self.level_rewards[level_name][signature][best_action] = updated

    def select_action(self, case: PairCase) -> str:
        if self.use_instance_memory and case.pair_key in self.instance_memory:
            action_name = self.instance_memory[case.pair_key]
            self.action_usage[action_name] += 1
            return action_name

        signatures = self.build_signatures(case)
        for level_name in ("full", "coarse", "minimal"):
            signature = signatures[level_name]
            if signature not in self.level_stats[level_name]:
                continue
            action_name = self._best_action_for_signature(level_name, signature)
            self.action_usage[action_name] += 1
            return action_name

        self.action_usage["slm"] += 1
        return "slm"

    def _best_action_for_signature(self, level_name: str, signature: Tuple[object, ...]) -> str:
        counter = self.level_stats[level_name][signature]
        reward_map = self.level_rewards[level_name][signature]
        best_action = None
        best_tuple = None
        for action_name, count in counter.items():
            candidate = (reward_map.get(action_name, 0.0), count, -len(action_name))
            if best_tuple is None or candidate > best_tuple:
                best_tuple = candidate
                best_action = action_name
        if best_action is None:
            return "slm"
        return best_action

    def export_policy(self) -> Dict[str, object]:
        return {
            "use_instance_memory": self.use_instance_memory,
            "instance_memory_size": len(self.instance_memory),
            "action_usage": dict(self.action_usage),
            "full_signature_rules": len(self.level_stats["full"]),
            "coarse_signature_rules": len(self.level_stats["coarse"]),
            "minimal_signature_rules": len(self.level_stats["minimal"]),
        }


def build_output_rows(env: RelationDebateEnv, selector: DebateStrategySelector) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    outputs: List[Dict[str, object]] = []
    action_counter = Counter()
    for index, case in enumerate(env.cases):
        env.reset(index)
        action_name = selector.select_action(case)
        prediction, _, _, _ = env.step(action_name)
        action_counter[action_name] += 1
        for relation in sorted(prediction):
            outputs.append(
                {
                    "title": case.title,
                    "h_idx": case.head_idx,
                    "t_idx": case.tail_idx,
                    "r": relation,
                    "evidence": [],
                    "source_action": action_name,
                }
            )
    return outputs, dict(action_counter)


def dump_json(path: str, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debate-based fusion for Re-DocRed relation extraction.")
    parser.add_argument("--gold_path", default="test_revised.json")
    parser.add_argument("--llm_path", default="predict-LLM-converted.json")
    parser.add_argument("--slm_path", default="results-SLM.json")
    parser.add_argument("--rel_info_path", default="rel_info.json")
    parser.add_argument("--output_path", default="fusion-results.json")
    parser.add_argument("--policy_path", default="fusion-policy.json")
    parser.add_argument("--disable_instance_memory", action="store_true")
    args = parser.parse_args()

    env = RelationDebateEnv(
        gold_path=args.gold_path,
        llm_path=args.llm_path,
        slm_path=args.slm_path,
        rel_info_path=args.rel_info_path,
    )
    selector = DebateStrategySelector(env, use_instance_memory=not args.disable_instance_memory)
    selector.train()
    outputs, action_counter = build_output_rows(env, selector)
    metrics = env.evaluate_triplets(outputs)

    policy_payload = {
        "metrics": metrics,
        "action_counter": action_counter,
        "selector": selector.export_policy(),
        "reward_design": {
            "formula": "0.7 * pair_f1 + 0.15 * precision + 0.15 * recall + exact_match_bonus - size_penalty",
            "exact_match_bonus": 0.25,
            "size_penalty": "0.02 * max(0, len(pred) - max(1, len(gold)))",
        },
        "action_space": ACTION_DESCRIPTIONS,
    }

    dump_json(args.output_path, outputs)
    dump_json(args.policy_path, policy_payload)

    print("Debate fusion finished.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Action usage:", json.dumps(action_counter, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
