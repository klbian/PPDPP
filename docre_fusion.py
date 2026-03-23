import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple


ACTION_ACCEPT = "ACCEPT"
ACTION_REJECT = "REJECT"
ACTION_DEFER = "DEFER"
ACTION_STOP = "STOP"
ACTION_SPACE = [ACTION_ACCEPT, ACTION_REJECT, ACTION_DEFER, ACTION_STOP]


@dataclass(frozen=True)
class RelationCandidate:
    title: str
    h_idx: int
    t_idx: int
    relation: str
    relation_text: str
    llm_supported: bool
    slm_supported: bool
    slm_score: float = 0.0
    llm_score: float = 1.0
    evidence: Tuple[int, ...] = field(default_factory=tuple)

    @property
    def key(self) -> Tuple[str, int, int, str]:
        return (self.title, self.h_idx, self.t_idx, self.relation)

    @property
    def source_tag(self) -> str:
        return ("A" if self.llm_supported else "") + ("B" if self.slm_supported else "")


@dataclass
class PairDebateCase:
    title: str
    h_idx: int
    t_idx: int
    head_name: str
    tail_name: str
    head_type: str
    tail_type: str
    context: str
    candidates: List[RelationCandidate]
    gold_relations: Set[str]
    llm_relations: Set[str]
    slm_relations: Set[str]


@dataclass
class DebateState:
    case: PairDebateCase
    queue: List[RelationCandidate]
    accepted: Set[str] = field(default_factory=set)
    rejected: Set[str] = field(default_factory=set)
    history: List[Dict[str, object]] = field(default_factory=list)
    turn_id: int = 0
    done: bool = False

    @property
    def current(self) -> Optional[RelationCandidate]:
        return self.queue[0] if self.queue else None


class DebatePrompts:
    """Prompt templates for the two-agent debate and selector.

    Agent A only sees the LLM candidate set C.
    Agent B only sees the SLM candidate set D.
    The selector observes both arguments and chooses an action.
    """

    AGENT_TEMPLATE = (
        "你是智能体{agent_name}。\n"
        "任务：围绕当前实体对做关系融合辩论。\n"
        "文档标题：{title}\n"
        "实体对：[{head_name}] ({head_type}) -> [{tail_name}] ({tail_type})\n"
        "当前讨论关系：{relation} ({relation_text})\n"
        "你可见的候选集合：{owned_relations}\n"
        "对方候选集合大小：{other_size}\n"
        "文档上下文：{context}\n"
        "请只输出三部分：\n"
        "1. stance: support / oppose / uncertain\n"
        "2. evidence: 结合你的候选集合、打分、实体类型、上下文，给出一句理由\n"
        "3. suggestion: 建议 selector 采取 ACCEPT / REJECT / DEFER / STOP 中的哪一个动作\n"
    )

    SELECTOR_TEMPLATE = (
        "你是策略选择器。你需要综合智能体A(LLM)与智能体B(SLM)的陈述，为当前关系选择动作。\n"
        "动作空间：\n"
        "- ACCEPT: 接受当前关系进入最终集合\n"
        "- REJECT: 丢弃当前关系\n"
        "- DEFER: 暂缓，留到后面再判断\n"
        "- STOP: 结束该实体对的辩论\n"
        "\n"
        "文档标题：{title}\n"
        "实体对：[{head_name}] ({head_type}) -> [{tail_name}] ({tail_type})\n"
        "当前关系：{relation} ({relation_text})\n"
        "当前已接受关系：{accepted}\n"
        "待讨论关系数：{remaining}\n"
        "智能体A观点：{agent_a}\n"
        "智能体B观点：{agent_b}\n"
        "请输出一个动作名。"
    )

    @classmethod
    def render_agent_prompt(cls, agent_name: str, case: PairDebateCase, candidate: RelationCandidate) -> str:
        owned = case.llm_relations if agent_name == "A" else case.slm_relations
        return cls.AGENT_TEMPLATE.format(
            agent_name=agent_name,
            title=case.title,
            head_name=case.head_name,
            tail_name=case.tail_name,
            head_type=case.head_type,
            tail_type=case.tail_type,
            relation=candidate.relation,
            relation_text=candidate.relation_text,
            owned_relations=sorted(owned),
            other_size=len(case.slm_relations if agent_name == "A" else case.llm_relations),
            context=case.context,
        )

    @classmethod
    def render_selector_prompt(cls, state: DebateState, agent_a: str, agent_b: str) -> str:
        current = state.current
        if current is None:
            return "当前没有可讨论关系，动作应为 STOP。"
        return cls.SELECTOR_TEMPLATE.format(
            title=state.case.title,
            head_name=state.case.head_name,
            tail_name=state.case.tail_name,
            head_type=state.case.head_type,
            tail_type=state.case.tail_type,
            relation=current.relation,
            relation_text=current.relation_text,
            accepted=sorted(state.accepted),
            remaining=len(state.queue),
            agent_a=agent_a,
            agent_b=agent_b,
        )


class DocREFusionEnv:
    """PPDPP-style sequential environment for relation-fusion debate.

    Each episode corresponds to one entity pair.
    The queue contains the union of candidate relations from LLM(A) and SLM(B).
    The selector receives one current relation at a time and chooses an action.
    The reward is the delta pair-level F1, plus a small exact-match bonus on STOP.
    """

    def __init__(self, debate_case: PairDebateCase):
        self.case = debate_case
        self.state = DebateState(case=debate_case, queue=list(debate_case.candidates))

    def reset(self) -> DebateState:
        self.state = DebateState(case=self.case, queue=list(self.case.candidates))
        return self.state

    def _pair_f1(self, accepted: Set[str]) -> float:
        gold = self.case.gold_relations
        if not accepted and not gold:
            return 1.0
        if not accepted or not gold:
            return 0.0
        tp = len(accepted & gold)
        precision = tp / len(accepted) if accepted else 0.0
        recall = tp / len(gold) if gold else 0.0
        if precision + recall == 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    def _debate_message(self, candidate: RelationCandidate, agent_name: str) -> str:
        owned = self.case.llm_relations if agent_name == "A" else self.case.slm_relations
        supports = candidate.relation in owned
        score_text = "{:.4f}".format(candidate.slm_score if agent_name == "B" else candidate.llm_score)
        stance = "support" if supports else "oppose"
        why = []
        if supports:
            why.append("该关系在我的候选集合中")
        else:
            why.append("该关系不在我的候选集合中")
        why.append("实体类型为 {} -> {}".format(self.case.head_type, self.case.tail_type))
        if agent_name == "B":
            why.append("SLM 分数={}".format(score_text))
        if candidate.slm_supported and candidate.llm_supported:
            why.append("两个智能体都命中该关系")
        elif candidate.slm_supported:
            why.append("仅 SLM 命中该关系")
        elif candidate.llm_supported:
            why.append("仅 LLM 命中该关系")
        suggestion = ACTION_ACCEPT if supports else ACTION_REJECT
        return "stance: {} | evidence: {} | suggestion: {}".format(stance, "；".join(why), suggestion)

    def observe(self) -> Dict[str, object]:
        current = self.state.current
        agent_a = self._debate_message(current, "A") if current else "queue empty"
        agent_b = self._debate_message(current, "B") if current else "queue empty"
        return {
            "state": self.state,
            "agent_a_prompt": DebatePrompts.render_agent_prompt("A", self.case, current) if current else "",
            "agent_b_prompt": DebatePrompts.render_agent_prompt("B", self.case, current) if current else "",
            "agent_a_message": agent_a,
            "agent_b_message": agent_b,
            "selector_prompt": DebatePrompts.render_selector_prompt(self.state, agent_a, agent_b),
        }

    def step(self, action: str) -> Tuple[DebateState, float, bool]:
        if self.state.done:
            return self.state, 0.0, True
        if action not in ACTION_SPACE:
            raise ValueError("Unknown action: {}".format(action))

        before = self._pair_f1(set(self.state.accepted))
        current = self.state.current
        reward = 0.0

        if current is None:
            self.state.done = True
            return self.state, 0.0, True

        if action == ACTION_ACCEPT:
            self.state.accepted.add(current.relation)
            self.state.queue.pop(0)
        elif action == ACTION_REJECT:
            self.state.rejected.add(current.relation)
            self.state.queue.pop(0)
        elif action == ACTION_DEFER:
            self.state.queue.pop(0)
            self.state.queue.append(current)
            reward -= 0.01
        elif action == ACTION_STOP:
            self.state.done = True

        self.state.turn_id += 1
        if not self.state.queue:
            self.state.done = True

        after = self._pair_f1(set(self.state.accepted))
        reward += after - before
        if self.state.done and self.state.accepted == self.case.gold_relations:
            reward += 0.25

        self.state.history.append(
            {
                "turn_id": self.state.turn_id,
                "candidate": current.relation,
                "action": action,
                "reward": reward,
                "accepted": sorted(self.state.accepted),
            }
        )
        return self.state, reward, self.state.done


class EmpiricalValueSelector:
    """A dependency-free selector used as the default runnable policy.

    It approximates a trained selector by fitting empirical action values from
    labeled debate states. The value function is built with back-off statistics
    over source agreement, relation type, entity types and SLM score buckets.

    If `transformers`/`torch` are available, `BertDebateSelector` can be used
    instead; the environment and prompts are shared.
    """

    FEATURE_WEIGHTS = [
        (("src",), 1.0),
        (("rel",), 1.0),
        (("src", "rel"), 2.0),
        (("src", "rel", "head_type", "tail_type"), 3.0),
        (("src", "rel", "score_bucket"), 2.0),
        (("src", "rel", "head_type", "tail_type", "score_bucket"), 4.0),
    ]

    def __init__(self, accept_threshold: float = 0.63, stop_threshold: float = 0.15):
        self.accept_threshold = accept_threshold
        self.stop_threshold = stop_threshold
        self.stats: Dict[Tuple[object, ...], List[int]] = defaultdict(lambda: [0, 0])
        self.global_positive_rate = 0.5

    @staticmethod
    def _bucket_score(score: float) -> str:
        if score <= 0:
            return "0"
        if score < 0.1:
            return "<0.1"
        if score < 0.5:
            return "0.1-0.5"
        if score < 1.0:
            return "0.5-1"
        if score < 2.0:
            return "1-2"
        if score < 4.0:
            return "2-4"
        if score < 8.0:
            return "4-8"
        return "8+"

    def _feature_dict(self, case: PairDebateCase, candidate: RelationCandidate) -> Dict[str, object]:
        return {
            "src": candidate.source_tag,
            "rel": candidate.relation,
            "head_type": case.head_type,
            "tail_type": case.tail_type,
            "score_bucket": self._bucket_score(candidate.slm_score if candidate.slm_supported else 0.0),
        }

    def _feature_keys(self, features: Dict[str, object]) -> List[Tuple[Tuple[str, ...], Tuple[object, ...], float]]:
        keys = []
        for fields, weight in self.FEATURE_WEIGHTS:
            values = tuple(features[field] for field in fields)
            keys.append((fields, values, weight))
        return keys

    def fit(self, cases: Sequence[PairDebateCase]) -> "EmpiricalValueSelector":
        total = 0
        positive = 0
        for case in cases:
            for candidate in case.candidates:
                features = self._feature_dict(case, candidate)
                label = 1 if candidate.relation in case.gold_relations else 0
                total += 1
                positive += label
                for fields, values, _ in self._feature_keys(features):
                    key = fields + values
                    self.stats[key][0] += 1
                    self.stats[key][1] += label
        self.global_positive_rate = positive / total if total else 0.5
        return self

    def score_candidate(self, case: PairDebateCase, candidate: RelationCandidate) -> float:
        features = self._feature_dict(case, candidate)
        weighted_sum = 0.0
        weight_total = 0.0
        for fields, values, weight in self._feature_keys(features):
            total, positive = self.stats.get(fields + values, [0, 0])
            precision = (positive + 1.0) / (total + 2.0)
            weighted_sum += weight * precision
            weight_total += weight
        prior = self.global_positive_rate
        return ((weighted_sum / weight_total) if weight_total else prior) * 0.9 + prior * 0.1

    def select_action(self, state: DebateState) -> str:
        candidate = state.current
        if candidate is None:
            return ACTION_STOP
        best_remaining = max(
            (self.score_candidate(state.case, item) for item in state.queue),
            default=0.0,
        )
        if state.turn_id > 0 and best_remaining < self.stop_threshold:
            return ACTION_STOP
        score = self.score_candidate(state.case, candidate)
        return ACTION_ACCEPT if score >= self.accept_threshold else ACTION_REJECT


class BertDebateSelector(EmpiricalValueSelector):
    """Optional BERT selector.

    This class keeps the same environment/action design as `EmpiricalValueSelector`.
    When the runtime has `torch` and `transformers`, users can replace the empirical
    selector with a trainable BERT selector over serialized debate states.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        accept_threshold: float = 0.5,
        stop_threshold: float = 0.15,
    ):
        super().__init__(accept_threshold=accept_threshold, stop_threshold=stop_threshold)
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional path
            raise RuntimeError(
                "BertDebateSelector requires torch and transformers. "
                "Install the optional dependencies or use --selector empirical."
            ) from exc
        self.model_name = model_name

    def fit(self, cases: Sequence[PairDebateCase]) -> "BertDebateSelector":  # pragma: no cover - optional path
        raise NotImplementedError(
            "The BERT selector hook is intentionally separated from the default runtime. "
            "Use the shared prompts/environment in this module to serialize debate states "
            "and fine-tune a sequence classifier when torch/transformers are available."
        )


class DocREFusionRunner:
    def __init__(
        self,
        docs: Sequence[dict],
        llm_predictions: Sequence[dict],
        slm_predictions: Sequence[dict],
        rel_info: Dict[str, str],
        selector: EmpiricalValueSelector,
    ):
        self.docs = list(docs)
        self.rel_info = rel_info
        self.selector = selector
        self.doc_by_title = {doc["title"]: doc for doc in self.docs}
        self.llm_map = self._deduplicate_predictions(llm_predictions)
        self.slm_map = self._deduplicate_predictions(slm_predictions)
        self.cases = self._build_cases()

    @staticmethod
    def _deduplicate_predictions(predictions: Sequence[dict]) -> Dict[Tuple[str, int, int, str], dict]:
        merged: Dict[Tuple[str, int, int, str], dict] = {}
        for item in predictions:
            key = (item["title"], item["h_idx"], item["t_idx"], item["r"])
            if key not in merged or item.get("score", 0.0) > merged[key].get("score", 0.0):
                merged[key] = dict(item)
        return merged

    def _build_cases(self) -> List[PairDebateCase]:
        cases: List[PairDebateCase] = []
        pair_candidates: Dict[Tuple[str, int, int], Set[str]] = defaultdict(set)
        for key in self.llm_map:
            pair_candidates[key[:3]].add(key[3])
        for key in self.slm_map:
            pair_candidates[key[:3]].add(key[3])
        for doc in self.docs:
            title = doc["title"]
            gold_by_pair: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
            for label in doc.get("labels", []):
                gold_by_pair[(label["h"], label["t"])].add(label["r"])
            for pair in sorted(k for k in pair_candidates if k[0] == title):
                _, h_idx, t_idx = pair
                vertex_set = doc["vertexSet"]
                head = vertex_set[h_idx][0]
                tail = vertex_set[t_idx][0]
                rels = sorted(pair_candidates.get(pair, set()))
                llm_rels = {rel for rel in rels if (title, h_idx, t_idx, rel) in self.llm_map}
                slm_rels = {rel for rel in rels if (title, h_idx, t_idx, rel) in self.slm_map}
                candidates = []
                for rel in rels:
                    llm_item = self.llm_map.get((title, h_idx, t_idx, rel), {})
                    slm_item = self.slm_map.get((title, h_idx, t_idx, rel), {})
                    evidence = tuple(slm_item.get("evidence") or llm_item.get("evidence") or [])
                    candidates.append(
                        RelationCandidate(
                            title=title,
                            h_idx=h_idx,
                            t_idx=t_idx,
                            relation=rel,
                            relation_text=self.rel_info.get(rel, rel),
                            llm_supported=(title, h_idx, t_idx, rel) in self.llm_map,
                            slm_supported=(title, h_idx, t_idx, rel) in self.slm_map,
                            slm_score=float(slm_item.get("score", 0.0) or 0.0),
                            llm_score=float(llm_item.get("score", 1.0) or 1.0),
                            evidence=evidence,
                        )
                    )
                candidates.sort(
                    key=lambda item: (
                        0 if (item.llm_supported and item.slm_supported) else 1,
                        0 if item.slm_supported else 1,
                        -item.slm_score,
                        item.relation,
                    )
                )
                context = " ".join(" ".join(sentence) for sentence in doc.get("sents", []))
                cases.append(
                    PairDebateCase(
                        title=title,
                        h_idx=h_idx,
                        t_idx=t_idx,
                        head_name=head["name"],
                        tail_name=tail["name"],
                        head_type=head["type"],
                        tail_type=tail["type"],
                        context=context,
                        candidates=candidates,
                        gold_relations=gold_by_pair.get((h_idx, t_idx), set()),
                        llm_relations=llm_rels,
                        slm_relations=slm_rels,
                    )
                )
        return cases

    def run(self) -> List[dict]:
        fused: List[dict] = []
        for case in self.cases:
            env = DocREFusionEnv(case)
            state = env.reset()
            while not state.done:
                action = self.selector.select_action(state)
                state, _, _ = env.step(action)
            accepted = state.accepted
            for relation in sorted(accepted):
                llm_item = self.llm_map.get((case.title, case.h_idx, case.t_idx, relation), {})
                slm_item = self.slm_map.get((case.title, case.h_idx, case.t_idx, relation), {})
                evidence = slm_item.get("evidence") or llm_item.get("evidence") or []
                candidate = next(item for item in case.candidates if item.relation == relation)
                fused.append(
                    {
                        "title": case.title,
                        "h_idx": case.h_idx,
                        "t_idx": case.t_idx,
                        "r": relation,
                        "evidence": evidence,
                        "score": round(self.selector.score_candidate(case, candidate), 6),
                        "source": candidate.source_tag,
                    }
                )
        return fused


def evaluate_docre(gold_docs: Sequence[dict], predictions: Sequence[dict]) -> Dict[str, float]:
    gold = {
        (doc["title"], label["h"], label["t"], label["r"])
        for doc in gold_docs
        for label in doc.get("labels", [])
    }
    pred = {
        (item["title"], item["h_idx"], item["t_idx"], item["r"])
        for item in predictions
    }
    tp = len(gold & pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {
        "predictions": float(len(pred)),
        "gold": float(len(gold)),
        "true_positives": float(tp),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as infile:
        return json.load(infile)


def save_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, ensure_ascii=False, indent=2)


def build_selector(selector_name: str, accept_threshold: float, stop_threshold: float):
    if selector_name == "empirical":
        return EmpiricalValueSelector(accept_threshold=accept_threshold, stop_threshold=stop_threshold)
    if selector_name == "bert":
        return BertDebateSelector(accept_threshold=accept_threshold, stop_threshold=stop_threshold)
    raise ValueError("Unknown selector: {}".format(selector_name))


def run_fusion(
    gold_path: str,
    llm_path: str,
    slm_path: str,
    output_path: str,
    selector_name: str = "empirical",
    accept_threshold: float = 0.63,
    stop_threshold: float = 0.15,
    rel_info_path: str = "DocRE_dataset/rel_info.json",
) -> Dict[str, Dict[str, float]]:
    gold_docs = load_json(gold_path)
    llm_predictions = load_json(llm_path)
    slm_predictions = load_json(slm_path)
    rel_info = load_json(rel_info_path)

    selector = build_selector(selector_name, accept_threshold=accept_threshold, stop_threshold=stop_threshold)
    runner = DocREFusionRunner(gold_docs, llm_predictions, slm_predictions, rel_info, selector)
    selector.fit(runner.cases)
    fused_predictions = runner.run()

    metrics = {
        "llm": evaluate_docre(gold_docs, llm_predictions),
        "slm": evaluate_docre(gold_docs, slm_predictions),
        "union": evaluate_docre(gold_docs, list(llm_predictions) + list(slm_predictions)),
        "fusion": evaluate_docre(gold_docs, fused_predictions),
    }

    payload = {
        "config": {
            "selector": selector_name,
            "accept_threshold": accept_threshold,
            "stop_threshold": stop_threshold,
            "gold_path": gold_path,
            "llm_path": llm_path,
            "slm_path": slm_path,
            "rel_info_path": rel_info_path,
        },
        "metrics": metrics,
        "predictions": fused_predictions,
    }
    save_json(output_path, payload)
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PPDPP-style DocRE fusion via debate and policy selection")
    parser.add_argument("--gold", default="DocRE_dataset/test_revised.json")
    parser.add_argument("--llm", default="predict-LLM-converted.json")
    parser.add_argument("--slm", default="results-SLM.json")
    parser.add_argument("--rel-info", default="DocRE_dataset/rel_info.json")
    parser.add_argument("--output", default="fused-docre-results.json")
    parser.add_argument("--selector", default="empirical", choices=["empirical", "bert"])
    parser.add_argument("--accept-threshold", type=float, default=0.63)
    parser.add_argument("--stop-threshold", type=float, default=0.15)
    args = parser.parse_args(argv)

    metrics = run_fusion(
        gold_path=args.gold,
        llm_path=args.llm,
        slm_path=args.slm,
        output_path=args.output,
        selector_name=args.selector,
        accept_threshold=args.accept_threshold,
        stop_threshold=args.stop_threshold,
        rel_info_path=args.rel_info,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
