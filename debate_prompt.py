"""Prompt templates for pair-wise debate-based relation fusion on Re-DocRed.

The prompts are designed for a two-agent setting:
- Agent A only sees the LLM candidate list C.
- Agent B only sees the SLM candidate list D.
- A strategy selector observes the pair state and chooses an action.

The implementation in :mod:`debate_fusion` uses deterministic prompt rendering so the
pipeline can run without external LLM dependencies, while keeping the debate design
explicit and reusable.
"""

from typing import Dict, Iterable, List


def _format_relations(relations: Iterable[Dict[str, object]]) -> str:
    rows: List[str] = []
    for item in relations:
        rel_id = item.get("r", "NA")
        rel_name = item.get("relation_text", rel_id)
        score = item.get("score")
        score_text = "NA" if score is None else f"{float(score):.4f}"
        rows.append(f"- {rel_id} ({rel_name}), score={score_text}")
    return "\n".join(rows) if rows else "- <empty>"


def build_agent_a_prompt(case: Dict[str, object]) -> str:
    return (
        "You are Agent A in a document-level relation extraction debate.\n"
        "You ONLY trust the LLM candidate set C and must argue which relations should be kept.\n"
        "Focus on recall: defend plausible missing positives, but avoid relations that clearly violate"
        " the entity types or pair direction.\n\n"
        f"Document title: {case['title']}\n"
        f"Head entity: {case['head_name']} ({case['head_type']})\n"
        f"Tail entity: {case['tail_name']} ({case['tail_type']})\n"
        f"LLM candidates C:\n{_format_relations(case['llm_candidates'])}\n\n"
        "Required output format:\n"
        "1. Strong keep relations\n2. Weak keep relations\n3. Relations to reject\n"
        "4. Short reason grounded in candidate confidence, entity types, and directionality."
    )


def build_agent_b_prompt(case: Dict[str, object]) -> str:
    return (
        "You are Agent B in a document-level relation extraction debate.\n"
        "You ONLY trust the SLM candidate set D and must argue which relations should be kept.\n"
        "Focus on precision: prefer high-confidence relations and prune noisy long-tail guesses.\n\n"
        f"Document title: {case['title']}\n"
        f"Head entity: {case['head_name']} ({case['head_type']})\n"
        f"Tail entity: {case['tail_name']} ({case['tail_type']})\n"
        f"SLM candidates D:\n{_format_relations(case['slm_candidates'])}\n\n"
        "Required output format:\n"
        "1. High-confidence keep relations\n2. Borderline relations\n3. Relations to prune\n"
        "4. Short reason grounded in score, agreement, entity types, and list consistency."
    )


def build_selector_prompt(case: Dict[str, object], action_descriptions: Dict[str, str]) -> str:
    action_text = "\n".join(f"- {name}: {desc}" for name, desc in action_descriptions.items())
    return (
        "You are the strategy selector.\n"
        "Your job is to choose one action after reading Agent A and Agent B's arguments.\n"
        "Optimize pair-level F1: preserve true positives from both sides, but aggressively suppress"
        " extra false positives.\n\n"
        f"Document title: {case['title']}\n"
        f"Head entity: {case['head_name']} ({case['head_type']})\n"
        f"Tail entity: {case['tail_name']} ({case['tail_type']})\n"
        f"Overlap size: {case['overlap_size']}\n"
        f"LLM candidate count: {len(case['llm_candidates'])}\n"
        f"SLM candidate count: {len(case['slm_candidates'])}\n\n"
        "Available actions:\n"
        f"{action_text}\n\n"
        "Decision rule:\n"
        "- prefer SLM-centered actions when D is high confidence and coherent;\n"
        "- prefer union-like actions only when A contributes a small, type-compatible recall gain;\n"
        "- prefer intersection/none when both sides are noisy or contradictory;\n"
        "- use top-k truncation to keep only the most reliable relations for the pair."
    )
