"""Reflexion Agent — triển khai đầy đủ ReAct và Reflexion Agent với LLM thật."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .llm_runtime import actor_answer, evaluator, reflector
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord


def _classify_failure_mode(
    judge_reason: str,
    traces: list[AttemptTrace],
    agent_type: str,
) -> str:
    """Phân loại failure mode dựa trên judge reason và trace history."""
    reason_lower = judge_reason.lower()

    # Looping: cùng answer lặp lại qua các attempt
    if len(traces) >= 2:
        answers = [t.answer for t in traces]
        if len(set(answers)) == 1:
            return "looping"

    # Reflection overfit: answer thay đổi nhưng vẫn sai sau nhiều attempt (reflexion only)
    if agent_type == "reflexion" and len(traces) >= 3:
        if all(t.score == 0 for t in traces):
            return "reflection_overfit"

    # Incomplete multi-hop: chỉ hoàn thành 1 hop
    if any(kw in reason_lower for kw in ["first hop", "partial", "intermediate", "incomplete", "one hop", "stopped"]):
        return "incomplete_multi_hop"

    # Entity drift: entity sai ở hop 2
    if any(kw in reason_lower for kw in ["drift", "wrong entity", "wrong second", "incorrect entity"]):
        return "entity_drift"

    return "wrong_final_answer"


# Bonus: adaptive_max_attempts — điều chỉnh số lần retry theo difficulty
def _get_adaptive_max_attempts(difficulty: str, base_max: int) -> int:
    """Điều chỉnh max_attempts dựa trên độ khó câu hỏi."""
    multiplier = {"easy": 0.67, "medium": 1.0, "hard": 1.33}
    return max(1, round(base_max * multiplier.get(difficulty, 1.0)))


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    adaptive_attempts: bool = False  # Bonus: adaptive_max_attempts

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        total_tokens = 0
        total_latency = 0

        # Bonus: adaptive_max_attempts
        max_att = (
            _get_adaptive_max_attempts(example.difficulty, self.max_attempts)
            if self.adaptive_attempts
            else self.max_attempts
        )

        for attempt_id in range(1, max_att + 1):
            # --- Actor: gọi LLM để trả lời ---
            actor_resp = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            answer = actor_resp.result
            total_tokens += actor_resp.tokens_used
            total_latency += actor_resp.latency_ms

            # --- Evaluator: gọi LLM để chấm điểm ---
            eval_resp = evaluator(example, answer)
            judge = eval_resp.result
            total_tokens += eval_resp.tokens_used
            total_latency += eval_resp.latency_ms

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                token_estimate=actor_resp.tokens_used + eval_resp.tokens_used,
                latency_ms=actor_resp.latency_ms + eval_resp.latency_ms,
            )

            final_answer = answer
            final_score = judge.score

            if judge.score == 1:
                traces.append(trace)
                break

            # --- Reflexion loop ---
            if self.agent_type == "reflexion" and attempt_id < max_att:
                ref_resp = reflector(example, attempt_id, judge)
                ref_entry = ref_resp.result
                total_tokens += ref_resp.tokens_used
                total_latency += ref_resp.latency_ms

                # Cập nhật reflection memory cho Actor dùng lần sau
                reflection_memory.append(ref_entry.next_strategy)
                reflections.append(ref_entry)
                trace.reflection = ref_entry
                # Cộng thêm token/latency của reflector vào trace
                trace.token_estimate += ref_resp.tokens_used
                trace.latency_ms += ref_resp.latency_ms

            traces.append(trace)

        # Phân loại failure mode
        failure_mode = (
            "none"
            if final_score == 1
            else _classify_failure_mode(
                traces[-1].reason if traces else "",
                traces,
                self.agent_type,
            )
        )

        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )


class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)


class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3, adaptive: bool = True) -> None:
        super().__init__(
            agent_type="reflexion",
            max_attempts=max_attempts,
            adaptive_attempts=adaptive,
        )
