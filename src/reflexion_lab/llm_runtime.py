"""Real LLM Runtime — thay thế mock_runtime.py bằng OpenAI API calls."""
from __future__ import annotations
import json
import time
import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .schemas import QAExample, JudgeResult, ReflectionEntry
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@dataclass
class LLMResponse:
    """Wrapper chứa kết quả LLM cùng metadata token/latency."""
    result: Any
    tokens_used: int = 0
    latency_ms: int = 0


def _call_llm(messages: list[dict], temperature: float = 0.3) -> tuple[str, int, int]:
    """Gọi OpenAI API và trả về (content, total_tokens, latency_ms)."""
    start = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=512,
    )
    latency_ms = int((time.time() - start) * 1000)
    content = response.choices[0].message.content.strip()
    total_tokens = response.usage.total_tokens if response.usage else 0
    return content, total_tokens, latency_ms


def _parse_json(text: str) -> dict:
    """Parse JSON từ LLM response, xử lý cả trường hợp có markdown code block."""
    text = text.strip()
    if text.startswith("```"):
        # Bỏ ```json ... ``` wrapper
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> LLMResponse:
    """Gọi LLM để trả lời câu hỏi multi-hop QA."""
    context_text = "\n\n".join(
        f"### {chunk.title}\n{chunk.text}" for chunk in example.context
    )

    user_content = f"""Question: {example.question}

Context:
{context_text}"""

    if reflection_memory:
        memory_text = "\n".join(f"- Attempt {i+1}: {m}" for i, m in enumerate(reflection_memory))
        user_content += f"""

Reflection Memory (lessons from previous attempts):
{memory_text}

Use these lessons to improve your answer. Do NOT repeat previous mistakes."""

    user_content += "\n\nProvide ONLY the final short answer, nothing else."

    messages = [
        {"role": "system", "content": ACTOR_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    content, tokens, latency = _call_llm(messages, temperature=0.3)
    return LLMResponse(result=content, tokens_used=tokens, latency_ms=latency)


def evaluator(example: QAExample, answer: str) -> LLMResponse:
    """Gọi LLM để đánh giá câu trả lời so với gold answer."""
    user_content = f"""Question: {example.question}
Gold Answer: {example.gold_answer}
Predicted Answer: {answer}

Evaluate whether the predicted answer is correct. Respond in JSON format."""

    messages = [
        {"role": "system", "content": EVALUATOR_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    content, tokens, latency = _call_llm(messages, temperature=0.0)

    try:
        parsed = _parse_json(content)
        result = JudgeResult(
            score=parsed.get("score", 0),
            reason=parsed.get("reason", ""),
            missing_evidence=parsed.get("missing_evidence", []),
            spurious_claims=parsed.get("spurious_claims", []),
        )
    except (json.JSONDecodeError, Exception):
        # Fallback: nếu parse lỗi, dùng exact match
        from .utils import normalize_answer
        is_match = normalize_answer(example.gold_answer) == normalize_answer(answer)
        result = JudgeResult(
            score=1 if is_match else 0,
            reason=content if content else "Parse error, fell back to exact match.",
        )

    return LLMResponse(result=result, tokens_used=tokens, latency_ms=latency)


def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> LLMResponse:
    """Gọi LLM để phân tích lỗi và đề xuất chiến thuật mới."""
    user_content = f"""Question: {example.question}
Gold Answer: {example.gold_answer}
Failed Attempt #{attempt_id}
Evaluator Reason: {judge.reason}
Missing Evidence: {', '.join(judge.missing_evidence) if judge.missing_evidence else 'N/A'}
Spurious Claims: {', '.join(judge.spurious_claims) if judge.spurious_claims else 'N/A'}

Analyze the failure and suggest a strategy for the next attempt. Respond in JSON format."""

    messages = [
        {"role": "system", "content": REFLECTOR_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    content, tokens, latency = _call_llm(messages, temperature=0.2)

    try:
        parsed = _parse_json(content)
        result = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=parsed.get("failure_reason", judge.reason),
            lesson=parsed.get("lesson", "Review all context paragraphs."),
            next_strategy=parsed.get("next_strategy", "Re-read context and complete all hops."),
        )
    except (json.JSONDecodeError, Exception):
        result = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="Failed to parse reflection, using default.",
            next_strategy="Re-read all context paragraphs carefully and complete every reasoning hop.",
        )

    return LLMResponse(result=result, tokens_used=tokens, latency_ms=latency)
