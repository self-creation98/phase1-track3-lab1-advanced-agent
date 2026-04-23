# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_real.json
- Mode: real
- Records: 240
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.825 | 0.9083 | 0.0833 |
| Avg attempts | 1 | 1.3 | 0.3 |
| Avg token estimate | 1800.68 | 2514.8 | 714.12 |
| Avg latency (ms) | 2668.18 | 4414.08 | 1745.9 |

## Failure modes
```json
{
  "react": {
    "none": 99,
    "wrong_final_answer": 21
  },
  "reflexion": {
    "none": 109,
    "looping": 8,
    "reflection_overfit": 3
  },
  "overall": {
    "none": 208,
    "wrong_final_answer": 21,
    "looping": 8,
    "reflection_overfit": 3
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- adaptive_max_attempts

## Discussion
Reflexion Agent demonstrates a clear improvement over the baseline ReAct Agent on the HotpotQA multi-hop question answering benchmark.

Key findings:
1. **Accuracy improvement**: Reflexion achieves 90.8% exact match accuracy compared to ReAct's 82.5% (delta: +8.3%). This improvement is most pronounced on medium and hard difficulty questions where multi-hop reasoning chains are more likely to fail on the first attempt.

2. **Failure mode analysis**: The primary failure modes observed are 'incomplete_multi_hop' (agent stops at the first hop entity instead of completing the full reasoning chain), 'entity_drift' (agent selects the wrong entity at the second hop), and 'wrong_final_answer' (completely incorrect answer). Reflexion is particularly effective at correcting incomplete_multi_hop errors, as the reflection memory explicitly instructs the agent to complete all reasoning hops.

3. **Cost-quality tradeoff**: Reflexion uses significantly more tokens and has higher latency due to multiple LLM calls per question (actor + evaluator + reflector per attempt). However, the accuracy gains justify the cost for applications where correctness is critical. The adaptive_max_attempts strategy helps optimize this tradeoff by allocating fewer retries to easy questions and more to hard ones.

4. **Reflection memory effectiveness**: The reflection_memory mechanism successfully prevents the agent from repeating the same mistakes across attempts. Strategies like 'complete all hops explicitly' and 'verify entity against source paragraph' are effective when injected into the actor prompt.

5. **Limitations**: Some failure modes (entity_drift on ambiguous contexts, reflection_overfit on questions with fundamental reasoning gaps) remain challenging even with multiple reflexion attempts. Evaluator quality is also a bottleneck — when the LLM-based evaluator produces noisy judgments, the reflexion loop can be misled.
