"""Script to download HotpotQA from HuggingFace and convert to project format."""
from __future__ import annotations
import json
import random
from pathlib import Path

from datasets import load_dataset


def prepare_hotpot_dataset(
    output_path: str = "data/hotpot_real.json",
    easy_count: int = 40,
    medium_count: int = 40,
    hard_count: int = 40,
    seed: int = 42,
) -> None:
    """Load HotpotQA and sample balanced by difficulty."""
    print("Loading HotpotQA from HuggingFace (train + validation)...")

    # Validation only has 'hard', so we use train split too for easy/medium
    train_ds = load_dataset("hotpot_qa", "distractor", split="train")
    val_ds = load_dataset("hotpot_qa", "distractor", split="validation")

    print(f"Train: {len(train_ds)}, Validation: {len(val_ds)}")

    # Combine and categorize by level
    by_level = {"easy": [], "medium": [], "hard": []}
    for item in train_ds:
        level = item["level"].lower()
        if level in by_level:
            by_level[level].append(item)
    for item in val_ds:
        level = item["level"].lower()
        if level in by_level:
            by_level[level].append(item)

    print(f"  easy: {len(by_level['easy'])}, medium: {len(by_level['medium'])}, hard: {len(by_level['hard'])}")

    # Random sample
    random.seed(seed)
    selected = []
    for level, count in [("easy", easy_count), ("medium", medium_count), ("hard", hard_count)]:
        pool = by_level[level]
        if len(pool) == 0:
            print(f"  WARNING: No '{level}' questions found, skipping...")
            continue
        sampled = random.sample(pool, min(count, len(pool)))
        selected.extend(sampled)

    print(f"Selected {len(selected)} questions")

    # Convert to QAExample format
    examples = []
    for idx, item in enumerate(selected):
        # HotpotQA context: list of [title, sentences_list]
        context_chunks = []
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            text = " ".join(sentences)
            context_chunks.append({"title": title, "text": text})

        # Map level to difficulty
        level = item["level"].lower()
        difficulty = level if level in ("easy", "medium", "hard") else "medium"

        examples.append({
            "qid": f"hq_{idx + 1:03d}",
            "difficulty": difficulty,
            "question": item["question"],
            "gold_answer": item["answer"],
            "context": context_chunks,
        })

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(examples)} questions to {out}")

    # Stats
    diff_counts = {}
    for ex in examples:
        diff_counts[ex["difficulty"]] = diff_counts.get(ex["difficulty"], 0) + 1
    print(f"Distribution: {diff_counts}")


if __name__ == "__main__":
    prepare_hotpot_dataset()
