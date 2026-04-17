"""
Example script demonstrating the mock evaluation pipeline with automatic
accuracy analysis output.
"""

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_trial_bench import (  # noqa: E402  (import after sys.path update)
    load_config,
    AgentTrialBench,
    AgentMetadata,
    TaskCategory,
)


async def main():
    base_dir = Path(__file__).resolve().parent
    dataset_path = base_dir / "sample_dataset.csv"
    ground_truth_path = base_dir / "sample_ground_truth.json"
    work_dir = base_dir / "sample_output"

    config = load_config(
        llm_config={
            "model": "stub-model",
            "base_url": "https://example.com",
            "api_key": "mock-key",
            "temperature": 0.1,
            "max_tokens": 512,
        },
        agent_config={
            "url": "mock://local-agent",
            "capabilities": ["web_retrieval"],
            "timeout": 10,
        },
        dataset_config={
            "path": str(dataset_path),
            "question_id_field": "id",
            "ground_truth_path": str(ground_truth_path),
            "mock_response_field": "mock_response",
        },
        evaluator_config={
            "type": "hybrid",
            "rule_based_weight": 0.6,
            "llm_based_weight": 0.4,
            "use_llm_evaluation": False,
        },
        work_dir=str(work_dir),
    )

    evaluator = AgentTrialBench(config)
    agent = AgentMetadata(
        url="mock://local-agent",
        capabilities=[TaskCategory.WEB_RETRIEVAL],
        timeout=10,
    )

    await evaluator.evaluate_agent_with_dataset(
        agent_metadata=agent,
        dataset_path=str(dataset_path),
        max_tasks=3,
    )

    summary = evaluator.export_results("json")

    print("=== Overall Summary ===")
    print(json.dumps(summary["overall"], indent=2, ensure_ascii=False))
    if "accuracy_analysis" in summary:
        print("\n=== Accuracy Analysis ===")
        print(json.dumps(summary["accuracy_analysis"], indent=2, ensure_ascii=False))

    print(f"\nDetailed JSON saved at: {work_dir / 'evaluation_results.json'}")
    print(f"Accuracy analysis saved at: {work_dir / 'accuracy_analysis.json'}")


if __name__ == "__main__":
    asyncio.run(main())
