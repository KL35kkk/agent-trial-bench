#!/usr/bin/env python3
"""
Agent Trial Eval - Config-based Evaluation Example
Demonstrates the new configuration-driven architecture
"""

import asyncio
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_trial_bench import (
    AgentTrialBench,
    EvaluationConfig,
    AgentMetadata,
    TaskCategory,
    load_config
)


async def config_based_evaluation():
    """Example using configuration file"""
    
    # Load configuration from file
    config_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs",
        "example_config.json"
    )
    
    # Replace environment variables in config
    config = load_config(config_path)
    
    # Override API key from environment if needed
    if not config.llm_config.api_key or config.llm_config.api_key.startswith("${"):
        config.llm_config.api_key = os.environ.get("ARK_API_KEY", "")
    
    # Create evaluator with config
    evaluator = AgentTrialBench(config=config)
    
    # Create agent metadata from config
    agent = AgentMetadata(
        url=config.agent_config.url,
        capabilities=config.agent_config.capabilities,
        timeout=config.agent_config.timeout,
        close_endpoint=config.agent_config.close_endpoint
    )
    
    # Evaluate using dataset
    dataset_path = config.dataset_config.path
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            dataset_path
        )
    
    print(f"Starting evaluation with config from: {config_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Agent: {agent.url}")
    print("-" * 80)
    
    results = await evaluator.evaluate_agent_with_dataset(
        agent_metadata=agent,
        dataset_path=dataset_path,
        max_tasks=config.max_tasks
    )
    
    print(f"\nEvaluation completed. Total results: {len(results)}")
    
    # Export results
    summary = evaluator.export_results("json")
    
    print(f"\nTotal Score: {summary['overall']['average_score']:.3f}")
    print(f"Successful Tasks: {summary['overall']['successful_tasks']}")
    print(f"Failed Tasks: {summary['overall']['failed_tasks']}")
    
    # Export CSV as well
    evaluator.export_results("csv")
    print("\nResults exported to output/ directory")
    
    return results


async def programmatic_config():
    """Example creating config programmatically"""
    
    from agent_trial_bench import (
        LLMConfig,
        AgentConfig,
        DatasetConfig,
        EvaluatorConfig,
        RunnerConfig,
        EvaluationConfig
    )
    
    # Create configuration programmatically
    config = EvaluationConfig(
        llm_config=LLMConfig(
            model=os.environ.get("LLM_MODEL", "doubao-seed-1-6-251015"),
            base_url=os.environ.get("LLM_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
            api_key=os.environ.get("ARK_API_KEY", ""),
            temperature=0.3,
            max_tokens=2000
        ),
        agent_config=AgentConfig(
            url=os.environ.get("AGENT_URL", "http://localhost:8002"),
            capabilities=[TaskCategory.WEB_RETRIEVAL, TaskCategory.ONCHAIN_RETRIEVAL],
            timeout=30
        ),
        dataset_config=DatasetConfig(
            type="csv",
            path="data/benchmark.csv"
        ),
        evaluator_config=EvaluatorConfig(
            type="hybrid",
            rule_based_weight=0.3,
            llm_based_weight=0.7
        ),
        runner_config=RunnerConfig(
            type="local",
            max_workers=4
        ),
        work_dir="output"
    )
    
    # Create evaluator
    evaluator = AgentTrialBench(config=config)
    
    # Use evaluator...
    agent = AgentMetadata(
        url=config.agent_config.url,
        capabilities=config.agent_config.capabilities,
        timeout=config.agent_config.timeout
    )
    
    # Single evaluation
    result = await evaluator.evaluate_agent(
        question="What is the date (UTC) when the US SEC approved Bitcoin spot ETF?",
        agent_metadata=agent,
        category=TaskCategory.WEB_RETRIEVAL,
        expected_answer="2024/1/10"
    )
    
    print(f"Evaluation Score: {result.evaluation_score}")
    print(f"Reasoning: {result.evaluation_reasoning}")
    
    return result


async def main():
    """Main function"""
    try:
        # Try config-based evaluation
        await config_based_evaluation()
    except FileNotFoundError:
        print("Config file not found, trying programmatic config...")
        await programmatic_config()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

