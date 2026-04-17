#!/usr/bin/env python3
"""
Agent Trial Eval Basic Usage Example
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_trial_bench import (
    AgentTrialBench, 
    AgentMetadata, 
    TaskCategory, 
    EvaluationMethod,
    EvaluationConfig,
    LLMConfig,
    AgentConfig,
    DatasetConfig,
    EvaluatorConfig,
    RunnerConfig
)

async def basic_evaluation_example():
    """Basic evaluation example"""
    # Create configuration
    config = EvaluationConfig(
        llm_config=LLMConfig(
            model="doubao-seed-1-6",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=os.environ.get("ARK_API_KEY", ""),
            temperature=0.3,
            max_tokens=2000
        ),
        agent_config=AgentConfig(
            url="http://localhost:8002",
            capabilities=[TaskCategory.WEB_RETRIEVAL, TaskCategory.ONCHAIN_RETRIEVAL],
            timeout=30,
            close_endpoint="http://localhost:8002/close"
        ),
        dataset_config=DatasetConfig(),
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
    
    # Create SDK instance
    evaluator = AgentTrialBench(config)
    
    # Define Agent
    agent = AgentMetadata(
        url="http://localhost:8002",
        capabilities=[TaskCategory.WEB_RETRIEVAL, TaskCategory.ONCHAIN_RETRIEVAL],
        timeout=30,
        close_endpoint="http://localhost:8002/close"
    )
    
    # Evaluation question
    question = "What is the date (UTC) when the US SEC approved Bitcoin spot ETF?"
    
    # Execute evaluation
    result = await evaluator.evaluate_agent(
        question=question,
        agent_metadata=agent,
        category=TaskCategory.WEB_RETRIEVAL,
        context={"category": "regulatory_events"},
        expected_answer="2024/1/10"
    )
    
    print(f"Evaluation Score: {result.evaluation_score}")
    print(f"Agent Response: {result.agent_response}")
    print(f"Reasoning: {result.evaluation_reasoning}")
    
    return result

async def multiple_agents_evaluation_example():
    """Multiple agents evaluation example"""
    # Create configuration
    config = EvaluationConfig(
        llm_config=LLMConfig(
            model="doubao-seed-1-6",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=os.environ.get("ARK_API_KEY", ""),
            temperature=0.3,
            max_tokens=2000
        ),
        agent_config=AgentConfig(
            url="http://localhost:8002",
            capabilities=[TaskCategory.WEB_RETRIEVAL],
            timeout=30
        ),
        work_dir="output"
    )
    
    # Create evaluator
    evaluator = AgentTrialBench(config)
    
    # Define multiple Agents
    agents = [
        AgentMetadata(
            url="http://localhost:8002",
            capabilities=[TaskCategory.WEB_RETRIEVAL],
            timeout=30,
            close_endpoint="http://localhost:8002/close"
        ),
        AgentMetadata(
            url="http://localhost:8003",
            capabilities=[TaskCategory.WEB_RETRIEVAL, TaskCategory.ONCHAIN_RETRIEVAL],
            timeout=30,
            close_endpoint="http://localhost:8003/close"
        )
    ]
    
    # Evaluation question
    question = "Briefly describe the differences between Optimistic Rollup and ZK-Rollup in terms of exit time and security assumptions"
    
    # Execute multi-agent evaluation
    results = []
    for agent in agents:
        result = await evaluator.evaluate_agent(
            question=question,
            agent_metadata=agent,
            category=TaskCategory.WEB_RETRIEVAL,
            context={"category": "scaling_solutions"}
        )
        results.append(result)
    
    return results

async def export_results_example():
    """Export results example"""
    # Create configuration
    config = EvaluationConfig(
        llm_config=LLMConfig(
            model="doubao-seed-1-6",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=os.environ.get("ARK_API_KEY", ""),
            temperature=0.3,
            max_tokens=2000
        ),
        agent_config=AgentConfig(
            url="http://localhost:8002",
            capabilities=[TaskCategory.WEB_RETRIEVAL],
            timeout=30
        ),
        work_dir="custom_output"
    )
    
    # Create evaluator
    evaluator = AgentTrialBench(config)
    
    # Execute some evaluations
    agent = AgentMetadata(
        url="http://localhost:8002",
        capabilities=[TaskCategory.WEB_RETRIEVAL],
        timeout=30
    )
    
    questions = [
        ("What is the date (UTC) when the US SEC approved Bitcoin spot ETF?", "2024/1/10"),
        ("What is the topic0 (event signature hash) of ERC-20 Transfer event?", "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef")
    ]
    
    results = []
    for question, expected_answer in questions:
        result = await evaluator.evaluate_agent(
            question=question,
            agent_metadata=agent,
            category=TaskCategory.WEB_RETRIEVAL,
            expected_answer=expected_answer
        )
        results.append(result)
    
    # Export JSON format
    json_results = evaluator.export_results("json")
    
    # Export CSV format
    csv_results = evaluator.export_results("csv")
    
    return json_results, csv_results

async def main():
    """Main function"""
    try:
        # Basic evaluation
        await basic_evaluation_example()
        
        # Multi-agent evaluation
        await multiple_agents_evaluation_example()
        
        # Export results
        await export_results_example()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
