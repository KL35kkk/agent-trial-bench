#!/usr/bin/env python3
"""
Agent Trial Eval Batch Evaluation Example
"""

import asyncio
import csv
import json
import sys
import os
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent_trial_bench import (
    AgentTrialBench, 
    AgentMetadata, 
    EvaluationResult, 
    TaskCategory,
    EvaluationConfig,
    LLMConfig,
    AgentConfig,
    DatasetConfig,
    RunnerConfig
)

class BatchEvaluator:
    """Batch evaluator"""
    
    def __init__(self):
        # Create configuration
        config = EvaluationConfig(
            llm_config=LLMConfig(
                model="doubao-seed-1-6-251015",
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key=os.environ.get("ARK_API_KEY", ""),
                temperature=0.3,
                max_tokens=2000
            ),
            agent_config=AgentConfig(
                url="http://localhost:8002",
                capabilities=[TaskCategory.ONCHAIN_RETRIEVAL],
                timeout=30
            ),
            dataset_config=DatasetConfig(),
            runner_config=RunnerConfig(
                type="local",
                max_workers=4
            ),
            work_dir="output"
        )
        self.evaluator = AgentTrialBench(config)
        self.results: List[EvaluationResult] = []
    
    async def load_benchmark_data(self, csv_file: str) -> List[Dict[str, Any]]:
        """Load benchmark data"""
        tasks = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tasks.append({
                    'id': row['id'],
                    'question': row['question'],
                    'answer': row['answer'],
                    'category': row.get('category', ''),
                    'task_type': row.get('task_type', 'general')
                })
        return tasks
    
    async def evaluate_benchmark(
        self,
        benchmark_tasks: List[Dict[str, Any]],
        agent_metadata: AgentMetadata,
        max_tasks: int = None
    ) -> List[EvaluationResult]:
        """Evaluate benchmark tasks"""
        
        if max_tasks:
            benchmark_tasks = benchmark_tasks[:max_tasks]
        
        results = []
        
        for i, task in enumerate(benchmark_tasks):
            try:
                result = await self.evaluator.evaluate_agent(
                    question=task['question'],
                    agent_metadata=agent_metadata,
                    task_type=task['task_type'],
                    context={
                        'expected_answer': task['answer'],
                        'category': task['category'],
                        'benchmark_id': task['id']
                    }
                )
                
                results.append(result)
                
            except Exception as e:
                # Create failure result
                failed_result = EvaluationResult(
                    task_id=f"failed_{task['id']}",
                    question=task['question'],
                    agent_response="",
                    evaluation_score=0.0,
                    evaluation_reasoning="",
                    confidence=0.0,
                    processing_time=0.0,
                    tools_used=[],
                    metadata={},
                    status="failed",
                    error=str(e)
                )
                results.append(failed_result)
        
        self.results = results
        return results
    
    def export_results(self, output_dir: str = "output"):
        """Export evaluation results"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export JSON
        json_file = f"{output_dir}/evaluation_results.json"
        json_data = self.evaluator.export_results("json")
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        # Export CSV
        csv_file = f"{output_dir}/evaluation_results.csv"
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'task_id', 'question', 'agent_response', 'evaluation_score',
                'evaluation_reasoning', 'confidence', 'processing_time',
                'tools_used', 'status', 'error'
            ])
            
            for result in self.results:
                writer.writerow([
                    result.task_id,
                    result.question,
                    result.agent_response,
                    result.evaluation_score,
                    result.evaluation_reasoning,
                    result.confidence,
                    result.processing_time,
                    ','.join(result.tools_used),
                    result.status.value,
                    result.error or ''
                ])
        
        # Generate summary report
        self._generate_summary_report(output_dir)
    
    def _generate_summary_report(self, output_dir: str):
        """Generate summary report"""
        successful_results = [r for r in self.results if r.status.value == "completed"]
        failed_results = [r for r in self.results if r.status.value == "failed"]
        
        if successful_results:
            avg_score = sum(r.evaluation_score for r in successful_results) / len(successful_results)
            avg_confidence = sum(r.confidence for r in successful_results) / len(successful_results)
            avg_time = sum(r.processing_time for r in successful_results) / len(successful_results)
        else:
            avg_score = avg_confidence = avg_time = 0.0
        
        report = {
            "summary": {
                "total_tasks": len(self.results),
                "successful_tasks": len(successful_results),
                "failed_tasks": len(failed_results),
                "success_rate": len(successful_results) / len(self.results) if self.results else 0,
                "average_score": avg_score,
                "average_confidence": avg_confidence,
                "average_processing_time": avg_time
            },
            "score_distribution": {
                "excellent": len([r for r in successful_results if r.evaluation_score >= 0.8]),
                "good": len([r for r in successful_results if 0.6 <= r.evaluation_score < 0.8]),
                "fair": len([r for r in successful_results if 0.4 <= r.evaluation_score < 0.6]),
                "poor": len([r for r in successful_results if r.evaluation_score < 0.4])
            }
        }
        
        report_file = f"{output_dir}/evaluation_summary.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

async def main():
    """Main function"""
    # Create batch evaluator
    evaluator = BatchEvaluator()
    
    try:
        # Load benchmark data
        benchmark_tasks = await evaluator.load_benchmark_data("data/benchmark.csv")
        
        # Define Agent
        agent = AgentMetadata(
            url="http://localhost:8002",  # Handcrafted Agent
            capabilities=[TaskCategory.ONCHAIN_RETRIEVAL],
            timeout=30,
            close_endpoint="http://localhost:8002/close"
        )
        
        # Execute batch evaluation (limit to first 5 tasks)
        results = await evaluator.evaluate_benchmark(
            benchmark_tasks=benchmark_tasks,
            agent_metadata=agent,
            max_tasks=5
        )
        
        # Export results
        evaluator.export_results("output")
        
    except Exception as e:
        # Handle errors silently
        pass

if __name__ == "__main__":
    asyncio.run(main())
