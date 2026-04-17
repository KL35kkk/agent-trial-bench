#!/usr/bin/env python3
"""
Agent Trial Eval Enhanced Batch Evaluation Example
Supports intelligent evaluation method selection based on task categories
"""

import asyncio
import csv
import json
import sys
import os
from typing import List, Dict, Any
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent_trial_bench import (
    AgentTrialBench, 
    AgentMetadata, 
    EvaluationResult,
    EvaluationConfig,
    LLMConfig,
    AgentConfig,
    DatasetConfig,
    EvaluatorConfig,
    RunnerConfig,
    TaskCategory
)

class EnhancedBatchEvaluator:
    """Enhanced batch evaluator"""
    
    def __init__(self, config: EvaluationConfig = None):
        if config is None:
            # Create default config
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
                dataset_config=DatasetConfig(),
                work_dir="output"
            )
        self.evaluator = AgentTrialBench(config)
        self.results: List[EvaluationResult] = []
        self.category_stats = defaultdict(lambda: {"total": 0, "successful": 0, "failed": 0, "scores": []})
    
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
                    'category': row.get('category', 'general'),
                    'task_type': row.get('task_type', 'general'),
                    'evaluation_method': row.get('evaluation_method', 'hybrid')
                })
        return tasks
    
    def _select_evaluation_method(self, task: Dict[str, Any]) -> str:
        """Intelligently select evaluation method based on task category"""
        category = task['category'].lower()
        task_type = task['task_type'].lower()
        
        # Web retrieval tasks: mainly use rule-based evaluation
        if category == "web_retrieval":
            if task_type in ["single_doc_fact_qa", "event_timeline", "top_k_aggregation"]:
                return "rule_based"
            elif task_type in ["abstractive_with_evidence", "disambiguation"]:
                return "llm_based"
            else:
                return "hybrid"
        
        # Web + On-chain retrieval: use hybrid evaluation
        elif category == "web_onchain_retrieval":
            if task_type in ["claim_verification", "reconciliation"]:
                return "hybrid"
            elif task_type == "cross_source_explanation":
                return "llm_based"
            else:
                return "hybrid"
        
        # On-chain retrieval: select based on task type
        elif category == "onchain_retrieval":
            if task_type in ["event_timeline", "parameter_permission_change"]:
                return "rule_based"
            elif task_type in ["technical_knowledge"]:
                return "llm_based"
            else:
                return "hybrid"
        
        # Default to hybrid evaluation
        return "hybrid"
    
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
            # Intelligently select evaluation method
            evaluation_method = self._select_evaluation_method(task)
            
            try:
                result = await self.evaluator.evaluate_agent(
                    question=task['question'],
                    agent_metadata=agent_metadata,
                    task_type=task['task_type'],
                    context={
                        'expected_answer': task['answer'],
                        'category': task['category'],
                        'benchmark_id': task['id']
                    },
                    category=task['category'],
                    evaluation_method=evaluation_method,
                    expected_answer=task['answer']
                )
                
                results.append(result)
                
                # Update statistics
                self.category_stats[task['category']]['total'] += 1
                if result.status.value == "completed":
                    self.category_stats[task['category']]['successful'] += 1
                    self.category_stats[task['category']]['scores'].append(result.evaluation_score)
                else:
                    self.category_stats[task['category']]['failed'] += 1
                
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
                self.category_stats[task['category']]['total'] += 1
                self.category_stats[task['category']]['failed'] += 1
        
        self.results = results
        return results
    
    def export_results(self, output_dir: str = "output"):
        """Export evaluation results"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export JSON
        json_file = f"{output_dir}/enhanced_evaluation_results.json"
        json_data = self.evaluator.export_results("json")
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        # Export CSV
        csv_file = f"{output_dir}/enhanced_evaluation_results.csv"
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'task_id', 'question', 'category', 'task_type', 'evaluation_method',
                'agent_response', 'evaluation_score', 'evaluation_reasoning', 
                'confidence', 'processing_time', 'tools_used', 'status', 'error'
            ])
            
            for result in self.results:
                # Get additional information from tasks
                task_info = {}
                for task in self.evaluator.tasks.values():
                    if task.task_id == result.task_id:
                        task_info = {
                            'category': task.category,
                            'task_type': task.task_type,
                            'evaluation_method': task.evaluation_method
                        }
                        break
                
                writer.writerow([
                    result.task_id,
                    result.question,
                    task_info.get('category', ''),
                    task_info.get('task_type', ''),
                    task_info.get('evaluation_method', ''),
                    result.agent_response,
                    result.evaluation_score,
                    result.evaluation_reasoning,
                    result.confidence,
                    result.processing_time,
                    ','.join(result.tools_used),
                    result.status.value,
                    result.error or ''
                ])
        
        # Generate enhanced summary report
        self._generate_enhanced_summary_report(output_dir)
    
    def _generate_enhanced_summary_report(self, output_dir: str):
        """Generate enhanced summary report"""
        successful_results = [r for r in self.results if r.status.value == "completed"]
        failed_results = [r for r in self.results if r.status.value == "failed"]
        
        # Overall statistics
        total_tasks = len(self.results)
        successful_tasks = len(successful_results)
        failed_tasks = len(failed_results)
        
        if successful_results:
            avg_score = sum(r.evaluation_score for r in successful_results) / len(successful_results)
            avg_confidence = sum(r.confidence for r in successful_results) / len(successful_results)
            avg_time = sum(r.processing_time for r in successful_results) / len(successful_results)
        else:
            avg_score = avg_confidence = avg_time = 0.0
        
        # Statistics by category
        category_analysis = {}
        for category, stats in self.category_stats.items():
            if stats['total'] > 0:
                success_rate = stats['successful'] / stats['total']
                avg_category_score = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0.0
                category_analysis[category] = {
                    "total_tasks": stats['total'],
                    "successful_tasks": stats['successful'],
                    "failed_tasks": stats['failed'],
                    "success_rate": success_rate,
                    "average_score": avg_category_score
                }
        
        report = {
            "summary": {
                "total_tasks": total_tasks,
                "successful_tasks": successful_tasks,
                "failed_tasks": failed_tasks,
                "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
                "average_score": avg_score,
                "average_confidence": avg_confidence,
                "average_processing_time": avg_time
            },
            "category_analysis": category_analysis,
            "score_distribution": {
                "excellent": len([r for r in successful_results if r.evaluation_score >= 0.8]),
                "good": len([r for r in successful_results if 0.6 <= r.evaluation_score < 0.8]),
                "fair": len([r for r in successful_results if 0.4 <= r.evaluation_score < 0.6]),
                "poor": len([r for r in successful_results if r.evaluation_score < 0.4])
            }
        }
        
        report_file = f"{output_dir}/enhanced_evaluation_summary.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

async def main():
    """Main function"""
    # Create enhanced batch evaluator
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
            timeout=30
        ),
        dataset_config=DatasetConfig(),
        evaluator_config=EvaluatorConfig(
            type="hybrid",
            use_llm_evaluation=True,
            llm_evaluation_threshold=0.5,
            rule_based_weight=0.3,
            llm_based_weight=0.7
        ),
        runner_config=RunnerConfig(
            type="local",
            max_workers=4
        ),
        work_dir="output"
    )
    
    evaluator = EnhancedBatchEvaluator(config)
    
    try:
        # Load benchmark data
        benchmark_tasks = await evaluator.load_benchmark_data("data/benchmark.csv")
        
        # Define Agent
        agent = AgentMetadata(
            url="http://localhost:8002",  # Handcrafted Agent
            capabilities=["web3_analysis", "blockchain_exploration"],
            timeout=30,
            close_endpoint="http://localhost:8002/close"
        )
        
        # Execute enhanced batch evaluation (limit to first 6 tasks)
        results = await evaluator.evaluate_benchmark(
            benchmark_tasks=benchmark_tasks,
            agent_metadata=agent,
            max_tasks=6
        )
        
        # Export results
        evaluator.export_results("output")
        
    except Exception as e:
        # Handle errors silently
        pass

if __name__ == "__main__":
    asyncio.run(main())