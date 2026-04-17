#!/usr/bin/env python3
"""
Agent Trial Bench – Domain-Agnostic Benchmark for AI Agents.

Modular, process-aware agent evaluation framework.  Web3 is supported as a
first-class domain plugin; the core evaluator is fully domain-agnostic.
"""

import asyncio
import csv
import time
import os
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging

# Import enums and dataclasses
from .enums import TaskCategory, EvaluationMethod, EvaluationStatus
from .dataclasses import AgentMetadata, EvaluationTask, EvaluationResult

# Import evaluation capabilities
from .evaluation.base_evaluator import BaseEvaluator
from .evaluation.llm_evaluator import LLMEvaluator
from .evaluation.hybrid_evaluator import HybridEvaluator

# Import new modular components
from .config import (
    EvaluationConfig, LLMConfig, AgentConfig, DatasetConfig,
    EvaluatorConfig, RunnerConfig, StorageConfig,
    BusinessConfig, InfrastructureConfig
)
from .evaluation_engine import EvaluationEngine
from .evaluation.accuracy_analysis import EvaluationAccuracyAnalyzer
from .runners.local_runner import LocalRunner
from .runners.agent_runner import AgentRunner
from .summarizers.default import DefaultSummarizer
from .storage import ResultStorage
from .task_manager import TaskManager
from .calibration import CalibrationManager, CalibrationReport

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# AgentMetadata, EvaluationTask, EvaluationResult are now in .dataclasses

class AgentTrialBench:
    """Agent Trial Bench Evaluator with modular architecture.
    
    This version separates concerns:
    - EvaluationEngine: Handles evaluation logic
    - Runner: Handles task execution
    - Summarizer: Handles result aggregation and reporting
    """
    
    def __init__(self, 
                 config: EvaluationConfig):
        """
        Initialize Agent Trial Bench Evaluator.
        
        Args:
            config: EvaluationConfig instance
        """
        if config is None:
            raise ValueError("config must be provided")
        
        self.config = config
        self.output_path = config.work_dir
        
        # Initialize components
        self.agent_runner = AgentRunner(config={'timeout': 30})
        self.evaluation_engine = EvaluationEngine(
            llm_config=config.llm_config.to_dict(),
            evaluator_config=config.evaluator_config.to_dict() if config.evaluator_config else None,
            agent_runner=self.agent_runner
        )
        
        runner_config = config.runner_config.to_dict() if config.runner_config else {}
        self.runner = LocalRunner(config=runner_config)
        self.summarizer = DefaultSummarizer(config={})
        
        # Initialize storage and task manager
        # Create default storage config if not provided
        if config.storage_config is None:
            config.infrastructure_config.storage_config = StorageConfig()
        storage_config_dict = config.storage_config.to_dict()
        self.storage = ResultStorage(work_dir=config.work_dir, storage_config=storage_config_dict)
        self.task_manager = TaskManager(storage=self.storage)
        
        # Task storage (in-memory)
        self.results: List[Dict[str, Any]] = []
        self._ground_truth: Dict[str, float] = self._load_ground_truth(config.dataset_config)
        self.calibration_config = None
        self.calibration_targets: Dict[str, float] = {}
        self._calibration_manager: Optional[CalibrationManager] = None
        self._calibration_applied = False
        if config.evaluator_config and config.evaluator_config.calibration_config:
            self.calibration_config = config.evaluator_config.calibration_config
            self.calibration_targets = self._load_calibration_targets(self.calibration_config)
        
        # Handle reuse_results if specified
        if config.reuse_results:
            self._load_existing_results(config.reuse_results)
    
    def _load_existing_results(self, reuse_mode: str):
        """Load existing results for reuse"""
        if reuse_mode == "latest":
            # Load latest results
            latest_task_id = self.storage.get_latest_results()
            if latest_task_id:
                result = self.storage.load_result(latest_task_id)
                if result:
                    self.results.append(result)
                    logger.info(f"Loaded latest result: {latest_task_id}")
        else:
            # Load specific timestamp or task ID
            result = self.storage.load_result(reuse_mode)
            if result:
                self.results.append(result)
                logger.info(f"Loaded result: {reuse_mode}")
    
    async def evaluate_agent(self,
        question: str,
        agent_metadata: AgentMetadata,
        context: Optional[Dict[str, Any]] = None,
        category: TaskCategory = TaskCategory.WEB_RETRIEVAL,
        evaluation_method: Optional[EvaluationMethod] = None,
                            expected_answer: Optional[str] = None) -> EvaluationResult:
        """
        Evaluate single Agent.
        
        Args:
            question: Evaluation question
            agent_metadata: Agent metadata
            context: Additional context
            category: Task category
            evaluation_method: Evaluation method (auto-selected if None)
            expected_answer: Expected answer (optional)
            
        Returns:
            EvaluationResult instance
        """
        context = dict(context or {})
        if agent_metadata.url.startswith("mock://") and expected_answer:
            context.setdefault("_expected_answer", expected_answer)
        if "question_id" not in context and context.get("dataset_id"):
            context["question_id"] = context["dataset_id"]
        ground_truth_from_context = context.get("ground_truth_score")
        
        result_dict = await self.evaluation_engine.evaluate_task(
            question=question,
            agent_url=agent_metadata.url,
            agent_timeout=agent_metadata.timeout,
            expected_answer=expected_answer,
            category=category,
            evaluation_method=evaluation_method,
            context=context,
            close_endpoint=agent_metadata.close_endpoint
        )
        
        # Convert to EvaluationResult
        task_id = f"task_{int(time.time())}_{len(self.results)}"
        result = EvaluationResult(
                task_id=task_id,
            question=result_dict["question"],
            agent_response=result_dict["agent_response"],
            evaluation_score=result_dict["evaluation_score"],
            evaluation_reasoning=result_dict["evaluation_reasoning"],
            confidence=result_dict["confidence"],
            processing_time=result_dict["processing_time"],
            tools_used=result_dict.get("tools_used", []),
            metadata=result_dict.get("metadata", {}),
            status=EvaluationStatus(result_dict["status"]),
            error=result_dict.get("error")
        )
        result_dict["task_id"] = task_id
        
        # Augment result metadata for analytics & persistence
        ground_truth_score = ground_truth_from_context
        if ground_truth_score is None:
            lookup_key = result_dict.get("question_id") or question
            ground_truth_score = self._ground_truth.get(lookup_key)
        if ground_truth_score is not None:
            result_dict["ground_truth_score"] = ground_truth_score
        result_dict["expected_answer"] = expected_answer
        result_dict["question_id"] = result_dict.get("question_id") or context.get("question_id") or question
        result_dict["dataset_id"] = context.get("dataset_id")
        
        # Store result (in-memory)
        self.results.append(result_dict)
        self._record_history_entry(result_dict)
        
        # Persist result if enabled
        storage_config = self.config.infrastructure_config.storage_config
        if storage_config and storage_config.enable_persistence:
            self.task_manager.save_result(task_id, result)
            
            # Auto-save if enabled
            if storage_config.auto_save and len(self.results) % storage_config.save_interval == 0:
                self._auto_save_results()
        
        return result
    
    async def evaluate_agent_with_dataset(self,
        agent_metadata: AgentMetadata,
        dataset_path: str,
        max_tasks: Optional[int] = None,
        n_replicas: int = 1,
        k: Optional[int] = None,
        pass_k_threshold: float = 0.7) -> List[EvaluationResult]:
        """
        Evaluate Agent using dataset.
        
        Args:
            agent_metadata: Agent metadata
            dataset_path: Path to dataset CSV file
            max_tasks: Maximum number of tasks to evaluate (None = all)
            n_replicas: Number of replicas per question (for consistency/Pass@k)
            k: Optional Pass@k parameter (stored in context for downstream analyzers)
            pass_k_threshold: Score threshold for success when computing Pass@k
            
        Returns:
            List of EvaluationResult instances
        """
        # Load dataset
        tasks = []
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Accept any known TaskCategory value; fall back to GENERAL
                    # for unrecognised strings so generic datasets just work.
                    category_map = {member.value: member for member in TaskCategory}
                    
                    method_map = {
                        "rule_based": EvaluationMethod.RULE_BASED,
                        "llm_based": EvaluationMethod.LLM_BASED,
                        "hybrid": EvaluationMethod.HYBRID,
                        "cascade": EvaluationMethod.CASCADE,
                    }
                    
                    question_id = row.get(self.config.dataset_config.question_id_field, row.get("id", str(len(tasks) + 1)))
                    ground_truth_val = row.get("ground_truth_score")
                    ground_truth_score = None
                    if ground_truth_val not in (None, ""):
                        try:
                            ground_truth_score = float(ground_truth_val)
                            self._ground_truth[str(question_id)] = ground_truth_score
                        except ValueError:
                            ground_truth_score = None
                    mock_response = None
                    mock_field = self.config.dataset_config.mock_response_field
                    if mock_field:
                        mock_response = row.get(mock_field)
                    if not mock_response:
                        mock_response = row.get("mock_response")
                    if not mock_response:
                        mock_response = row.get("answer", "")
                    
                    task_data = {
                        "question": row["question"],
                        "expected_answer": row.get("answer", ""),
                        "category": category_map.get(row.get("category", "general"), TaskCategory.GENERAL),
                        "evaluation_method": method_map.get(row.get("evaluation_method", "hybrid"), EvaluationMethod.HYBRID),
                        "question_id": question_id,
                        "context": {
                            "dataset_id": question_id,
                            "question_id": question_id,
                            "category": row.get("category", "web_retrieval"),
                            "task_type": row.get("task_type"),
                            "ground_truth_score": ground_truth_score,
                            "mock_response": mock_response,
                        }
                    }
                    tasks.append(task_data)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
        
        if max_tasks:
            tasks = tasks[:max_tasks]

        # Duplicate tasks for multiple replicas if requested
        replicas = max(1, int(n_replicas or 1))
        if replicas > 1:
            replicated_tasks = []
            for replica_idx in range(replicas):
                for task in tasks:
                    task_copy = json.loads(json.dumps(task))
                    task_copy["context"]["replica_index"] = replica_idx
                    replicated_tasks.append(task_copy)
            tasks = replicated_tasks
        
        # Evaluate all tasks
        results = []
        for i, task_data in enumerate(tasks):
            try:
                logger.info(f"Evaluating task {i+1}/{len(tasks)}: {task_data['question'][:50]}...")
                
                result = await self.evaluate_agent(
                    question=task_data["question"],
                    agent_metadata=agent_metadata,
                    context=task_data["context"],
                    category=task_data["category"],
                    evaluation_method=task_data["evaluation_method"],
                    expected_answer=task_data["expected_answer"]
                )
                # annotate pass@k related metadata for downstream analysis
                result_dict = self.results[-1]
                result_dict.setdefault("pass_k_threshold", pass_k_threshold)
                result_dict.setdefault("k", k)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to evaluate task {i+1}: {e}")
                # Create failed result
                failed_result = EvaluationResult(
                    task_id=f"failed_{i+1}",
                    question=task_data["question"],
                    agent_response="",
                    evaluation_score=0.0,
                    evaluation_reasoning="",
                    confidence=0.0,
                    processing_time=0.0,
                    tools_used=[],
                    metadata={},
                    status=EvaluationStatus.FAILED,
                    error=str(e)
                )
                results.append(failed_result)
                self.results.append({
                    "status": "failed",
                    "question": task_data["question"],
                    "error": str(e)
                })
        
        return results
    
    def export_results(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """
        Export evaluation results.
        
        Args:
            format: Export format ('json' or 'csv')
            
        Returns:
            Exported data (dict for JSON, string for CSV)
        """
        calibration_report = self._apply_score_calibration()
        # Convert results to summary format
        summary = self.summarizer.summarize(self.results)
        if calibration_report:
            summary["calibration"] = calibration_report
        accuracy_analysis = self._run_accuracy_analysis()
        if accuracy_analysis:
            summary["accuracy_analysis"] = accuracy_analysis
        
        if format == "json":
            # Save summary
            summary_path = self.summarizer.export(summary, self.output_path, "json")
            
            # Also save detailed results
            details_path = os.path.join(self.output_path, "evaluation_results.json")
            details_payload = {
                    "total_score": summary["overall"]["average_score"],
                    "total_tasks": summary["overall"]["total_tasks"],
                    "successful_tasks": summary["overall"]["successful_tasks"],
                    "failed_tasks": summary["overall"]["failed_tasks"],
                    "results": self.results
                }
            if accuracy_analysis:
                details_payload["accuracy_analysis"] = accuracy_analysis
            if calibration_report:
                details_payload["calibration"] = calibration_report
            with open(details_path, 'w', encoding='utf-8') as f:
                json.dump(details_payload, f, indent=2, ensure_ascii=False)
            
            return summary
        
        elif format == "csv":
            # Export summary as CSV
            summary_path = self.summarizer.export(summary, self.output_path, "csv")
            
            # Also export detailed results
            csv_path = os.path.join(self.output_path, "evaluation_results.csv")
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'task_id', 'question', 'category', 'evaluation_method',
                    'agent_response', 'evaluation_score', 'evaluation_reasoning',
                    'confidence', 'processing_time', 'tools_used', 'status', 'error'
                ])
                for result in self.results:
                    writer.writerow([
                        result.get('task_id', ''),
                        result.get('question', ''),
                        result.get('category', ''),
                        result.get('evaluation_method', ''),
                        result.get('agent_response', ''),
                        result.get('evaluation_score', 0.0),
                        result.get('evaluation_reasoning', ''),
                        result.get('confidence', 0.0),
                        result.get('processing_time', 0.0),
                        ','.join(result.get('tools_used', [])),
                        result.get('status', ''),
                        result.get('error', '')
                    ])
            
            return summary_path
        
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _auto_save_results(self):
        """Auto-save results periodically"""
        # Results are already saved individually, this is for batch operations
        pass
    
    def _run_accuracy_analysis(self) -> Optional[Dict[str, Any]]:
        """Run accuracy analysis automatically after evaluations complete."""
        if not self.results:
            return None
        analyzer = EvaluationAccuracyAnalyzer()
        ground_truth = self._build_ground_truth_lookup()
        analysis = analyzer.comprehensive_analysis(
            self.results,
            ground_truth if ground_truth else None
        )
        alerts = (analysis.get("variance_alerts") or {}).get("alerts") or []
        if alerts:
            logger.warning("Variance alerts detected for %d questions", len(alerts))
        try:
            os.makedirs(self.output_path, exist_ok=True)
            analysis_path = os.path.join(self.output_path, "accuracy_analysis.json")
            with open(analysis_path, "w", encoding="utf-8") as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning(f"Failed to persist accuracy analysis: {exc}")
        return analysis
    
    def _record_history_entry(self, result: Dict[str, Any]):
        """Append evaluation metadata to persistent history for accuracy analysis."""
        if not hasattr(self, "storage") or not self.storage:
            return
        history_entry = {
            "task_id": result.get("task_id"),
            "question_id": result.get("question_id") or result.get("question"),
            "question": result.get("question"),
            "score": result.get("evaluation_score"),
            "confidence": result.get("confidence"),
            "status": result.get("status"),
            "category": result.get("category"),
            "evaluation_method": result.get("evaluation_method"),
            "evaluated_at": result.get("evaluated_at"),
            "ground_truth_score": result.get("ground_truth_score"),
        }
        self.storage.append_history_entry(history_entry)
    
    def _load_ground_truth(self, dataset_config: DatasetConfig) -> Dict[str, float]:
        """Load ground truth metadata from dataset or external file."""
        ground_truth: Dict[str, float] = {}
        if not dataset_config:
            return ground_truth
        
        dataset_path = dataset_config.path
        if dataset_path and os.path.exists(dataset_path):
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        question_id = row.get(dataset_config.question_id_field) or row.get("id") or row.get("question")
                        question_text = row.get("question")
                        if not question_id:
                            continue
                        score_val = row.get("ground_truth_score")
                        if score_val is None:
                            continue
                        try:
                            numeric_score = float(score_val)
                            ground_truth[str(question_id)] = numeric_score
                            if question_text:
                                ground_truth[str(question_text)] = numeric_score
                        except ValueError:
                            continue
            except Exception as exc:
                logger.debug(f"Skip loading inline ground truth: {exc}")
        
        # Prefer explicit ground truth path; otherwise fall back to a default sibling file
        candidate_path = dataset_config.ground_truth_path
        if not candidate_path and dataset_path:
            dataset_dir = os.path.dirname(os.path.abspath(dataset_path))
            default_candidate = os.path.join(dataset_dir, "ground_truth_sample.json")
            if os.path.exists(default_candidate):
                candidate_path = default_candidate
        if candidate_path:
            resolved_path = self._resolve_ground_truth_path(candidate_path, dataset_path)
            if resolved_path and os.path.exists(resolved_path):
                try:
                    with open(resolved_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        for key, value in data.items():
                            try:
                                ground_truth[str(key)] = float(value)
                            except (TypeError, ValueError):
                                continue
                    elif isinstance(data, list):
                        for item in data:
                            if not isinstance(item, dict):
                                continue
                            qid = item.get("question_id") or item.get("id") or item.get("question")
                            value = item.get("ground_truth_score") or item.get("score")
                            question_text = item.get("question")
                            if qid is None or value is None:
                                continue
                            try:
                                numeric_score = float(value)
                                ground_truth[str(qid)] = numeric_score
                                if question_text:
                                    ground_truth[str(question_text)] = numeric_score
                            except (TypeError, ValueError):
                                continue
                except Exception as exc:
                    logger.warning(f"Failed to load ground truth file {resolved_path}: {exc}")
        return ground_truth
    
    def _load_calibration_targets(self, calibration_config) -> Dict[str, float]:
        """Load calibration targets from an external dataset."""
        targets: Dict[str, float] = {}
        if not calibration_config or not calibration_config.path:
            return targets
        path = calibration_config.path
        try:
            if path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for key, value in data.items():
                        try:
                            targets[str(key)] = float(value)
                        except (TypeError, ValueError):
                            continue
                elif isinstance(data, list):
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        qid = item.get(calibration_config.question_field) or item.get("question_id")
                        value = item.get(calibration_config.score_field) or item.get("score")
                        if qid is None or value is None:
                            continue
                        try:
                            targets[str(qid)] = float(value)
                        except (TypeError, ValueError):
                            continue
            elif path.endswith(".csv"):
                with open(path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        qid = row.get(calibration_config.question_field) or row.get("question_id")
                        value = row.get(calibration_config.score_field) or row.get("score")
                        if qid is None or value is None:
                            continue
                        try:
                            targets[str(qid)] = float(value)
                        except ValueError:
                            continue
        except Exception as exc:
            logger.warning(f"Failed to load calibration dataset {path}: {exc}")
        return targets
    
    def _resolve_ground_truth_path(self, path: str, dataset_path: Optional[str]) -> Optional[str]:
        """Resolve ground truth path relative to dataset if needed."""
        if os.path.isabs(path):
            return path
        candidates = []
        if dataset_path:
            dataset_dir = os.path.dirname(os.path.abspath(dataset_path))
            candidates.append(os.path.join(dataset_dir, path))
        candidates.append(os.path.abspath(path))
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[-1]
    
    def _build_ground_truth_lookup(self) -> Dict[str, float]:
        """Combine preloaded ground truth with result-level overrides."""
        lookup = dict(self._ground_truth)
        for result in self.results:
            qid = result.get("question_id") or result.get("question")
            value = result.get("ground_truth_score")
            if qid and value is not None:
                lookup[str(qid)] = float(value)
                question_text = result.get("question")
                if question_text:
                    lookup[str(question_text)] = float(value)
        return lookup
    
    def _apply_score_calibration(self) -> Optional[Dict[str, Any]]:
        """Apply calibration to evaluation scores using labeled pairs."""
        if not self.calibration_config or self._calibration_applied:
            return None
        calibration_pairs: List[Tuple[float, float]] = []
        for result in self.results:
            question_id = result.get("question_id")
            target = None
            if question_id and self.calibration_targets:
                target = self.calibration_targets.get(str(question_id))
            if target is None:
                target = result.get("ground_truth_score")
            if target is None:
                continue
            raw_score = float(result.get("evaluation_score", 0.0))
            calibration_pairs.append((raw_score, float(target)))
        if len(calibration_pairs) < max(2, self.calibration_config.min_pairs):
            return None
        self._calibration_manager = CalibrationManager(self.calibration_config.method)
        self._calibration_manager.fit(calibration_pairs)
        if not self._calibration_manager.fitted:
            return None
        for result in self.results:
            raw_score = float(result.get("evaluation_score", 0.0))
            calibrated = self._calibration_manager.transform(raw_score)
            result.setdefault("details", {})
            result["details"]["calibration"] = {
                "raw_score": raw_score,
                "calibrated_score": calibrated,
                "method": self.calibration_config.method,
            }
            result["raw_score"] = raw_score
            result["calibrated_score"] = calibrated
            result["evaluation_score"] = calibrated
        self._calibration_applied = True
        report = self._calibration_manager.report(len(calibration_pairs))
        return {
            "method": report.method,
            "pairs_used": report.pairs_used,
            "parameters": report.parameters,
        }
    
    def get_task_status(self, task_id: str) -> Optional[EvaluationStatus]:
        """Get task status by task_id"""
        return self.task_manager.get_task_status(task_id)
    
    def list_tasks(self, status: Optional[EvaluationStatus] = None) -> List[EvaluationTask]:
        """List all tasks, optionally filtered by status"""
        return self.task_manager.list_tasks(status)
    
    def resume_tasks(self, task_ids: Optional[List[str]] = None) -> List[str]:
        """
        Resume incomplete tasks.
        
        Args:
            task_ids: List of task IDs to resume (None = all incomplete)
            
        Returns:
            List of task IDs that need to be resumed
        """
        return self.task_manager.resume_tasks(task_ids)
    
    def load_existing_results(self, reuse_mode: str = "latest") -> List[Dict[str, Any]]:
        """
        Load existing results for reuse.
        
        Args:
            reuse_mode: "latest" or specific task_id/timestamp
            
        Returns:
            List of result dictionaries
        """
        if reuse_mode == "latest":
            latest_task_id = self.storage.get_latest_results()
            if latest_task_id:
                result = self.storage.load_result(latest_task_id)
                if result:
                    return [result]
        else:
            result = self.storage.load_result(reuse_mode)
            if result:
                return [result]
        
        return []


# Convenience function for backward compatibility (deprecated)
async def evaluate_agent(question: str, agent_url: str, llm_config: Dict[str, Any], 
                        output_path: str = "output", **kwargs) -> EvaluationResult:
    """
    Convenient Agent evaluation function (deprecated).
    
    This function is deprecated. Please use AgentTrialBench with EvaluationConfig instead.
    """
    import warnings
    warnings.warn(
        "evaluate_agent() is deprecated. Use AgentTrialBench with EvaluationConfig instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Create minimal config
    llm_cfg = LLMConfig.from_dict(llm_config)
    agent_cfg = AgentConfig(
        url=agent_url,
        capabilities=kwargs.get("capabilities", [TaskCategory.WEB_RETRIEVAL]),
        timeout=kwargs.get("timeout", 30),
        close_endpoint=kwargs.get("close_endpoint"),
        api_key=kwargs.get("api_key")
    )
    dataset_cfg = DatasetConfig()
    config = EvaluationConfig(
        llm_config=llm_cfg,
        agent_config=agent_cfg,
        dataset_config=dataset_cfg,
        work_dir=output_path
    )
    
    evaluator = AgentTrialBench(config=config)
    
    agent_metadata = AgentMetadata(
        url=agent_url,
        capabilities=kwargs.get("capabilities", [TaskCategory.WEB_RETRIEVAL]),
        timeout=kwargs.get("timeout", 30),
        close_endpoint=kwargs.get("close_endpoint"),
        api_key=kwargs.get("api_key")
    )
    
    return await evaluator.evaluate_agent(
        question=question,
        agent_metadata=agent_metadata,
        context=kwargs.get("context", {}),
        category=kwargs.get("category", TaskCategory.WEB_RETRIEVAL),
        evaluation_method=kwargs.get("evaluation_method"),
        expected_answer=kwargs.get("expected_answer")
    )
