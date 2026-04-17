"""
Evaluation Engine for Agent Trial Bench
Core evaluation logic separated from execution
"""

import time
from typing import Dict, Any, List, Optional
import logging

from .evaluation.base_evaluator import BaseEvaluator
from .evaluation.llm_evaluator import LLMEvaluator
from .evaluation.hybrid_evaluator import HybridEvaluator
from .evaluation.cascade_evaluator import CascadeEvaluator
from .enums import EvaluationMethod, TaskCategory, EvaluationStatus
from .runners.agent_runner import AgentRunner

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """Core evaluation engine that handles evaluation logic.
    
    This class is responsible for:
    - Managing evaluators
    - Executing evaluations
    - Processing evaluation results
    """
    
    def __init__(self, llm_config: Dict[str, Any], evaluator_config: Optional[Dict[str, Any]] = None,
                 agent_runner: Optional[AgentRunner] = None):
        """
        Initialize evaluation engine.
        
        Args:
            llm_config: LLM configuration
            evaluator_config: Evaluator configuration (optional)
        """
        self.llm_config = llm_config
        self.evaluator_config = evaluator_config or {}
        self.evaluators: Dict[str, BaseEvaluator] = {}
        self.agent_runner = agent_runner or AgentRunner(config={'timeout': 30})
        self.default_method = (self.evaluator_config.get("type") or "hybrid").lower()
        self._llm_evaluator_config: Dict[str, Any] = {}
        
        # Initialize evaluators
        self._init_evaluators()
    
    def _init_evaluators(self):
        """Initialize evaluators with LLM configuration"""
        try:
            # Validate LLM configuration
            required_keys = ["model", "base_url", "api_key"]
            missing_keys = [key for key in required_keys if key not in self.llm_config]
            if missing_keys:
                logger.warning(f"Missing LLM config keys: {missing_keys}. Some evaluators may not work.")
                return
            
            # Initialize LLM Evaluator
            llm_evaluator_config = {
                "model": self.llm_config["model"],
                "base_url": self.llm_config["base_url"],
                "api_key": self.llm_config["api_key"],
                "temperature": self.llm_config.get("temperature", 0.3),
                "max_tokens": self.llm_config.get("max_tokens", 2000),
                "enable_cache": self.llm_config.get("enable_cache", False),
                "cache_dir": self.llm_config.get("cache_dir", "output/llm_cache"),
                "random_seed": self.llm_config.get("random_seed"),
            }
            self._llm_evaluator_config = llm_evaluator_config
            self.evaluators["llm"] = LLMEvaluator(llm_evaluator_config)
            
            # Initialize Hybrid Evaluator
            hybrid_config = {
                "use_llm_evaluation": self.evaluator_config.get("use_llm_evaluation", True),
                "llm_evaluation_threshold": self.evaluator_config.get("llm_evaluation_threshold", 0.5),
                "rule_based_weight": self.evaluator_config.get("rule_based_weight", 0.3),
                "llm_based_weight": self.evaluator_config.get("llm_based_weight", 0.7),
                "llm_config": llm_evaluator_config
            }
            self.evaluators["hybrid"] = HybridEvaluator(hybrid_config)
            cascade_cfg = self.evaluator_config.get("cascade_config") or {}
            cascade_cfg.setdefault("llm_config", llm_evaluator_config)
            self.evaluators["cascade"] = CascadeEvaluator(cascade_cfg)
            
            logger.info("Evaluators initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize evaluators: {e}")
            logger.info("Using simplified evaluation logic")
    
    def select_evaluation_method(self, category: TaskCategory) -> EvaluationMethod:
        """Select an evaluation method for ``category``.

        Priority:
        1. If ``evaluator_config.type`` is explicitly set, honour it.
        2. Optional ``category_method_map`` lets integrators pin a method per
           category (e.g. ``{"fact_qa": "rule_based"}``).
        3. Fall back to a set of category-family defaults that work for both
           general and Web3 evaluations.
        """
        if self.default_method == "cascade":
            return EvaluationMethod.CASCADE

        custom_map = (self.evaluator_config.get("category_method_map") or {}) if self.evaluator_config else {}
        if category.value in custom_map:
            try:
                return EvaluationMethod(custom_map[category.value])
            except ValueError:
                logger.warning(
                    "Invalid evaluation method '%s' for category %s; ignoring.",
                    custom_map[category.value],
                    category.value,
                )

        # Default heuristics – favour rule-based when expected answers are
        # highly formatted (fact QA, web retrieval), LLM-based when semantic
        # judgement dominates, hybrid otherwise.
        rule_based = {TaskCategory.FACT_QA, TaskCategory.WEB_RETRIEVAL}
        llm_based = {
            TaskCategory.CONVERSATIONAL,
            TaskCategory.REASONING,
            TaskCategory.ONCHAIN_RETRIEVAL,
        }
        if category in rule_based:
            return EvaluationMethod.RULE_BASED
        if category in llm_based:
            return EvaluationMethod.LLM_BASED
        return EvaluationMethod.HYBRID
    
    def get_evaluator(self, method: EvaluationMethod) -> Optional[BaseEvaluator]:
        """Get evaluator for the given method"""
        if method == EvaluationMethod.LLM_BASED:
            return self.evaluators.get("llm")
        elif method == EvaluationMethod.HYBRID:
            return self.evaluators.get("hybrid")
        elif method == EvaluationMethod.CASCADE:
            return self.evaluators.get("cascade")
        else:
            # Rule-based evaluation uses simplified logic
            return None
    
    async def evaluate_response(self,
                               question: str,
                               agent_response: str,
                               expected_answer: Optional[str] = None,
                               category: TaskCategory = TaskCategory.GENERAL,
                               evaluation_method: Optional[EvaluationMethod] = None,
                               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate agent response.
        
        Args:
            question: Evaluation question
            agent_response: Agent response text
            expected_answer: Expected answer (optional)
            category: Task category
            evaluation_method: Evaluation method (auto-selected if None)
            context: Additional context
            
        Returns:
            Evaluation result dictionary
        """
        # Select evaluation method if not provided
        if evaluation_method is None:
            evaluation_method = self.select_evaluation_method(category)
        
        # Get evaluator
        evaluator = self.get_evaluator(evaluation_method)
        
        if evaluator:
            try:
                # Use professional evaluator
                result = await evaluator.evaluate(
                    question=question,
                    agent_response=agent_response,
                    expected_answer=expected_answer,
                    context=context or {}
                )
                return {
                    "evaluation_score": result.get("score", 0.0),
                    "evaluation_reasoning": result.get("reasoning", ""),
                    "details": result.get("details", {}),
                    "evaluation_method": evaluation_method.value
                }
            except Exception as e:
                logger.warning(f"Professional evaluator failed: {e}, using fallback")
        
        # Fallback to simplified evaluation
        return await self._simple_evaluate(question, agent_response, expected_answer, context)
    
    async def _simple_evaluate(self,
                              question: str,
                              agent_response: str,
                              expected_answer: Optional[str],
                              context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Simplified evaluation fallback"""
        score = 0.0
        reasoning = ""
        
        if expected_answer:
            # Simple string matching
            if expected_answer.lower().strip() in agent_response.lower():
                score = 0.8
                reasoning = "Expected answer found in response"
            else:
                # Check for key information
                key_terms = set(expected_answer.lower().split())
                response_terms = set(agent_response.lower().split())
                overlap = len(key_terms.intersection(response_terms)) / len(key_terms) if key_terms else 0
                score = overlap * 0.6
                reasoning = f"Partial match: {overlap:.2%} key terms found"
        else:
            # No expected answer - basic quality check
            if len(agent_response) > 50:
                score = 0.5
                reasoning = "Response has reasonable length"
            else:
                score = 0.2
                reasoning = "Response is too short"
        
        return {
            "evaluation_score": score,
            "evaluation_reasoning": reasoning,
            "details": {},
            "evaluation_method": "simplified"
        }
    
    async def evaluate_task(self,
                           question: str,
                           agent_url: str,
                           agent_timeout: int = 30,
                           expected_answer: Optional[str] = None,
                           category: TaskCategory = TaskCategory.GENERAL,
                           evaluation_method: Optional[EvaluationMethod] = None,
                           context: Optional[Dict[str, Any]] = None,
                           close_endpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a complete task (call agent + evaluate response).
        
        Args:
            question: Evaluation question
            agent_url: Agent API URL
            agent_timeout: Agent request timeout
            expected_answer: Expected answer (optional)
            category: Task category
            evaluation_method: Evaluation method (auto-selected if None)
            context: Additional context
            close_endpoint: Agent close endpoint (optional)
            
        Returns:
            Complete evaluation result dictionary
        """
        start_time = time.time()
        context = dict(context or {})
        question_id = context.get("question_id") or context.get("dataset_id") or context.get("id")
        
        try:
            # Call agent API
            if agent_url and agent_url.startswith("mock://"):
                agent_response_data = self._simulate_agent_response(
                    question=question,
                    expected_answer=expected_answer,
                    context=context
                )
            else:
                context.pop("mock_response", None)
                context.pop("_expected_answer", None)
                agent_response_data = await self.agent_runner.call_agent_api(
                    url=agent_url,
                    question=question,
                    context=context,
                    timeout=agent_timeout
                )
            
            agent_response = agent_response_data.get("answer", "")
            confidence = agent_response_data.get("confidence", 0.0)
            processing_time = time.time() - start_time
            
            # Evaluate response
            evaluation_result = await self.evaluate_response(
                question=question,
                agent_response=agent_response,
                expected_answer=expected_answer,
                category=category,
                evaluation_method=evaluation_method,
                context=context
            )
            
            # Close agent if needed
            if close_endpoint:
                await self.agent_runner.close_agent(close_endpoint)
            
            evaluated_at = time.time()
            return {
                "status": EvaluationStatus.COMPLETED.value,
                "question": question,
                "question_id": question_id,
                "agent_response": agent_response,
                "evaluation_score": evaluation_result["evaluation_score"],
                "evaluation_reasoning": evaluation_result["evaluation_reasoning"],
                "confidence": confidence,
                "processing_time": processing_time,
                "tools_used": agent_response_data.get("tools_used", []),
                "metadata": agent_response_data.get("metadata", {}),
                "details": evaluation_result.get("details", {}),
                "category": category.value,
                "evaluation_method": evaluation_result.get("evaluation_method", "unknown"),
                "evaluated_at": evaluated_at,
                "error": None
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Task evaluation failed: {e}")
            
            evaluated_at = time.time()
            return {
                "status": EvaluationStatus.FAILED.value,
                "question": question,
                "question_id": question_id,
                "agent_response": "",
                "evaluation_score": 0.0,
                "evaluation_reasoning": "",
                "confidence": 0.0,
                "processing_time": processing_time,
                "tools_used": [],
                "metadata": {},
                "details": {},
                "category": category.value,
                "evaluation_method": evaluation_method.value if evaluation_method else "unknown",
                "evaluated_at": evaluated_at,
                "error": str(e)
            }
    
    def _simulate_agent_response(self,
                                 question: str,
                                 expected_answer: Optional[str],
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a deterministic mock agent response for offline evaluation."""
        reference = (
            context.get("mock_response")
            or context.get("_expected_answer")
            or expected_answer
            or f"I am unable to locate the answer to: {question}"
        )
        response = reference
        confidence = 0.85
        dataset_id = context.get("dataset_id") or context.get("question_id")
        if dataset_id:
            response = f"[mock:{dataset_id}] {reference}"
        return {
            "answer": response,
            "confidence": confidence,
            "tools_used": [],
            "metadata": {"mock": True, "source_question": question}
        }
