"""
Cascade Evaluator for multi-channel scoring with disagreement fallback.
"""

from typing import Dict, Any, List, Optional, Tuple

from .base_evaluator import BaseEvaluator
from .hybrid_evaluator import HybridEvaluator
from .llm_evaluator import LLMEvaluator


class CascadeEvaluator(BaseEvaluator):
    """Sequential evaluator that chains multiple evaluators and falls back when
    disagreement or low confidence is detected."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        config = config or {}
        self.llm_config = config.get("llm_config", {})
        self.disagreement_threshold = config.get("disagreement_threshold", 0.25)
        self.low_confidence_threshold = config.get("low_confidence_threshold", 0.45)
        self.stage_defaults = config.get("stages") or self._default_stages()
        self.fallback_stage_cfg = config.get("fallback_stage")

        self.stage_evaluators: List[Tuple[Dict[str, Any], BaseEvaluator]] = []
        for stage_cfg in self.stage_defaults:
            self.stage_evaluators.append(
                (stage_cfg, self._build_stage_evaluator(stage_cfg)))

        self.fallback_evaluator: Optional[Tuple[Dict[str, Any], BaseEvaluator]] = None
        if self.fallback_stage_cfg:
            self.fallback_evaluator = (
                self.fallback_stage_cfg,
                self._build_stage_evaluator(self.fallback_stage_cfg),
            )

    def _default_stages(self) -> List[Dict[str, Any]]:
        """Provide a simple two-stage pipeline as default."""
        return [
            {
                "name": "rule_screen",
                "type": "hybrid",
                "weight": 0.5,
                "use_llm_evaluation": False,
            },
            {
                "name": "llm_judge",
                "type": "llm",
                "weight": 0.5,
                "llm_config": {},
            },
        ]

    def _build_stage_evaluator(self, stage_cfg: Dict[str, Any]) -> BaseEvaluator:
        """Create underlying evaluator for a stage."""
        stage_type = stage_cfg.get("type", "hybrid").lower()
        if stage_type == "llm":
            llm_cfg = dict(self.llm_config)
            llm_cfg.update(stage_cfg.get("llm_config", {}))
            return LLMEvaluator(llm_cfg)
        # default to hybrid/rule evaluator
        hybrid_cfg = {
            "use_llm_evaluation": stage_cfg.get("use_llm_evaluation", True),
            "llm_evaluation_threshold": stage_cfg.get("llm_evaluation_threshold", 0.5),
            "rule_based_weight": stage_cfg.get("rule_based_weight", 0.4),
            "llm_based_weight": stage_cfg.get("llm_based_weight", 0.6),
            "llm_config": self.llm_config,
        }
        hybrid_cfg.update(stage_cfg.get("hybrid_config", {}))
        return HybridEvaluator(hybrid_cfg)

    async def evaluate(
        self,
        question: str,
        agent_response: str,
        expected_answer: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        stage_results: List[Dict[str, Any]] = []
        needs_fallback = False

        for stage_cfg, evaluator in self.stage_evaluators:
            result = await evaluator.evaluate(
                question=question,
                agent_response=agent_response,
                expected_answer=expected_answer,
                context=context or {},
            )
            confidence = self._extract_confidence(result, stage_cfg)
            stage_results.append({
                "name": stage_cfg.get("name", evaluator.__class__.__name__),
                "score": result.get("score", 0.0),
                "confidence": confidence,
                "reasoning": result.get("reasoning", ""),
                "details": result.get("details", {}),
                "weight": stage_cfg.get("weight", 1.0),
            })
            if confidence < self.low_confidence_threshold:
                needs_fallback = True

        if len(stage_results) > 1:
            scores = [s["score"] for s in stage_results]
            if max(scores) - min(scores) >= self.disagreement_threshold:
                needs_fallback = True

        if needs_fallback and self.fallback_evaluator:
            fallback_cfg, evaluator = self.fallback_evaluator
            fallback_result = await evaluator.evaluate(
                question=question,
                agent_response=agent_response,
                expected_answer=expected_answer,
                context=context or {},
            )
            stage_results.append({
                "name": fallback_cfg.get("name", "fallback"),
                "score": fallback_result.get("score", 0.0),
                "confidence": self._extract_confidence(fallback_result, fallback_cfg),
                "reasoning": fallback_result.get("reasoning", ""),
                "details": fallback_result.get("details", {}),
                "weight": fallback_cfg.get("weight", 1.0),
                "fallback": True,
            })

        final_score = self._aggregate_score(stage_results)
        combined_reasoning = self._compose_reasoning(stage_results)

        return {
            "score": final_score,
            "reasoning": combined_reasoning,
            "details": {
                "stage_results": stage_results,
                "evaluation_method": "cascade",
                "disagreement_threshold": self.disagreement_threshold,
            },
        }

    def _aggregate_score(self, stages: List[Dict[str, Any]]) -> float:
        if not stages:
            return 0.0
        weighted = 0.0
        total_weight = 0.0
        for stage in stages:
            weight = max(0.0, float(stage.get("weight", 1.0)))
            weighted += stage["score"] * weight
            total_weight += weight
        if total_weight == 0:
            return 0.0
        return max(0.0, min(1.0, weighted / total_weight))

    def _compose_reasoning(self, stages: List[Dict[str, Any]]) -> str:
        parts = []
        for stage in stages:
            prefix = "[fallback] " if stage.get("fallback") else ""
            parts.append(
                f"{prefix}{stage['name']}: {stage['score']:.2f} "
                f"(conf={stage['confidence']:.2f}) -> {stage['reasoning']}"
            )
        return "\n".join(parts)

    def _extract_confidence(self, result: Dict[str, Any], stage_cfg: Dict[str, Any]) -> float:
        details = result.get("details", {}) or {}
        confidence = details.get("confidence", result.get("confidence", 0.0))
        if isinstance(confidence, (int, float)):
            return max(0.0, min(1.0, float(confidence)))
        return stage_cfg.get("default_confidence", 0.7)

    def get_capabilities(self) -> List[str]:
        return [
            "cascade_evaluation",
            "multi_stage_scoring",
            "disagreement_detection",
            "fallback_llm_judge",
        ]
