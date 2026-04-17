"""
Hybrid Evaluator for Agent Trial Bench
Enhanced hybrid evaluator with improved scoring system
"""

from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from .llm_evaluator import LLMEvaluator
from .enhanced_scoring import EnhancedScoringSystem, ScoringMethod, ScoringResult


class HybridEvaluator(BaseEvaluator):
    """Hybrid evaluator"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        config = config or {}

        self.use_llm_evaluation = config.get("use_llm_evaluation", True)
        self.llm_evaluation_threshold = config.get("llm_evaluation_threshold", 0.5)
        self.rule_based_weight = config.get("rule_based_weight", 0.3)
        self.llm_based_weight = config.get("llm_based_weight", 0.7)
        self.use_enhanced_scoring = config.get("use_enhanced_scoring", True)

        llm_config = config.get("llm_config", {})
        self.llm_evaluator = LLMEvaluator(llm_config)
        self.enhanced_scoring = EnhancedScoringSystem()

    async def evaluate(
        self,
        question: str,
        agent_response: str,
        expected_answer: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Enhanced hybrid evaluation of Agent response"""

        rule_score, enhanced_reasoning, rule_dimensions = self._evaluate_rule_component(
            question, agent_response, expected_answer, context
        )

        llm_score, llm_reasoning, llm_dimensions = await self._evaluate_llm_component(
            question, agent_response, expected_answer, context
        )

        weight_context = self._calculate_dynamic_weights(rule_score, llm_score)
        effective_llm_weight = weight_context["llm_weight"]
        threshold_note = weight_context["threshold_note"]

        weight_sum = weight_context["rule_weight"] + effective_llm_weight
        if weight_sum == 0:
            final_score = 0.0
        else:
            final_score = (
                rule_score * weight_context["rule_weight"] + llm_score * effective_llm_weight
            ) / weight_sum

        combined_dimensions = self._combine_dimension_breakdown(
            rule_dimensions,
            llm_dimensions,
            weight_context["rule_weight"],
            effective_llm_weight,
        )

        reasoning_lines = [
            f"Rule-based evaluation: {rule_score:.2f} (weight: {weight_context['rule_weight']:.2f})",
            f"LLM evaluation: {llm_score:.2f} (weight: {effective_llm_weight:.2f})",
            f"Comprehensive score: {final_score:.2f}",
        ]
        if threshold_note:
            reasoning_lines.append(threshold_note)
        if self.use_enhanced_scoring and expected_answer:
            reasoning_lines.append(f"Enhanced scoring details: {enhanced_reasoning}")
        if combined_dimensions:
            dimension_summary = ", ".join(
                f"{dim}: {value:.2f}" for dim, value in combined_dimensions.items()
            )
            reasoning_lines.append(f"Dimension breakdown -> {dimension_summary}")

        return {
            "score": min(1.0, max(0.0, final_score)),
            "reasoning": "\n".join(reasoning_lines),
            "details": {
                "rule_based_score": rule_score,
                "llm_based_score": llm_score,
                "rule_based_weight": weight_context["rule_weight"],
                "llm_based_weight": self.llm_based_weight,
                "effective_llm_weight": effective_llm_weight,
                "llm_reasoning": llm_reasoning,
                "enhanced_scoring": self.use_enhanced_scoring,
                "enhanced_reasoning": enhanced_reasoning,
                "weighting_notes": weight_context["notes"],
                "dimension_breakdown": combined_dimensions,
                "evaluation_method": "enhanced_hybrid",
            },
        }

    def _evaluate_rule_component(
        self,
        question: str,
        agent_response: str,
        expected_answer: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> Tuple[float, str, Dict[str, float]]:
        if self.use_enhanced_scoring and expected_answer:
            try:
                scoring_method = self._select_optimal_scoring_method(question, expected_answer, context)
                enhanced_result = self.enhanced_scoring.score_answer(
                    expected=expected_answer,
                    agent=agent_response,
                    method=scoring_method,
                    context={"question": question, "context": context or {}},
                )
                score = max(0.0, min(1.0, enhanced_result.score * enhanced_result.confidence))
                dimensions = self._dimensions_from_enhanced(enhanced_result, agent_response)
                return score, enhanced_result.reasoning, dimensions
            except Exception as exc:
                fallback_result = self._rule_based_evaluation(
                    question, agent_response, expected_answer, context
                )
                fallback_score = max(0.0, min(1.0, fallback_result.score * fallback_result.confidence))
                fallback_dimensions = self._dimensions_from_enhanced(fallback_result, agent_response)
                return fallback_score, f"Enhanced scoring failed, fallback triggered: {exc}", fallback_dimensions

        fallback_result = self._rule_based_evaluation(question, agent_response, expected_answer, context)
        fallback_score = max(0.0, min(1.0, fallback_result.score * fallback_result.confidence))
        fallback_dimensions = self._dimensions_from_enhanced(fallback_result, agent_response)
        if expected_answer:
            reasoning = "Enhanced scoring disabled; question-context fallback applied"
        else:
            reasoning = "No expected answer provided; question-context scoring applied"
        return fallback_score, reasoning, fallback_dimensions

    async def _evaluate_llm_component(
        self,
        question: str,
        agent_response: str,
        expected_answer: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> Tuple[float, str, Dict[str, float]]:
        if not self.use_llm_evaluation:
            return 0.0, "LLM evaluation disabled", {}

        try:
            llm_result = await self.llm_evaluator.evaluate(
                question, agent_response, expected_answer, context
            )
            score = float(llm_result.get("score", 0.0))
            reasoning = llm_result.get("reasoning", "")
            details = llm_result.get("details", {}) or {}
            confidence = float(details.get("confidence", 1.0))
            llm_dimensions = details.get("dimension_breakdown", {}) or {}
            adjusted_score = max(0.0, min(1.0, score))
            weighted_score = adjusted_score * min(1.0, max(0.0, confidence))
            return weighted_score, reasoning, llm_dimensions
        except Exception as exc:
            return 0.0, f"LLM evaluation failed: {exc}", {}

    def _rule_based_evaluation(
        self,
        question: str,
        agent_response: str,
        expected_answer: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ScoringResult:
        """Question-context fallback scoring when enhanced scoring is unavailable."""

        context_payload = {
            "question": question,
            "expected": expected_answer,
            "context": context or {},
        }
        reference = expected_answer or question or ""
        return self.enhanced_scoring.score_against_question(
            question or "",
            agent_response,
            context=context_payload,
            reference=reference,
        )

    def _combine_dimension_breakdown(
        self,
        rule_dimensions: Dict[str, float],
        llm_dimensions: Dict[str, float],
        rule_weight: float,
        llm_weight: float,
    ) -> Dict[str, float]:
        dimensions = {}
        target_dims = {"accuracy", "completeness", "professionalism", "usefulness"}
        for dim in target_dims:
            weighted_sum = 0.0
            total_weight = 0.0
            if dim in rule_dimensions:
                weighted_sum += rule_dimensions[dim] * rule_weight
                total_weight += rule_weight
            if dim in llm_dimensions:
                weighted_sum += llm_dimensions[dim] * llm_weight
                total_weight += llm_weight
            if total_weight > 0:
                dimensions[dim] = max(0.0, min(1.0, weighted_sum / total_weight))
        return dimensions

    def _dimensions_from_enhanced(
        self,
        enhanced_result,
        agent_response: str,
    ) -> Dict[str, float]:
        breakdown = enhanced_result.breakdown or {}
        professionalism = self.enhanced_scoring._calculate_professionalism(agent_response)
        accuracy = breakdown.get("factual") or breakdown.get("semantic") or enhanced_result.score
        completeness = breakdown.get("completeness") or breakdown.get("content") or enhanced_result.score
        usefulness = breakdown.get("content") or enhanced_result.score
        usefulness = min(1.0, 0.6 * (usefulness or enhanced_result.score) + 0.4 * professionalism)
        return {
            "accuracy": max(0.0, min(1.0, accuracy)),
            "completeness": max(0.0, min(1.0, completeness)),
            "professionalism": max(0.0, min(1.0, professionalism)),
            "usefulness": max(0.0, min(1.0, usefulness)),
        }


    def _calculate_dynamic_weights(self, rule_score: float, llm_score: float) -> Dict[str, Any]:
        """Adjust weights based on confidence and threshold signals."""

        notes: List[str] = []
        rule_weight = self.rule_based_weight
        llm_weight = self.llm_based_weight
        threshold_note = ""

        if not self.use_llm_evaluation:
            return {
                "rule_weight": 1.0,
                "llm_weight": 0.0,
                "threshold_note": "LLM evaluation disabled; full weight on rule component.",
                "notes": notes,
            }

        if llm_score < self.llm_evaluation_threshold:
            llm_weight = 0.0
            threshold_note = (
                f"LLM score {llm_score:.2f} < threshold {self.llm_evaluation_threshold:.2f}; ignored."
            )
        elif llm_score >= 0.85:
            notes.append("High-confidence LLM score detected; boosting LLM weight.")
            llm_weight = min(1.0, llm_weight + 0.1)
            rule_weight = max(0.0, rule_weight - 0.1)

        if rule_score < 0.3 and llm_weight > 0:
            notes.append("Low rule score detected; shifting weight towards LLM.")
            llm_weight = min(1.0, llm_weight + 0.05)
            rule_weight = max(0.0, rule_weight - 0.05)

        normalization_factor = rule_weight + llm_weight
        if normalization_factor == 0:
            rule_weight = 0.0
            llm_weight = 0.0
        else:
            rule_weight /= normalization_factor
            llm_weight /= normalization_factor

        return {
            "rule_weight": rule_weight,
            "llm_weight": llm_weight,
            "threshold_note": threshold_note,
            "notes": notes,
        }

    def _select_optimal_scoring_method(
        self,
        question: str,
        expected_answer: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ScoringMethod:
        question_text = " ".join(
            filter(None, [question or "", expected_answer or "", (context or {}).get("category") or ""])
        ).lower()

        if any(keyword in question_text for keyword in ["date", "when", "time", "timestamp"]):
            return ScoringMethod.FORMAT_STRICT
        if any(keyword in question_text for keyword in ["price", "amount", "value", "cost", "quantity", "total"]):
            return ScoringMethod.FORMAT_STRICT
        if any(keyword in question_text for keyword in ["which", "where", "list", "compare"]):
            return ScoringMethod.CONTENT_FOCUSED
        if any(
            keyword in question_text
            for keyword in ["how", "what is", "explain", "describe"]
        ):
            return ScoringMethod.CONTENT_FOCUSED
        return ScoringMethod.HYBRID_BALANCED

    def get_capabilities(self) -> List[str]:
        """Get evaluator capabilities."""
        return [
            "hybrid_evaluation",
            "rule_based_evaluation",
            "llm_based_evaluation",
            "enhanced_scoring",
            "format_normalization",
            "semantic_similarity",
            "content_focused_evaluation",
            "multi_dimensional_scoring",
        ]
