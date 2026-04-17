"""
Evaluation Accuracy Analysis Module
Analyzes the precision and reliability of the evaluation system itself
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class AccuracyMetrics:
    """Metrics for evaluating evaluation accuracy"""
    consistency_score: float  # Score consistency across multiple runs
    inter_annotator_agreement: float  # Agreement between different evaluators
    bias_detection: Dict[str, float]  # Detected biases
    false_positive_rate: float  # Incorrectly high scores
    false_negative_rate: float  # Incorrectly low scores
    score_distribution: Dict[str, int]  # Distribution of scores
    confidence_correlation: float  # Correlation between confidence and accuracy


class EvaluationAccuracyAnalyzer:
    """Analyzer for evaluating the accuracy of the evaluation system itself"""
    
    def __init__(self):
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def analyze_consistency(self, 
                           evaluation_results: List[Dict[str, Any]],
                           question_id_key: str = "question") -> Dict[str, Any]:
        """
        Analyze consistency of evaluations across multiple runs.
        
        Args:
            evaluation_results: List of evaluation result dictionaries
            question_id_key: Key to identify same questions across runs
            
        Returns:
            Consistency analysis results
        """
        # Group results by question
        question_groups = defaultdict(list)
        for result in evaluation_results:
            question = result.get(question_id_key, "")
            question_groups[question].append(result)
        
        consistency_scores = []
        for question, results in question_groups.items():
            if len(results) < 2:
                continue
            
            scores = [r.get("evaluation_score", 0.0) for r in results]
            # Calculate coefficient of variation (lower is more consistent)
            if len(scores) > 1:
                mean_score = sum(scores) / len(scores)
                if mean_score > 0:
                    variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
                    std_dev = variance ** 0.5
                    cv = std_dev / mean_score if mean_score > 0 else 1.0
                    consistency = 1.0 - min(1.0, cv)  # Convert to consistency score
                    consistency_scores.append(consistency)
        
        avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
        
        return {
            "average_consistency": avg_consistency,
            "consistency_scores": consistency_scores,
            "questions_analyzed": len(question_groups),
            "questions_with_multiple_runs": sum(1 for r in question_groups.values() if len(r) >= 2)
        }
    
    def detect_variance_alerts(
            self,
            evaluation_results: List[Dict[str, Any]],
            question_id_key: str = "question",
            variance_threshold: float = 0.2,
            min_runs: int = 2) -> Dict[str, Any]:
        """Identify questions whose scores vary too much across runs."""
        alerts = []
        groups = defaultdict(list)
        for result in evaluation_results:
            qid = result.get(question_id_key, "")
            groups[qid].append(result.get("evaluation_score", 0.0))
        for qid, scores in groups.items():
            if len(scores) < min_runs:
                continue
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            std_dev = variance ** 0.5
            if std_dev >= variance_threshold:
                alerts.append({
                    "question": qid,
                    "std_dev": std_dev,
                    "mean_score": mean_score,
                    "runs": len(scores),
                })
        return {
            "alerts": alerts,
            "threshold": variance_threshold,
            "min_runs": min_runs,
            "total_questions": len(groups),
        }
    
    def detect_bias(self, 
                    evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect potential biases in evaluation.
        
        Checks for:
        - Length bias (preferring longer/shorter answers)
        - Format bias (preferring certain formats)
        - Category bias (different scores for different categories)
        - Score clustering (too many scores at certain values)
        """
        biases = {}
        
        # Length bias analysis
        length_scores = []
        for result in evaluation_results:
            response = result.get("agent_response", "")
            score = result.get("evaluation_score", 0.0)
            if response:
                length_scores.append((len(response), score))
        
        if length_scores:
            # Calculate correlation between length and score
            lengths, scores = zip(*length_scores)
            length_bias = self._calculate_correlation(list(lengths), list(scores))
            biases["length_bias"] = abs(length_bias)
            biases["length_bias_direction"] = "positive" if length_bias > 0 else "negative"
        
        # Category bias analysis
        category_scores = defaultdict(list)
        for result in evaluation_results:
            category = result.get("category", "unknown")
            score = result.get("evaluation_score", 0.0)
            category_scores[category].append(score)
        
        if len(category_scores) > 1:
            category_means = {cat: sum(scores) / len(scores) 
                            for cat, scores in category_scores.items()}
            overall_mean = sum(result.get("evaluation_score", 0.0) 
                             for result in evaluation_results) / len(evaluation_results)
            
            category_bias = {}
            for cat, mean_score in category_means.items():
                bias = abs(mean_score - overall_mean)
                category_bias[cat] = bias
            
            biases["category_bias"] = category_bias
            biases["max_category_bias"] = max(category_bias.values()) if category_bias else 0.0
        
        # Score clustering analysis
        scores = [r.get("evaluation_score", 0.0) for r in evaluation_results]
        score_bins = {
            "very_low": sum(1 for s in scores if 0.0 <= s < 0.2),
            "low": sum(1 for s in scores if 0.2 <= s < 0.4),
            "medium": sum(1 for s in scores if 0.4 <= s < 0.6),
            "high": sum(1 for s in scores if 0.6 <= s < 0.8),
            "very_high": sum(1 for s in scores if 0.8 <= s <= 1.0),
        }
        
        # Check for clustering (too many scores in one bin)
        total = len(scores)
        if total > 0:
            max_bin_ratio = max(score_bins.values()) / total
            biases["score_clustering"] = max_bin_ratio
            biases["score_distribution"] = score_bins
        
        return biases
    
    def analyze_confidence_accuracy(self,
                                   evaluation_results: List[Dict[str, Any]],
                                   ground_truth: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Analyze correlation between confidence scores and actual accuracy.
        
        Args:
            evaluation_results: List of evaluation results
            ground_truth: Optional ground truth scores (question -> score mapping)
        """
        if not ground_truth:
            return {
                "confidence_accuracy_correlation": 0.0,
                "note": "Ground truth not provided, cannot calculate accuracy correlation"
            }
        
        confidence_scores = []
        actual_errors = []
        
        for result in evaluation_results:
            question = result.get("question", "")
            if question not in ground_truth:
                continue
            
            confidence = result.get("confidence", 0.0)
            predicted_score = result.get("evaluation_score", 0.0)
            true_score = ground_truth[question]
            
            error = abs(predicted_score - true_score)
            
            confidence_scores.append(confidence)
            actual_errors.append(error)
        
        if len(confidence_scores) > 1:
            # Negative correlation is good (higher confidence = lower error)
            correlation = -self._calculate_correlation(confidence_scores, actual_errors)
        else:
            correlation = 0.0
        
        return {
            "confidence_accuracy_correlation": correlation,
            "samples_analyzed": len(confidence_scores),
            "interpretation": "Higher correlation means confidence better predicts accuracy"
        }
    
    def detect_false_positives_negatives(self,
                                        evaluation_results: List[Dict[str, Any]],
                                        ground_truth: Dict[str, float],
                                        threshold: float = 0.7) -> Dict[str, Any]:
        """
        Detect false positives (incorrectly high scores) and false negatives (incorrectly low scores).
        
        Args:
            evaluation_results: List of evaluation results
            ground_truth: Ground truth scores (question -> score mapping)
            threshold: Score threshold for positive/negative classification
        """
        false_positives = 0
        false_negatives = 0
        true_positives = 0
        true_negatives = 0
        
        for result in evaluation_results:
            question = result.get("question", "")
            if question not in ground_truth:
                continue
            
            predicted_score = result.get("evaluation_score", 0.0)
            true_score = ground_truth[question]
            
            predicted_positive = predicted_score >= threshold
            true_positive = true_score >= threshold
            
            if predicted_positive and true_positive:
                true_positives += 1
            elif predicted_positive and not true_positive:
                false_positives += 1
            elif not predicted_positive and true_positive:
                false_negatives += 1
            else:
                true_negatives += 1
        
        total = len([r for r in evaluation_results if r.get("question") in ground_truth])
        
        if total == 0:
            return {
                "false_positive_rate": 0.0,
                "false_negative_rate": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }
        
        fp_rate = false_positives / total if total > 0 else 0.0
        fn_rate = false_negatives / total if total > 0 else 0.0
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "false_positive_rate": fp_rate,
            "false_negative_rate": fn_rate,
            "true_positives": true_positives,
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "total_samples": total
        }
    
    def comprehensive_analysis(self,
                              evaluation_results: List[Dict[str, Any]],
                              ground_truth: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive accuracy analysis.
        
        Args:
            evaluation_results: List of evaluation results
            ground_truth: Optional ground truth scores
        """
        analysis = {
            "total_evaluations": len(evaluation_results),
            "consistency": self.analyze_consistency(evaluation_results),
            "bias_detection": self.detect_bias(evaluation_results),
        }
        analysis["variance_alerts"] = self.detect_variance_alerts(evaluation_results)
        
        if ground_truth:
            analysis["confidence_accuracy"] = self.analyze_confidence_accuracy(
                evaluation_results, ground_truth
            )
            analysis["error_analysis"] = self.detect_false_positives_negatives(
                evaluation_results, ground_truth
            )
        
        # Overall accuracy score
        accuracy_components = []
        
        # Consistency component
        consistency = analysis["consistency"].get("average_consistency", 0.0)
        accuracy_components.append(("consistency", consistency, 0.3))
        
        # Bias component (lower bias is better)
        max_bias = analysis["bias_detection"].get("max_category_bias", 0.0)
        bias_score = max(0.0, 1.0 - max_bias)
        accuracy_components.append(("bias", bias_score, 0.2))
        
        if ground_truth:
            # Error component
            error_analysis = analysis.get("error_analysis", {})
            f1 = error_analysis.get("f1_score", 0.0)
            accuracy_components.append(("error_rate", f1, 0.5))
        
        # Calculate weighted accuracy score
        total_weight = sum(weight for _, _, weight in accuracy_components)
        overall_accuracy = sum(score * weight for _, score, weight in accuracy_components) / total_weight if total_weight > 0 else 0.0
        
        analysis["overall_accuracy_score"] = overall_accuracy
        analysis["accuracy_breakdown"] = {
            component: score for component, score, _ in accuracy_components
        }
        
        return analysis

    def calculate_pass_at_k(
        self,
        evaluation_results: List[Dict[str, Any]],
        k: int = 1,
        threshold: float = 0.7,
        question_id_key: str = "question",
    ) -> Dict[str, Any]:
        """
        Calculate Pass@k per question and overall (industry-standard metric).
        
        Args:
            evaluation_results: List of evaluation results (with evaluation_score)
            k: Pass@k parameter
            threshold: Score threshold for counting a success
            question_id_key: Key to group multiple runs of the same question
        """
        question_groups: Dict[str, List[float]] = defaultdict(list)
        for result in evaluation_results:
            qid = result.get(question_id_key, "")
            score = float(result.get("evaluation_score", 0.0))
            question_groups[qid].append(score)

        pass_details = {}
        per_question_pass = []
        for qid, scores in question_groups.items():
            n = len(scores)
            c = sum(1 for s in scores if s >= threshold)
            if n == 0 or k <= 0:
                continue
            if n < k:
                # Not enough samples; approximate with empirical pass rate
                pass_prob = c / n
            elif n - c < k:
                pass_prob = 1.0
            else:
                # Pass@k = 1 - comb(n - c, k) / comb(n, k)
                numerator = math.comb(n - c, k) if n - c >= k else 0
                denominator = math.comb(n, k)
                pass_prob = 1.0 - (numerator / denominator if denominator > 0 else 0.0)
            pass_details[qid] = {
                "n": n,
                "successes": c,
                "pass_at_k": pass_prob,
            }
            per_question_pass.append(pass_prob)

        overall = sum(per_question_pass) / len(per_question_pass) if per_question_pass else 0.0
        return {
            "overall_pass_at_k": overall,
            "threshold": threshold,
            "k": k,
            "total_questions": len(per_question_pass),
            "details": pass_details,
        }
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator_term = (n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)
        if denominator_term <= 0:
            return 0.0
        denominator = math.sqrt(denominator_term)
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
