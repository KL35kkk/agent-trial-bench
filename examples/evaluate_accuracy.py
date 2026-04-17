"""
Example script for analyzing evaluation system accuracy
"""

import asyncio
import json
from agent_trial_bench.evaluation.accuracy_analysis import EvaluationAccuracyAnalyzer


async def main():
    """Analyze evaluation accuracy"""
    
    # Load evaluation results
    try:
        with open("output/evaluation_results.json", "r") as f:
            results_data = json.load(f)
            evaluation_results = results_data.get("results", [])
    except FileNotFoundError:
        print("No evaluation results found. Please run an evaluation first.")
        return
    
    if not evaluation_results:
        print("No evaluation results to analyze.")
        return
    
    analyzer = EvaluationAccuracyAnalyzer()
    
    # Comprehensive analysis
    print("=" * 60)
    print("Evaluation System Accuracy Analysis")
    print("=" * 60)
    
    # Without ground truth (basic analysis)
    analysis = analyzer.comprehensive_analysis(evaluation_results)
    
    print(f"\nTotal Evaluations: {analysis['total_evaluations']}")
    print(f"\nOverall Accuracy Score: {analysis['overall_accuracy_score']:.3f}")
    
    # Consistency analysis
    print("\n" + "-" * 60)
    print("Consistency Analysis")
    print("-" * 60)
    consistency = analysis["consistency"]
    print(f"Average Consistency: {consistency['average_consistency']:.3f}")
    print(f"Questions Analyzed: {consistency['questions_analyzed']}")
    print(f"Questions with Multiple Runs: {consistency['questions_with_multiple_runs']}")
    
    # Bias detection
    print("\n" + "-" * 60)
    print("Bias Detection")
    print("-" * 60)
    bias = analysis["bias_detection"]
    
    if "length_bias" in bias:
        print(f"Length Bias: {bias['length_bias']:.3f} ({bias.get('length_bias_direction', 'unknown')})")
        if bias['length_bias'] > 0.3:
            print("  ⚠️  Warning: Significant length bias detected!")
    
    if "category_bias" in bias:
        print("\nCategory Bias:")
        for category, bias_value in bias["category_bias"].items():
            print(f"  {category}: {bias_value:.3f}")
        if bias.get("max_category_bias", 0.0) > 0.2:
            print("  ⚠️  Warning: Significant category bias detected!")
    
    if "score_clustering" in bias:
        print(f"\nScore Clustering: {bias['score_clustering']:.3f}")
        if bias['score_clustering'] > 0.5:
            print("  ⚠️  Warning: Scores are too clustered!")
        
        print("\nScore Distribution:")
        for bin_name, count in bias.get("score_distribution", {}).items():
            print(f"  {bin_name}: {count}")
    
    # Accuracy breakdown
    print("\n" + "-" * 60)
    print("Accuracy Breakdown")
    print("-" * 60)
    for component, score in analysis.get("accuracy_breakdown", {}).items():
        print(f"{component.capitalize()}: {score:.3f}")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("Recommendations")
    print("=" * 60)
    
    recommendations = []
    
    if consistency['average_consistency'] < 0.7:
        recommendations.append(
            "⚠️  Low consistency detected. Consider: "
            "- Using more deterministic evaluation methods\n"
            "  - Reducing randomness in LLM evaluation\n"
            "  - Increasing number of evaluation samples"
        )
    
    if bias.get("length_bias", 0.0) > 0.3:
        recommendations.append(
            "⚠️  Length bias detected. Consider: "
            "- Normalizing scores by answer length\n"
            "  - Using length-agnostic evaluation metrics"
        )
    
    if bias.get("max_category_bias", 0.0) > 0.2:
        recommendations.append(
            "⚠️  Category bias detected. Consider: "
            "- Calibrating scores across categories\n"
            "  - Using category-specific evaluation thresholds"
        )
    
    if bias.get("score_clustering", 0.0) > 0.5:
        recommendations.append(
            "⚠️  Score clustering detected. Consider: "
            "- Using more granular scoring scales\n"
            "  - Improving score differentiation"
        )
    
    if analysis['overall_accuracy_score'] < 0.7:
        recommendations.append(
            "⚠️  Overall accuracy is below optimal. Review evaluation methods and parameters."
        )
    
    if not recommendations:
        print("✅ No major issues detected. Evaluation system appears to be working well.")
    else:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec}")
    
    # Save detailed analysis
    output_file = "output/accuracy_analysis.json"
    with open(output_file, "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"\n📊 Detailed analysis saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())

