"""
Default Summarizer for Agent Trial Bench
"""

import os
import json
import csv
from typing import List, Dict, Any, Optional
from collections import defaultdict
import logging

from .base import BaseSummarizer

logger = logging.getLogger(__name__)


class DefaultSummarizer(BaseSummarizer):
    """Default summarizer that provides comprehensive result analysis.
    
    Features:
    - Overall statistics
    - Category-based analysis
    - Score distribution
    - Success/failure rates
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.summary_groups = self.config.get('summary_groups', [])
    
    def summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Summarize evaluation results.
        
        Args:
            results: List of evaluation result dictionaries
            
        Returns:
            Summary dictionary with:
            - overall: Overall statistics
            - by_category: Statistics grouped by category
            - by_method: Statistics grouped by evaluation method
            - score_distribution: Score distribution analysis
            - details: Detailed results
        """
        if not results:
            return {
                'overall': {
                    'total_tasks': 0,
                    'successful_tasks': 0,
                    'failed_tasks': 0,
                    'success_rate': 0.0,
                    'average_score': 0.0,
                    'average_confidence': 0.0,
                    'average_processing_time': 0.0
                },
                'by_category': {},
                'by_method': {},
                'score_distribution': {},
                'details': []
            }
        
        # Separate successful and failed results
        successful = [r for r in results if r.get('status') == 'completed']
        failed = [r for r in results if r.get('status') == 'failed']
        
        # Overall statistics
        overall = {
            'total_tasks': len(results),
            'successful_tasks': len(successful),
            'failed_tasks': len(failed),
            'success_rate': len(successful) / len(results) if results else 0.0,
            'average_score': sum(r.get('evaluation_score', 0.0) for r in successful) / len(successful) if successful else 0.0,
            'average_confidence': sum(r.get('confidence', 0.0) for r in successful) / len(successful) if successful else 0.0,
            'average_processing_time': sum(r.get('processing_time', 0.0) for r in successful) / len(successful) if successful else 0.0
        }
        
        # Category-based statistics
        by_category = self._aggregate_by_category(results)
        
        # Method-based statistics
        by_method = self._aggregate_by_method(results)
        
        # Score distribution
        score_distribution = self._calculate_score_distribution(successful)
        
        return {
            'overall': overall,
            'by_category': by_category,
            'by_method': by_method,
            'score_distribution': score_distribution,
            'details': results
        }
    
    def _aggregate_by_category(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Aggregate statistics by task category"""
        category_stats = defaultdict(lambda: {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'scores': [],
            'confidences': [],
            'processing_times': []
        })
        
        for result in results:
            category = result.get('category', 'unknown')
            stats = category_stats[category]
            stats['total'] += 1
            
            if result.get('status') == 'completed':
                stats['successful'] += 1
                stats['scores'].append(result.get('evaluation_score', 0.0))
                stats['confidences'].append(result.get('confidence', 0.0))
                stats['processing_times'].append(result.get('processing_time', 0.0))
            else:
                stats['failed'] += 1
        
        # Calculate averages
        aggregated = {}
        for category, stats in category_stats.items():
            scores = stats['scores']
            aggregated[category] = {
                'total_tasks': stats['total'],
                'successful_tasks': stats['successful'],
                'failed_tasks': stats['failed'],
                'success_rate': stats['successful'] / stats['total'] if stats['total'] > 0 else 0.0,
                'average_score': sum(scores) / len(scores) if scores else 0.0,
                'average_confidence': sum(stats['confidences']) / len(stats['confidences']) if stats['confidences'] else 0.0,
                'average_processing_time': sum(stats['processing_times']) / len(stats['processing_times']) if stats['processing_times'] else 0.0
            }
        
        return aggregated
    
    def _aggregate_by_method(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Aggregate statistics by evaluation method"""
        method_stats = defaultdict(lambda: {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'scores': []
        })
        
        for result in results:
            method = result.get('evaluation_method', 'unknown')
            stats = method_stats[method]
            stats['total'] += 1
            
            if result.get('status') == 'completed':
                stats['successful'] += 1
                stats['scores'].append(result.get('evaluation_score', 0.0))
            else:
                stats['failed'] += 1
        
        # Calculate averages
        aggregated = {}
        for method, stats in method_stats.items():
            scores = stats['scores']
            aggregated[method] = {
                'total_tasks': stats['total'],
                'successful_tasks': stats['successful'],
                'failed_tasks': stats['failed'],
                'success_rate': stats['successful'] / stats['total'] if stats['total'] > 0 else 0.0,
                'average_score': sum(scores) / len(scores) if scores else 0.0
            }
        
        return aggregated
    
    def _calculate_score_distribution(self, successful: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate score distribution"""
        distribution = {
            'excellent': 0,  # >= 0.8
            'good': 0,       # 0.6 - 0.8
            'fair': 0,       # 0.4 - 0.6
            'poor': 0        # < 0.4
        }
        
        for result in successful:
            score = result.get('evaluation_score', 0.0)
            if score >= 0.8:
                distribution['excellent'] += 1
            elif score >= 0.6:
                distribution['good'] += 1
            elif score >= 0.4:
                distribution['fair'] += 1
            else:
                distribution['poor'] += 1
        
        return distribution
    
    def export(self, summary: Dict[str, Any], output_path: str, format: str = 'json') -> str:
        """
        Export summary to file.
        
        Args:
            summary: Summary dictionary
            output_path: Output directory path
            format: Export format ('json', 'csv')
            
        Returns:
            Path to exported file
        """
        os.makedirs(output_path, exist_ok=True)
        
        if format == 'json':
            filepath = os.path.join(output_path, 'evaluation_summary.json')
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info(f'Summary exported to {filepath}')
            return filepath
        
        elif format == 'csv':
            filepath = os.path.join(output_path, 'evaluation_summary.csv')
            
            # Write overall statistics
            with open(filepath, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                overall = summary.get('overall', {})
                for key, value in overall.items():
                    writer.writerow([key, value])
                
                # Write category statistics
                writer.writerow([])
                writer.writerow(['Category Statistics'])
                writer.writerow(['Category', 'Total', 'Successful', 'Failed', 'Success Rate', 'Avg Score'])
                by_category = summary.get('by_category', {})
                for category, stats in by_category.items():
                    writer.writerow([
                        category,
                        stats.get('total_tasks', 0),
                        stats.get('successful_tasks', 0),
                        stats.get('failed_tasks', 0),
                        f"{stats.get('success_rate', 0.0):.2%}",
                        f"{stats.get('average_score', 0.0):.3f}"
                    ])
            
            logger.info(f'Summary exported to {filepath}')
            return filepath
        
        else:
            raise ValueError(f"Unsupported export format: {format}")

