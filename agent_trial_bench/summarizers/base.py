"""
Base Summarizer for Agent Trial Bench
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseSummarizer(ABC):
    """Base class for result summarizers.
    
    Args:
        config: Summarizer configuration
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.summarizer_type = self.__class__.__name__
    
    @abstractmethod
    def summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Summarize evaluation results.
        
        Args:
            results: List of evaluation result dictionaries
            
        Returns:
            Summary dictionary
        """
        pass
    
    @abstractmethod
    def export(self, summary: Dict[str, Any], output_path: str, format: str = 'json') -> str:
        """
        Export summary to file.
        
        Args:
            summary: Summary dictionary
            output_path: Output directory path
            format: Export format ('json', 'csv', 'html')
            
        Returns:
            Path to exported file
        """
        pass

