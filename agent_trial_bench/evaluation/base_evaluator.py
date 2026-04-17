"""
Base Evaluator for Agent Trial Bench
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseEvaluator(ABC):
    """Base evaluator class"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.evaluator_type = self.__class__.__name__
    
    @abstractmethod
    async def evaluate(self, 
                      question: str,
                      agent_response: str,
                      expected_answer: Optional[str] = None,
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate Agent response
        
        Args:
            question: Evaluation question
            agent_response: Agent response
            expected_answer: Expected answer
            context: Context information
            
        Returns:
            Dict containing evaluation results
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get evaluator capabilities"""
        pass