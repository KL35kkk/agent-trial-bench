"""
Data classes for Agent Trial Bench
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from .enums import TaskCategory, EvaluationMethod, EvaluationStatus


@dataclass
class AgentMetadata:
    """Agent metadata"""
    url: str
    capabilities: List[TaskCategory]
    timeout: int = 30
    close_endpoint: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class EvaluationTask:
    """Evaluation task"""
    task_id: str
    question: str
    agent_metadata: Any
    context: Dict[str, Any]
    category: TaskCategory
    evaluation_method: EvaluationMethod
    expected_answer: Optional[str] = None
    status: EvaluationStatus = EvaluationStatus.PENDING
    error: Optional[str] = None
    created_at: Optional[float] = None
    completed_at: Optional[float] = None
    agent_response: Optional[str] = None
    evaluation_result: Optional[Any] = None


@dataclass
class EvaluationResult:
    """Evaluation result"""
    task_id: str
    question: str
    agent_response: str
    evaluation_score: float
    evaluation_reasoning: str
    confidence: float
    processing_time: float
    tools_used: List[str]
    metadata: Dict[str, Any]
    status: EvaluationStatus
    error: Optional[str] = None

