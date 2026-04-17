"""
Task Manager for Agent Trial Bench
Handles task lifecycle, state management, and persistence
"""

import os
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from .enums import EvaluationStatus
from .dataclasses import EvaluationTask, EvaluationResult
from .storage import ResultStorage

logger = logging.getLogger(__name__)


class TaskManager:
    """Task manager for handling task lifecycle and state.
    
    Features:
    - Task creation and tracking
    - Task state management
    - Task persistence
    - Task recovery and resume
    """
    
    def __init__(self, storage: ResultStorage):
        """
        Initialize task manager.
        
        Args:
            storage: ResultStorage instance for persistence
        """
        self.storage = storage
        self.tasks: Dict[str, EvaluationTask] = {}
        self._task_counter = 0
    
    def _generate_task_id(self) -> str:
        """Generate unique task ID"""
        self._task_counter += 1
        timestamp = int(time.time())
        return f"task_{self._task_counter}_{timestamp}"
    
    def create_task(self, 
                   question: str,
                   agent_metadata: Any,
                   context: Dict[str, Any],
                   category: Any,
                   evaluation_method: Any,
                   expected_answer: Optional[str] = None) -> str:
        """
        Create a new task and return task_id.
        
        Args:
            question: Evaluation question
            agent_metadata: Agent metadata
            context: Context information
            category: Task category
            evaluation_method: Evaluation method
            expected_answer: Expected answer (optional)
            
        Returns:
            task_id: Unique task identifier
        """
        task_id = self._generate_task_id()
        
        task = EvaluationTask(
            task_id=task_id,
            question=question,
            agent_metadata=agent_metadata,
            context=context,
            category=category,
            evaluation_method=evaluation_method,
            expected_answer=expected_answer,
            status=EvaluationStatus.PENDING,
            created_at=time.time()
        )
        
        self.tasks[task_id] = task
        
        # Persist task
        self.save_task(task)
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[EvaluationTask]:
        """Get task by ID"""
        # First check in-memory cache
        if task_id in self.tasks:
            return self.tasks[task_id]
        
        # Try to load from storage
        task_data = self.storage.load_task(task_id)
        if task_data:
            # Reconstruct task from saved data
            # Note: This is a simplified version, full reconstruction would need
            # to handle AgentMetadata and enum types properly
            return None  # For now, return None if not in memory
        
        return None
    
    def update_task_status(self, task_id: str, status: EvaluationStatus, 
                          error: Optional[str] = None):
        """Update task status"""
        task = self.get_task(task_id)
        if not task:
            logger.warning(f"Task {task_id} not found")
            return
        
        task.status = status
        if error:
            task.error = error
        task.completed_at = time.time() if status in [EvaluationStatus.COMPLETED, EvaluationStatus.FAILED] else None
        
        self.save_task(task)
    
    def save_task(self, task: EvaluationTask):
        """Save task to storage"""
        task_data = {
            'task_id': task.task_id,
            'question': task.question,
            'category': task.category.value if hasattr(task.category, 'value') else str(task.category),
            'evaluation_method': task.evaluation_method.value if hasattr(task.evaluation_method, 'value') else str(task.evaluation_method),
            'expected_answer': task.expected_answer,
            'context': task.context,
            'status': task.status.value if hasattr(task.status, 'value') else str(task.status),
            'error': task.error,
            'created_at': task.created_at,
            'completed_at': task.completed_at,
            'agent_response': task.agent_response,
            'evaluation_result': task.evaluation_result
        }
        
        self.storage.save_task(task.task_id, task_data)
    
    def save_result(self, task_id: str, result: EvaluationResult):
        """Save evaluation result"""
        result_data = {
            'task_id': result.task_id,
            'question': result.question,
            'agent_response': result.agent_response,
            'evaluation_score': result.evaluation_score,
            'evaluation_reasoning': result.evaluation_reasoning,
            'confidence': result.confidence,
            'processing_time': result.processing_time,
            'tools_used': result.tools_used,
            'metadata': result.metadata,
            'status': result.status.value if hasattr(result.status, 'value') else str(result.status),
            'error': result.error
        }
        
        self.storage.save_result(task_id, result_data)
    
    def find_incomplete_tasks(self) -> List[str]:
        """Find all incomplete tasks"""
        incomplete = []
        
        # Check in-memory tasks
        for task_id, task in self.tasks.items():
            if task.status not in [EvaluationStatus.COMPLETED, EvaluationStatus.FAILED]:
                incomplete.append(task_id)
        
        # Check persisted tasks
        persisted_incomplete = self.storage.find_incomplete_tasks()
        for task_id in persisted_incomplete:
            if task_id not in incomplete:
                incomplete.append(task_id)
        
        return incomplete
    
    def list_tasks(self, status: Optional[EvaluationStatus] = None) -> List[EvaluationTask]:
        """List all tasks, optionally filtered by status"""
        tasks = list(self.tasks.values())
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        return tasks
    
    def get_task_status(self, task_id: str) -> Optional[EvaluationStatus]:
        """Get task status"""
        task = self.get_task(task_id)
        if task:
            return task.status
        return None
    
    def resume_tasks(self, task_ids: Optional[List[str]] = None) -> List[str]:
        """
        Resume incomplete tasks.
        
        Args:
            task_ids: List of task IDs to resume (None = all incomplete)
            
        Returns:
            List of task IDs that need to be resumed
        """
        if task_ids is None:
            task_ids = self.find_incomplete_tasks()
        
        return task_ids
    
    def load_results(self, task_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Load results for tasks.
        
        Args:
            task_ids: List of task IDs (None = all)
            
        Returns:
            List of result dictionaries
        """
        if task_ids is None:
            task_ids = self.storage.list_all_results()
        
        results = []
        for task_id in task_ids:
            result = self.storage.load_result(task_id)
            if result:
                results.append(result)
        
        return results

