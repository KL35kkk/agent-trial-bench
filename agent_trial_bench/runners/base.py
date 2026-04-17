"""
Base Runner for Agent Trial Bench
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class BaseRunner(ABC):
    """Base class for all runners. A runner is responsible for launching multiple tasks.
    
    Args:
        config: Runner configuration
        debug: Whether to run in debug mode
    """
    
    def __init__(self, config: Dict[str, Any] = None, debug: bool = False):
        self.config = config or {}
        self.debug = debug
        self.runner_type = self.__class__.__name__
    
    @abstractmethod
    async def launch(self, tasks: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """
        Launch multiple tasks.
        
        Args:
            tasks: A list of task configs
            
        Returns:
            A list of (task_id, exit_code) tuples
        """
        pass
    
    def summarize(self, status: List[Tuple[str, int]]) -> None:
        """
        Summarize the results of the tasks.
        
        Args:
            status: A list of (task_id, exit_code) tuples
        """
        failed_tasks = []
        for task_id, code in status:
            if code != 0:
                logger.error(f'{task_id} failed with exit code {code}')
                failed_tasks.append(task_id)
        
        total = len(status)
        succeeded = total - len(failed_tasks)
        
        logger.info(f'Runner {self.runner_type} completed: {succeeded}/{total} tasks succeeded')
        if failed_tasks:
            logger.warning(f'Failed tasks: {failed_tasks}')
    
    async def __call__(self, tasks: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """Launch tasks and return status"""
        status = await self.launch(tasks)
        self.summarize(status)
        return status

