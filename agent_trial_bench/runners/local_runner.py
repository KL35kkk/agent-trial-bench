"""
Local Runner for Agent Trial Bench
"""

import asyncio
from typing import List, Dict, Any, Tuple
import logging

from .base import BaseRunner

logger = logging.getLogger(__name__)


class LocalRunner(BaseRunner):
    """Local runner that executes tasks in parallel using asyncio.
    
    Args:
        config: Runner configuration
            - max_workers: Maximum number of concurrent tasks (default: 4)
            - timeout: Task timeout in seconds (default: 300)
        debug: Whether to run in debug mode
    """
    
    def __init__(self, config: Dict[str, Any] = None, debug: bool = False):
        super().__init__(config, debug)
        self.max_workers = self.config.get('max_workers', 4)
        self.timeout = self.config.get('timeout', 300)
    
    async def _execute_task(self, task: Dict[str, Any]) -> Tuple[str, int]:
        """
        Execute a single task.
        
        Args:
            task: Task configuration dict
            
        Returns:
            (task_id, exit_code) tuple
        """
        task_id = task.get('task_id', 'unknown')
        task_func = task.get('func')
        
        if not task_func:
            logger.error(f'Task {task_id} has no function to execute')
            return (task_id, 1)
        
        try:
            if asyncio.iscoroutinefunction(task_func):
                await asyncio.wait_for(task_func(**task.get('kwargs', {})), timeout=self.timeout)
            else:
                # For sync functions, run in executor
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: task_func(**task.get('kwargs', {}))),
                    timeout=self.timeout
                )
            return (task_id, 0)
        except asyncio.TimeoutError:
            logger.error(f'Task {task_id} timed out after {self.timeout}s')
            return (task_id, 124)  # Timeout exit code
        except Exception as e:
            logger.error(f'Task {task_id} failed with error: {e}')
            return (task_id, 1)
    
    async def launch(self, tasks: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """
        Launch multiple tasks in parallel.
        
        Args:
            tasks: A list of task configs
            
        Returns:
            A list of (task_id, exit_code) tuples
        """
        if not tasks:
            logger.warning('No tasks to execute')
            return []
        
        logger.info(f'Launching {len(tasks)} tasks with max_workers={self.max_workers}')
        
        # Create semaphore to limit concurrent tasks
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def execute_with_semaphore(task):
            async with semaphore:
                return await self._execute_task(task)
        
        # Execute all tasks
        results = await asyncio.gather(*[execute_with_semaphore(task) for task in tasks])
        
        return results

