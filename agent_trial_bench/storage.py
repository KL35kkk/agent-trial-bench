"""
Storage and persistence system for Agent Trial Bench
"""

import os
import json
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ResultStorage:
    """Result storage for persisting evaluation results and tasks.
    
    Features:
    - Save/load individual task results
    - Save/load complete evaluation results
    - Support for result versioning
    - Find existing results by task configuration
    """
    
    def __init__(self, work_dir: str, storage_config: Optional[Dict[str, Any]] = None):
        """
        Initialize result storage.
        
        Args:
            work_dir: Base working directory
            storage_config: Storage configuration dict
        """
        self.work_dir = work_dir
        self.config = storage_config or {}
        
        # Setup directories
        self.results_dir = os.path.join(work_dir, self.config.get('results_dir', 'results'))
        self.tasks_dir = os.path.join(work_dir, self.config.get('tasks_dir', 'tasks'))
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.tasks_dir, exist_ok=True)
        history_filename = self.config.get('history_filename', 'evaluation_history.jsonl')
        self.history_file = os.path.join(self.results_dir, history_filename)
        
        self.enable_versioning = self.config.get('enable_versioning', False)
        self.max_versions = self.config.get('max_versions', 10)
    
    def _get_task_filepath(self, task_id: str) -> str:
        """Get filepath for a task"""
        return os.path.join(self.tasks_dir, f'{task_id}.json')
    
    def _get_result_filepath(self, task_id: str) -> str:
        """Get filepath for a result"""
        return os.path.join(self.results_dir, f'{task_id}.json')
    
    def _generate_config_hash(self, config: Dict[str, Any]) -> str:
        """Generate hash for task configuration"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def save_task(self, task_id: str, task_data: Dict[str, Any]):
        """Save task data"""
        if not self.config.get('enable_persistence', True):
            return
        
        filepath = self._get_task_filepath(task_id)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(task_data, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved task {task_id} to {filepath}")
    
    def load_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Load task data"""
        filepath = self._get_task_filepath(task_id)
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load task {task_id}: {e}")
            return None
    
    def save_result(self, task_id: str, result: Dict[str, Any]):
        """Save evaluation result"""
        if not self.config.get('enable_persistence', True):
            return
        
        filepath = self._get_result_filepath(task_id)
        
        # Add metadata
        result_with_meta = {
            'task_id': task_id,
            'saved_at': datetime.now().isoformat(),
            **result
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_with_meta, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved result {task_id} to {filepath}")
    
    def load_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Load evaluation result"""
        filepath = self._get_result_filepath(task_id)
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load result {task_id}: {e}")
            return None
    
    def find_existing_results(self, task_config: Dict[str, Any]) -> List[str]:
        """Find existing results matching task configuration"""
        config_hash = self._generate_config_hash(task_config)
        
        # Search for results with matching config hash
        existing = []
        if os.path.exists(self.results_dir):
            for filename in os.listdir(self.results_dir):
                if not filename.endswith('.json'):
                    continue
                
                filepath = os.path.join(self.results_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        if result.get('config_hash') == config_hash:
                            existing.append(result.get('task_id', filename.replace('.json', '')))
                except Exception:
                    continue
        
        return existing
    
    def list_all_tasks(self) -> List[str]:
        """List all saved task IDs"""
        if not os.path.exists(self.tasks_dir):
            return []
        
        tasks = []
        for filename in os.listdir(self.tasks_dir):
            if filename.endswith('.json'):
                tasks.append(filename.replace('.json', ''))
        return tasks
    
    def list_all_results(self) -> List[str]:
        """List all saved result IDs"""
        if not os.path.exists(self.results_dir):
            return []
        
        results = []
        for filename in os.listdir(self.results_dir):
            if filename.endswith('.json'):
                results.append(filename.replace('.json', ''))
        return results
    
    def find_incomplete_tasks(self) -> List[str]:
        """Find tasks that are not completed"""
        incomplete = []
        
        for task_id in self.list_all_tasks():
            task = self.load_task(task_id)
            if task and task.get('status') not in ['completed', 'failed']:
                incomplete.append(task_id)
        
        return incomplete
    
    def get_latest_results(self) -> Optional[str]:
        """Get the latest result timestamp/directory"""
        if not os.path.exists(self.results_dir):
            return None
        
        # Find the most recently modified result file
        latest_time = 0
        latest_task_id = None
        
        for filename in os.listdir(self.results_dir):
            if not filename.endswith('.json'):
                continue
            
            filepath = os.path.join(self.results_dir, filename)
            mtime = os.path.getmtime(filepath)
            if mtime > latest_time:
                latest_time = mtime
                latest_task_id = filename.replace('.json', '')
        
        return latest_task_id
    
    def cleanup_old_versions(self):
        """Clean up old result versions if versioning is enabled"""
        if not self.enable_versioning:
            return
        
        # Implementation for version cleanup
        # This would track versions and remove old ones
        pass
    
    def append_history_entry(self, entry: Dict[str, Any]):
        """Append a single evaluation snapshot to the history log."""
        if not self.config.get('enable_persistence', True):
            return
        try:
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, 'a', encoding='utf-8') as f:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        except Exception as exc:
            logger.warning(f"Failed to append history entry: {exc}")
