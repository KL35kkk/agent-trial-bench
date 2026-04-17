"""
Infrastructure configuration for Agent Trial Bench
Infrastructure configs: Runner, Storage, etc.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class RunnerConfig:
    """Runner configuration - infrastructure"""
    type: str = "local"
    max_workers: int = 4
    max_workers_per_gpu: int = 1
    timeout: int = 300
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunnerConfig':
        """Create from dictionary"""
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class StorageConfig:
    """Storage and persistence configuration - infrastructure"""
    enable_persistence: bool = True
    auto_save: bool = True
    save_interval: int = 10  # Save after every N tasks
    results_dir: str = "results"  # Relative to work_dir
    tasks_dir: str = "tasks"  # Relative to work_dir
    enable_versioning: bool = False
    max_versions: int = 10  # Maximum number of result versions to keep
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StorageConfig':
        """Create from dictionary"""
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class InfrastructureConfig:
    """Infrastructure configuration bundle"""
    runner_config: Optional[RunnerConfig] = None
    storage_config: Optional[StorageConfig] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InfrastructureConfig':
        """Create from dictionary"""
        runner_config = None
        if 'runner_config' in data:
            runner_config = RunnerConfig.from_dict(data['runner_config'])
        storage_config = None
        if 'storage_config' in data:
            storage_config = StorageConfig.from_dict(data['storage_config'])
        
        return cls(
            runner_config=runner_config,
            storage_config=storage_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {}
        if self.runner_config:
            result['runner_config'] = self.runner_config.to_dict()
        if self.storage_config:
            result['storage_config'] = self.storage_config.to_dict()
        return result

