"""
Evaluation configuration that combines business and infrastructure configs
"""

import os
import json
from typing import Dict, Any, Optional

from .business import BusinessConfig
from .infrastructure import InfrastructureConfig


class EvaluationConfig:
    """Complete evaluation configuration combining business and infrastructure configs.
    
    This class combines:
    - BusinessConfig: LLM, Agent, Dataset, Evaluator configurations
    - InfrastructureConfig: Runner, Storage configurations
    - Global settings: work_dir, max_tasks, reuse_results
    """
    
    def __init__(self,
                 business_config: BusinessConfig,
                 infrastructure_config: Optional[InfrastructureConfig] = None,
                 work_dir: str = "output",
                 max_tasks: Optional[int] = None,
                 reuse_results: Optional[str] = None):
        """
        Initialize evaluation configuration.
        
        Args:
            business_config: Business configuration (LLM, Agent, Dataset, Evaluator)
            infrastructure_config: Infrastructure configuration (Runner, Storage)
            work_dir: Working directory for outputs
            max_tasks: Maximum number of tasks to evaluate
            reuse_results: Reuse mode ("latest" or specific task_id/timestamp)
        """
        self.business_config = business_config
        self.infrastructure_config = infrastructure_config or InfrastructureConfig()
        self.work_dir = work_dir
        self.max_tasks = max_tasks
        self.reuse_results = reuse_results
    
    # Convenience properties for backward compatibility
    @property
    def llm_config(self):
        """Get LLM config"""
        return self.business_config.llm_config
    
    @property
    def agent_config(self):
        """Get Agent config"""
        return self.business_config.agent_config
    
    @property
    def dataset_config(self):
        """Get Dataset config"""
        return self.business_config.dataset_config
    
    @property
    def evaluator_config(self):
        """Get Evaluator config"""
        return self.business_config.evaluator_config
    
    @property
    def runner_config(self):
        """Get Runner config"""
        return self.infrastructure_config.runner_config
    
    @property
    def storage_config(self):
        """Get Storage config"""
        return self.infrastructure_config.storage_config
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationConfig':
        """Create from dictionary (supports both old and new format)"""
        # Check if using new separated format
        if 'business_config' in data and 'infrastructure_config' in data:
            business_config = BusinessConfig.from_dict(data['business_config'])
            infrastructure_config = InfrastructureConfig.from_dict(data.get('infrastructure_config', {}))
        else:
            # Old format - extract business and infrastructure configs
            business_config = BusinessConfig.from_dict(data)
            infrastructure_config = InfrastructureConfig.from_dict(data)
        
        return cls(
            business_config=business_config,
            infrastructure_config=infrastructure_config,
            work_dir=data.get('work_dir', 'output'),
            max_tasks=data.get('max_tasks'),
            reuse_results=data.get('reuse_results')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (new separated format)"""
        result = {
            'business_config': self.business_config.to_dict(),
            'infrastructure_config': self.infrastructure_config.to_dict(),
            'work_dir': self.work_dir,
        }
        if self.max_tasks:
            result['max_tasks'] = self.max_tasks
        if self.reuse_results:
            result['reuse_results'] = self.reuse_results
        return result
    
    def to_dict_legacy(self) -> Dict[str, Any]:
        """Convert to legacy flat dictionary format (for backward compatibility)"""
        result = self.business_config.to_dict()
        result.update(self.infrastructure_config.to_dict())
        result['work_dir'] = self.work_dir
        if self.max_tasks:
            result['max_tasks'] = self.max_tasks
        if self.reuse_results:
            result['reuse_results'] = self.reuse_results
        return result
    
    @classmethod
    def from_file(cls, filepath: str) -> 'EvaluationConfig':
        """Load configuration from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            if filepath.endswith('.json'):
                data = json.load(f)
            elif filepath.endswith('.py'):
                # Support Python config files
                import importlib.util
                spec = importlib.util.spec_from_file_location("config", filepath)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                data = module.config if hasattr(module, 'config') else {}
            else:
                raise ValueError(f"Unsupported config file format: {filepath}")
        
        return cls.from_dict(data)
    
    def save(self, filepath: str, legacy_format: bool = False):
        """Save configuration to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            if legacy_format:
                json.dump(self.to_dict_legacy(), f, indent=2, ensure_ascii=False)
            else:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def load_config(filepath: Optional[str] = None, **kwargs) -> EvaluationConfig:
    """
    Load configuration from file or create from kwargs.
    
    Args:
        filepath: Path to config file (optional)
        **kwargs: Configuration parameters (used if filepath is None)
    
    Returns:
        EvaluationConfig instance
    """
    if filepath:
        return EvaluationConfig.from_file(filepath)
    else:
        # Create from kwargs
        from .business import BusinessConfig, LLMConfig, AgentConfig, DatasetConfig, EvaluatorConfig
        from .infrastructure import InfrastructureConfig, RunnerConfig, StorageConfig
        
        business_config = BusinessConfig(
            llm_config=LLMConfig.from_dict(kwargs.get('llm_config', {})),
            agent_config=AgentConfig.from_dict(kwargs.get('agent_config', {})),
            dataset_config=DatasetConfig.from_dict(kwargs.get('dataset_config', {})),
            evaluator_config=EvaluatorConfig.from_dict(kwargs['evaluator_config']) if 'evaluator_config' in kwargs else None
        )
        
        infrastructure_config = InfrastructureConfig(
            runner_config=RunnerConfig.from_dict(kwargs['runner_config']) if 'runner_config' in kwargs else None,
            storage_config=StorageConfig.from_dict(kwargs['storage_config']) if 'storage_config' in kwargs else None
        )
        
        return EvaluationConfig(
            business_config=business_config,
            infrastructure_config=infrastructure_config,
            work_dir=kwargs.get('work_dir', 'output'),
            max_tasks=kwargs.get('max_tasks'),
            reuse_results=kwargs.get('reuse_results')
        )

