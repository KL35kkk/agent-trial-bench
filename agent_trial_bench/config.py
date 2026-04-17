"""
Configuration system for Agent Trial Bench
"""

import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

# Import canonical enums; fall back to a minimal copy only when this module
# is imported before the rest of the package is ready.
try:
    from .enums import TaskCategory, EvaluationMethod
except ImportError:  # pragma: no cover – defensive fallback only
    from enum import Enum

    class TaskCategory(Enum):
        GENERAL = "general"
        FACT_QA = "fact_qa"
        RESEARCH = "research"
        REASONING = "reasoning"
        TOOL_USE = "tool_use"
        CODING = "coding"
        CONVERSATIONAL = "conversational"
        MULTI_STEP = "multi_step"
        WEB_RETRIEVAL = "web_retrieval"
        WEB_ONCHAIN_RETRIEVAL = "web_onchain_retrieval"
        ONCHAIN_RETRIEVAL = "onchain_retrieval"

    class EvaluationMethod(Enum):
        RULE_BASED = "rule_based"
        LLM_BASED = "llm_based"
        HYBRID = "hybrid"
        CASCADE = "cascade"


@dataclass
class LLMConfig:
    """LLM configuration"""
    model: str
    base_url: str
    api_key: str
    temperature: float = 0.3
    max_tokens: int = 2000
    enable_cache: bool = False
    cache_dir: str = "output/llm_cache"
    random_seed: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMConfig':
        """Create from dictionary"""
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class AgentConfig:
    """Agent configuration"""
    url: str
    capabilities: List[TaskCategory]
    timeout: int = 30
    close_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfig':
        """Create from dictionary"""
        # Convert capability strings to enums
        if 'capabilities' in data:
            caps = data['capabilities']
            if caps and isinstance(caps[0], str):
                data['capabilities'] = [TaskCategory(c) for c in caps]
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        # Convert enums to strings
        if 'capabilities' in result:
            result['capabilities'] = [c.value for c in result['capabilities']]
        return result


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    type: str = "csv"
    path: str = ""
    abbr: str = ""
    reader_cfg: Optional[Dict[str, Any]] = None
    ground_truth_path: Optional[str] = None
    question_id_field: str = "id"
    mock_response_field: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetConfig':
        """Create from dictionary"""
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class CalibrationConfig:
    """Calibration dataset configuration"""
    path: str
    method: str = "linear"  # linear or isotonic
    question_field: str = "question_id"
    score_field: str = "score"
    min_pairs: int = 5
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationConfig':
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluatorConfig:
    """Evaluator configuration"""
    type: str = "hybrid"
    rule_based_weight: float = 0.3
    llm_based_weight: float = 0.7
    llm_evaluation_threshold: float = 0.5
    use_llm_evaluation: bool = True
    cascade_config: Optional[Dict[str, Any]] = None
    calibration_config: Optional[CalibrationConfig] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluatorConfig':
        """Create from dictionary"""
        calibration_config = None
        if 'calibration_config' in data and data['calibration_config']:
            calibration_config = CalibrationConfig.from_dict(data['calibration_config'])
        data = dict(data)
        data['calibration_config'] = calibration_config
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        if self.calibration_config:
            result['calibration_config'] = self.calibration_config.to_dict()
        return result


@dataclass
class RunnerConfig:
    """Runner configuration"""
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
    """Storage and persistence configuration"""
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
class EvaluationConfig:
    """Complete evaluation configuration"""
    llm_config: LLMConfig
    agent_config: AgentConfig
    dataset_config: DatasetConfig
    evaluator_config: Optional[EvaluatorConfig] = None
    runner_config: Optional[RunnerConfig] = None
    storage_config: Optional[StorageConfig] = None
    work_dir: str = "output"
    max_tasks: Optional[int] = None
    reuse_results: Optional[str] = None  # "latest" or specific timestamp
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationConfig':
        """Create from dictionary"""
        llm_config = LLMConfig.from_dict(data['llm_config'])
        agent_config = AgentConfig.from_dict(data['agent_config'])
        dataset_config = DatasetConfig.from_dict(data.get('dataset_config', {}))
        evaluator_config = None
        if 'evaluator_config' in data:
            evaluator_config = EvaluatorConfig.from_dict(data['evaluator_config'])
        runner_config = None
        if 'runner_config' in data:
            runner_config = RunnerConfig.from_dict(data['runner_config'])
        storage_config = None
        if 'storage_config' in data:
            storage_config = StorageConfig.from_dict(data['storage_config'])
        
        return cls(
            llm_config=llm_config,
            agent_config=agent_config,
            dataset_config=dataset_config,
            evaluator_config=evaluator_config,
            runner_config=runner_config,
            storage_config=storage_config,
            work_dir=data.get('work_dir', 'output'),
            max_tasks=data.get('max_tasks'),
            reuse_results=data.get('reuse_results')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'llm_config': self.llm_config.to_dict(),
            'agent_config': self.agent_config.to_dict(),
            'dataset_config': self.dataset_config.to_dict(),
            'work_dir': self.work_dir,
        }
        if self.evaluator_config:
            result['evaluator_config'] = self.evaluator_config.to_dict()
        if self.runner_config:
            result['runner_config'] = self.runner_config.to_dict()
        if self.storage_config:
            result['storage_config'] = self.storage_config.to_dict()
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
    
    def save(self, filepath: str):
        """Save configuration to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def load_config(filepath: Optional[str] = None, **kwargs) -> EvaluationConfig:
    """
    Load configuration from file or create from kwargs
    
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
        llm_config = LLMConfig.from_dict(kwargs.get('llm_config', {}))
        agent_config = AgentConfig.from_dict(kwargs.get('agent_config', {}))
        dataset_config = DatasetConfig.from_dict(kwargs.get('dataset_config', {}))
        evaluator_config = None
        if 'evaluator_config' in kwargs:
            evaluator_config = EvaluatorConfig.from_dict(kwargs['evaluator_config'])
        runner_config = None
        if 'runner_config' in kwargs:
            runner_config = RunnerConfig.from_dict(kwargs['runner_config'])
        storage_config = None
        if 'storage_config' in kwargs:
            storage_config = StorageConfig.from_dict(kwargs['storage_config'])
        
        return EvaluationConfig(
            llm_config=llm_config,
            agent_config=agent_config,
            dataset_config=dataset_config,
            evaluator_config=evaluator_config,
            runner_config=runner_config,
            storage_config=storage_config,
            work_dir=kwargs.get('work_dir', 'output'),
            max_tasks=kwargs.get('max_tasks'),
            reuse_results=kwargs.get('reuse_results')
        )
