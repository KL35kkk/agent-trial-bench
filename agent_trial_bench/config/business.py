"""
Business configuration for Agent Trial Bench
Business configs: LLM, Agent, Dataset, Evaluator, etc.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

# Import enums
from ..enums import TaskCategory, EvaluationMethod


@dataclass
class LLMConfig:
    """LLM configuration - business"""
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
    """Agent configuration - business"""
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
    """Dataset configuration - business"""
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
    """Calibration dataset configuration - business"""
    path: str
    method: str = "linear"
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
    """Evaluator configuration - business"""
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
class BusinessConfig:
    """Business configuration bundle"""
    llm_config: LLMConfig
    agent_config: AgentConfig
    dataset_config: DatasetConfig
    evaluator_config: Optional[EvaluatorConfig] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BusinessConfig':
        """Create from dictionary"""
        llm_config = LLMConfig.from_dict(data['llm_config'])
        agent_config = AgentConfig.from_dict(data['agent_config'])
        dataset_config = DatasetConfig.from_dict(data.get('dataset_config', {}))
        evaluator_config = None
        if 'evaluator_config' in data:
            evaluator_config = EvaluatorConfig.from_dict(data['evaluator_config'])
        
        return cls(
            llm_config=llm_config,
            agent_config=agent_config,
            dataset_config=dataset_config,
            evaluator_config=evaluator_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'llm_config': self.llm_config.to_dict(),
            'agent_config': self.agent_config.to_dict(),
            'dataset_config': self.dataset_config.to_dict(),
        }
        if self.evaluator_config:
            result['evaluator_config'] = self.evaluator_config.to_dict()
        return result
