"""
Agent Trial Bench – Domain-Agnostic Benchmark for AI Agents.

A comprehensive, domain-pluggable evaluation framework for any kind of LLM
agent (research, reasoning, coding, tool-use, Web3 …) with support for:

- Multi-dimensional evaluation (factual accuracy, completeness, precision,
  relevance, conciseness, tool usage, trajectory quality)
- Rule-based, LLM-as-judge, hybrid and cascade evaluators
- Process-aware grading: Trajectory / Outcome / TrackedMetrics / Trials
- Pluggable ``DomainPlugin`` system – ``general`` and ``web3`` ship out of the
  box; add your own vertical without modifying the core
- Configurable LLM backends (OpenAI, Anthropic, Google, any OpenAI-compatible)
- Export results in JSON, CSV, and other formats
"""

__version__ = "2.1.0"
__author__ = "Agent Trial Bench Team"
__email__ = "dab@example.com"

# Import enums and dataclasses
from .enums import TaskCategory, EvaluationMethod, EvaluationStatus
from .dataclasses import AgentMetadata, EvaluationTask, EvaluationResult

# Import main classes and functions
from .evaluator import (
    AgentTrialBench,
    evaluate_agent,
)

# Import configuration system
from .config import (
    EvaluationConfig,
    LLMConfig,
    AgentConfig,
    DatasetConfig,
    EvaluatorConfig,
    RunnerConfig,
    StorageConfig,
    BusinessConfig,
    InfrastructureConfig,
    load_config,
)

# Import evaluation modules
from .evaluation import (
    BaseEvaluator,
    LLMEvaluator,
    HybridEvaluator,
)

# Import runners and summarizers
from .runners import BaseRunner, LocalRunner, MultiTrialRunner, MultiTrialResult
from .summarizers import BaseSummarizer, DefaultSummarizer

# Import v2 data structures
from .trajectory import (
    Step,
    StepType,
    Trajectory,
    Outcome,
    TrackedMetrics,
    GradeResult,
    Trial,
)

# Import datasets (optional – module may not be shipped in all builds)
try:
    from .datasets import TaskHub, TaskTier, HubTask  # type: ignore
except ImportError:  # pragma: no cover
    TaskHub = TaskTier = HubTask = None  # type: ignore

# Import tool mocking
from .tools import (
    MockToolRegistry,
    ToolFailureMode,
    register_default_tools,
    register_general_tools,
    register_web3_tools,
)

# Import scenario generator
from .scenario_generator import ScenarioGenerator, ScenarioTask, Perturbation

# Import CI gate
from .ci_gate import CIGate, GateConfig, GateResult

# Import analyzer
from .analyzer import (
    FailureCategory,
    FailureClassifier,
    ClassifiedFailure,
    RegressionDetector,
    RegressionReport,
    TaskSnapshot,
    RootCauseAnalyzer,
    RootCauseReport,
)

# Import graders
from .graders import (
    BaseGrader,
    DeterministicTestsGrader,
    LLMRubricGrader,
    StateCheckGrader,
    ToolCallsGrader,
    build_grader,
)

# Import domain plugin system
from .domains import (
    DomainPlugin,
    GeneralDomain,
    Web3Domain,
    register_domain,
    get_domain,
    list_domains,
)

# Import storage and task management
from .storage import ResultStorage
from .task_manager import TaskManager

__all__ = [
    # Main classes
    "AgentTrialBench",
    "AgentMetadata", 
    "EvaluationTask",
    "EvaluationResult",
    
    # Configuration
    "EvaluationConfig",
    "LLMConfig",
    "AgentConfig",
    "DatasetConfig",
    "EvaluatorConfig",
    "RunnerConfig",
    "StorageConfig",
    "BusinessConfig",
    "InfrastructureConfig",
    "load_config",
    
    # Storage and Task Management
    "ResultStorage",
    "TaskManager",
    
    # Enums
    "TaskCategory",
    "EvaluationMethod", 
    "EvaluationStatus",
    
    # Convenience functions (deprecated)
    "evaluate_agent",
    
    # Evaluation modules
    "BaseEvaluator",
    "LLMEvaluator", 
    "HybridEvaluator",
    
    # Runners
    "BaseRunner",
    "LocalRunner",
    "MultiTrialRunner",
    "MultiTrialResult",

    # v2 trajectory / trial data structures
    "Step",
    "StepType",
    "Trajectory",
    "Outcome",
    "TrackedMetrics",
    "GradeResult",
    "Trial",

    # Graders
    "BaseGrader",
    "DeterministicTestsGrader",
    "LLMRubricGrader",
    "StateCheckGrader",
    "ToolCallsGrader",
    "build_grader",

    # Domains
    "DomainPlugin",
    "GeneralDomain",
    "Web3Domain",
    "register_domain",
    "get_domain",
    "list_domains",

    # Datasets
    "TaskHub",
    "TaskTier",
    "HubTask",

    # Tool mocking
    "MockToolRegistry",
    "ToolFailureMode",
    "register_default_tools",

    # Scenario generator
    "ScenarioGenerator",
    "ScenarioTask",
    "Perturbation",

    # CI Gate
    "CIGate",
    "GateConfig",
    "GateResult",

    # Analyzer
    "FailureCategory",
    "FailureClassifier",
    "ClassifiedFailure",
    "RegressionDetector",
    "RegressionReport",
    "TaskSnapshot",
    "RootCauseAnalyzer",
    "RootCauseReport",
    
    # Summarizers
    "BaseSummarizer",
    "DefaultSummarizer",
    
    # Package info
    "__version__",
    "__author__",
    "__email__",
]

def main():
    """Main entry point for command line usage"""
    import asyncio
    from .examples.basic_usage import main as example_main
    asyncio.run(example_main())

if __name__ == "__main__":
    main()
