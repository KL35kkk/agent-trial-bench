"""
Example Python configuration file for Agent Trial Eval
Demonstrates separated business and infrastructure configurations
"""

import os

# ============================================
# Business Configuration
# ============================================

# LLM Configuration
llm_config = {
    "model": os.environ.get("LLM_MODEL", "gpt-4"),
    "base_url": os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
    "api_key": os.environ.get("OPENAI_API_KEY", ""),
    "temperature": 0.3,
    "max_tokens": 2000
}

# Agent Configuration
agent_config = {
    "url": os.environ.get("AGENT_URL", "http://localhost:8002"),
    "capabilities": ["web_retrieval", "web_onchain_retrieval", "onchain_retrieval"],
    "timeout": 30,
    "close_endpoint": os.environ.get("AGENT_CLOSE_ENDPOINT", "http://localhost:8002/close")
}

# Dataset Configuration
dataset_config = {
    "type": "csv",
    "path": "data/benchmark.csv",
    "abbr": "benchmark",
    "reader_cfg": {
        "input_columns": ["question"],
        "output_column": "answer"
    }
}

# Evaluator Configuration
evaluator_config = {
    "type": "hybrid",
    "rule_based_weight": 0.3,
    "llm_based_weight": 0.7,
    "llm_evaluation_threshold": 0.5,
    "use_llm_evaluation": True
}

# Business Config Bundle
business_config = {
    "llm_config": llm_config,
    "agent_config": agent_config,
    "dataset_config": dataset_config,
    "evaluator_config": evaluator_config
}

# ============================================
# Infrastructure Configuration
# ============================================

# Runner Configuration
runner_config = {
    "type": "local",
    "max_workers": 4,
    "max_workers_per_gpu": 1,
    "timeout": 300
}

# Storage Configuration
storage_config = {
    "enable_persistence": True,
    "auto_save": True,
    "save_interval": 10,  # Save after every N tasks
    "results_dir": "results",  # Relative to work_dir
    "tasks_dir": "tasks",  # Relative to work_dir
    "enable_versioning": False,
    "max_versions": 10
}

# Infrastructure Config Bundle
infrastructure_config = {
    "runner_config": runner_config,
    "storage_config": storage_config
}

# ============================================
# Complete Configuration
# ============================================

config = {
    "business_config": business_config,
    "infrastructure_config": infrastructure_config,
    "work_dir": "output",
    "max_tasks": None,  # Set to a number to limit tasks, None for all
    "reuse_results": None  # "latest" or specific task_id/timestamp to reuse
}

# Alternative: Legacy flat format (still supported for backward compatibility)
config_legacy = {
    **business_config,
    **infrastructure_config,
    "work_dir": "output",
    "max_tasks": None,
    "reuse_results": None
}
