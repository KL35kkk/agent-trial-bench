# Agent Trial Bench

**Agent Trial Bench** (`agent_trial_bench`, CLI `atb`) is a **production-grade, domain-agnostic evaluation pipeline for AI agents**.  It goes beyond "did the answer look right?" – it's a full evaluation infrastructure that continuously controls agent quality, cost, and risk across iterative development cycles.

The core is **fully domain-agnostic**: the same evaluators, graders and CI gate work for research agents, reasoning agents, coding agents, tool-using agents, customer-support agents, and so on.  Vertical-specific knowledge (format validators, trusted sources, expected tools, scoring vocabulary) lives entirely in pluggable **`DomainPlugin`** components.  Two plugins ship out of the box:

- `general` – URLs, emails, ISO dates, UUIDs, JSON, common research/coding/reasoning tool patterns, general-purpose trusted sources (Wikipedia, arXiv, Reuters, GitHub, …)
- `web3`    – Ethereum addresses, tx hashes, ENS; `onchain_retrieval` / `defi_analysis` tool patterns; Etherscan / DefiLlama / Dune trusted sources

Add a new vertical (medical, legal, e-commerce, robotics …) without forking the core – just register a `DomainPlugin`.

## What This Is (And What It Isn't)

| Old thinking | Production thinking |
|---|---|
| "Test good or not" | Stabilise quality, cost, and risk during continuous iteration |
| Score a single answer | Evaluate trajectory + outcome across N trials |
| Run once manually | Regression detection + CI/CD gate |
| Black-box agent call | Full trace: thought → tool call → observation → state |
| "Built for X vertical" | Domain-pluggable: `general` + `web3` + your own |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Domain Plugins                        │
│    general (URLs, research, reasoning, coding…)          │
│    web3    (address, tx_hash, etherscan, defi…)          │
│    your_vertical (custom validators / tools / sources)   │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                 Task Dataset Hub                         │
│         Gold (human) / Synthetic / Production            │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                 Scenario Generator                       │
│    tool_failure · missing_info · conflicting ·           │
│    price_change · user_redirect                          │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│              Agent Runner  (Multi-Trial)                 │
│      Trajectory · Outcome · TrackedMetrics               │
└───────────────┬──────────────────┬───────────────────────┘
                │                  │
┌───────────────▼──────┐  ┌────────▼────────────────────┐
│     4 Graders        │  │     Metrics System           │
│  deterministic_tests │  │  quality · efficiency        │
│  llm_rubric          │  │  cost    · stability         │
│  state_check         │  │  robustness                  │
│  tool_calls          │  └────────┬────────────────────┘
└───────────────┬──────┘           │
                └────────┬─────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    Analyzer                              │
│   Failure Clustering · Regression Diff                   │
│   Root Cause Attribution                                 │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                   CI / CD Gate                           │
│   success_rate ≥ threshold · cost Δ ≤ budget             │
│   no new failure modes  →  Pass / Fail                   │
└─────────────────────────────────────────────────────────┘
```

---

## Domain Plugin System

Every piece of vertical-specific knowledge – format validators, trusted-source allowlists, expected tool patterns, scoring calibrations, technical vocabulary – lives in a `DomainPlugin`.  Multiple plugins can be active at the same time; their knowledge is merged on lookup so a single grader instance works across all of them.

Built-in plugins:

| Plugin | Provides |
|---|---|
| `general` | `url` / `email` / `iso_date` / `uuid` / `integer` / `float` / `phone` / `json` validators · tool patterns for `fact_qa`, `research`, `reasoning`, `coding`, `tool_use` · trusted sources (Wikipedia, arXiv, Reuters, GitHub, …) |
| `web3` | `address` / `tx_hash` / `ens` / `bytes32` validators · tool patterns for `onchain_retrieval`, `defi_analysis`, `web_onchain_retrieval` · trusted sources (Etherscan family, DefiLlama, Dune, Messari, …) · hex-address normalisation |

Register your own vertical without modifying the core:

```python
from agent_trial_bench import DomainPlugin, register_domain

class MedicalDomain(DomainPlugin):
    name = "medical"
    validators = {"icd10": lambda s: bool(ICD10_RE.match(s))}
    trusted_sources = {"pubmed.ncbi.nlm.nih.gov", "who.int", "cdc.gov"}
    tool_patterns = {
        "clinical_qa": {
            "expected_tools": {"pubmed_search", "drug_lookup"},
            "required_params": ["query"],
        }
    }
    technical_terms = {"diagnosis", "etiology", "contraindication"}

register_domain(MedicalDomain())
```

All four graders and the scoring system will now transparently understand your domain.

---

## Modules

### Task Dataset Hub

Three-tier task library that prevents evaluation distortion:

| Tier | Source | Purpose |
|---|---|---|
| **Gold** | Human-annotated | Regression / quality gate |
| **Synthetic** | LLM-generated variants | Coverage expansion |
| **Production** | Live traffic replay | Real-world fidelity |

```python
from agent_trial_bench import TaskHub, TaskTier, HubTask

hub = TaskHub(seed=42)
hub.add(HubTask(task_id="q1", question="...", expected_answer="...",
                category="fact_qa", tier=TaskTier.GOLD))

# filtered sampling
batch = hub.sample(100, category_filter="research")

for task in hub.iter_tier(TaskTier.GOLD):
    ...
```

> Full CSV / DMind loaders are on the roadmap; the built-in `TaskHub` currently provides basic add/sample/iterate over in-memory tasks.

### Scenario Generator

Tests not just *capability* but *resilience under chaos*:

```python
from agent_trial_bench import ScenarioGenerator, Perturbation

gen = ScenarioGenerator(seed=42)

scenario = gen.generate(task, [
    Perturbation.TOOL_FAILURE,   # API timeout / HTTP error / rate limit
    Perturbation.MISSING_INFO,   # Remove a context key
    Perturbation.CONFLICTING,    # Inject a contradictory hint
    Perturbation.PRICE_CHANGE,   # Data changes mid-run
    Perturbation.USER_REDIRECT,  # User changes their question
])

scenarios = gen.generate_all_combinations(task, max_combinations=10)
```

The tool-failure perturbation consults the domain registry so it injects failures into tools that are actually relevant for the task's category.

### Agent Runner + Multi-Trial

Every task runs N times to handle non-determinism:

```python
from agent_trial_bench import MultiTrialRunner

runner = MultiTrialRunner(run_fn, num_trials=3, variance_threshold=0.15)
result = await runner.run(task)

print(result.mean_score)       # 0.87
print(result.variance)         # 0.02
print(result.worst_case_score) # 0.83
print(result.pass_at_k)        # {1: 0.83, 2: 0.97, 3: 1.0}
print(result.is_unstable)      # False
```

### 4 Graders (run in parallel per trial)

| Grader | Type | Evaluates |
|---|---|---|
| `DeterministicTestsGrader` | Code-based | Outcome: exact match, regex, MC options, domain-aware format checks (`url`, `email`, `iso_date`, `address`, `tx_hash`, …) |
| `LLMRubricGrader` | Model-based | Trajectory + Outcome: rubric scoring |
| `StateCheckGrader` | Code-based | Outcome: trusted-source reliability, state ↔ answer consistency, declarative `state_format_checks` |
| `ToolCallsGrader` | Code-based | Trajectory: success rate, tool-selection relevance, redundancy, loop detection, parameter well-formedness |

Which formats / tools / trusted sources apply is driven entirely by the registered `DomainPlugin`s plus per-task `context` fields – the graders themselves contain **no vertical-specific code**.

```python
from agent_trial_bench import build_grader

grader = build_grader("tool_calls", config={"max_tool_calls": 10})
result = await grader.grade(outcome, trajectory, task)

print(result.score)      # 0.82
print(result.passed)     # True
print(result.assertions) # [{"name": "tool_success_rate", "passed": True, "value": 0.9}, ...]
```

### Tool Mock Registry

Deterministic, replayable tool execution – no real API calls needed.  Tool mocks are grouped by domain so you only load what you need:

```python
from agent_trial_bench import (
    MockToolRegistry, ToolFailureMode,
    register_general_tools,   # web_search, document_fetch, calculator, python_exec, …
    register_web3_tools,      # blockchain_query, contract_read, etherscan_api, price_query, …
    register_default_tools,   # convenience = general + web3
)

registry = MockToolRegistry(cache_dir="output/tool_cache")
register_general_tools(registry)
register_web3_tools(registry)
# or: register_default_tools(registry, include_web3=False)   # general only

registry.inject_failure("web_search", ToolFailureMode.TIMEOUT)
result = await registry.call("web_search", {"query": "latest AI research"})
```

### Analyzer

```python
from agent_trial_bench import FailureClassifier, RegressionDetector, RootCauseAnalyzer

classifier = FailureClassifier()
failures = classifier.classify_batch(trials)

detector = RegressionDetector(regression_threshold=0.05)
report = detector.compare(baseline_snapshots, current_snapshots)
print(report.summary_text())

analyzer = RootCauseAnalyzer()
rc = analyzer.analyse_from_trials(trials, total_trials=300)
print(rc.summary_text())
# tool_error       ████████████ 60%
# reasoning_error  █████        25%
# planning_error   ███          15%
```

### CI/CD Gate

```python
from agent_trial_bench import CIGate, GateConfig

gate = CIGate(GateConfig(
    min_success_rate=0.92,
    max_cost_increase_pct=5.0,
    max_regressions=0,
    max_new_failure_modes=0,
))

result = gate.evaluate(current_results, baseline_results)
if not result.passed:
    print(result.report)
    sys.exit(1)
```

**CLI** (for GitHub Actions / Jenkins):

```bash
python -m agent_trial_bench.ci_gate \
  --current  output/results.json \
  --baseline output/baseline.json \
  --min-success-rate 0.92 \
  --max-cost-increase 5.0
# Exit 0 = PASS, 1 = FAIL
```

---

## Quick Start

### Install

```bash
pip install -r requirements.txt
# or, once published:
# pip install agent-trial-bench
```

### Run the mock pipeline (no real agent required)

```bash
python examples/general_usage.py     # domain-agnostic QA grading demo
python examples/mock_accuracy_run.py  # full pipeline with the bundled Web3 benchmark
```

### Evaluate a general-purpose agent

```python
import asyncio, os
from agent_trial_bench import (
    AgentTrialBench, EvaluationConfig, LLMConfig,
    AgentMetadata, TaskCategory,
)

async def main():
    config = EvaluationConfig(
        llm_config=LLMConfig(model="gpt-4", api_key=os.environ["OPENAI_API_KEY"]),
    )
    evaluator = AgentTrialBench(config)

    agent = AgentMetadata(
        url="http://localhost:8002",
        capabilities=[TaskCategory.FACT_QA],
    )

    result = await evaluator.evaluate_agent(
        question="Who proposed the Transformer architecture and in which year?",
        agent_metadata=agent,
        category=TaskCategory.FACT_QA,
        expected_answer="Vaswani et al., 2017",
    )
    print(f"Score: {result.evaluation_score:.2f}")

asyncio.run(main())
```

Swap in the Web3 vertical by changing `category=TaskCategory.ONCHAIN_RETRIEVAL` and pointing at a Web3 agent – everything else stays the same.

### Full dataset evaluation

```python
results = await evaluator.evaluate_agent_with_dataset(
    agent_metadata=agent,
    dataset_path="data/benchmark.csv",
)
summary = evaluator.export_results("json")
print(f"Success rate: {summary['overall']['success_rate']:.2%}")
```

---

## Task Categories

Generic, domain-agnostic categories always available:

| Category | Description |
|---|---|
| `general` | Unlabelled / mixed tasks – safe default |
| `fact_qa` | Single-answer factual questions |
| `research` | Open-ended multi-source investigation |
| `reasoning` | Math / logic / multi-step deduction |
| `tool_use` | Success depends on invoking the right tools |
| `coding` | Code generation / refactoring / debugging |
| `conversational` | Free-form dialogue / assistant behaviour |
| `multi_step` | Long-horizon planning with intermediate goals |
| `web_retrieval` | Fetch / reason over traditional web sources |

Web3-specific categories (provided by the `web3` plugin):

| Category | Description |
|---|---|
| `onchain_retrieval` | Query blockchain data directly |
| `web_onchain_retrieval` | Hybrid: web + chain verification |

CSV columns expected by the bundled loader: `id`, `question`, `answer`, `category`, `task_type`, `evaluation_method`, `difficulty` (optional), `ground_truth_score` (optional).

### Bundled Datasets

- `data/benchmark.csv` – a 100-task Web3 benchmark kept from v1 as a reference vertical
- `data/dmind/` – 3 156 multiple-choice questions across 9 Web3 domains

Both are examples of what a domain-specific dataset looks like; nothing in the core stops you from pointing the pipeline at a totally different dataset.

---

## Agent Response Schema

To unlock trajectory-based grading (`tool_calls`, `state_check`), agents should return structured JSON:

```json
{
  "answer": "Vaswani et al., 2017",
  "confidence": 0.92,
  "trajectory": {
    "steps": [
      {"step_id": 1, "type": "reasoning", "content": "...", "timestamp": 1704067200.0},
      {"step_id": 2, "type": "tool_call", "tool_name": "arxiv_search",
       "tool_input": {"query": "Attention is all you need"},
       "tool_output": {"hits": 1},
       "success": true, "timestamp": 1704067201.5},
      {"step_id": 3, "type": "reasoning", "content": "Found the paper...", "timestamp": 1704067203.0}
    ],
    "n_turns": 3,
    "total_duration": 3.0
  },
  "state": {
    "primary_source": "https://arxiv.org/abs/1706.03762",
    "sources_consulted": ["arxiv.org", "nature.com"]
  },
  "tools_used": ["arxiv_search"],
  "token_usage": {"prompt": 150, "completion": 80, "total": 230}
}
```

Agents that return only `{"answer": "..."}` still work – the framework degrades gracefully to deterministic + LLM grading only.

---

## Module Reference

### Data Structures

| Class | Description |
|---|---|
| `Step` | Single action within a trajectory |
| `Trajectory` | Full execution trace (steps, counters, loop detection) |
| `Outcome` | Final answer + state + sources |
| `TrackedMetrics` | `n_turns`, `n_toolcalls`, `tokens`, `latency`, `cost` |
| `GradeResult` | Score + assertions from one grader |
| `Trial` | One complete run: Trajectory + Outcome + Metrics + GradeResults |

### Graders

```python
from agent_trial_bench import build_grader

grader = build_grader("deterministic_tests",
                      config={"regex_patterns": [r"\d{4}-\d{2}-\d{2}"]})
```

Available: `deterministic_tests`, `llm_rubric`, `state_check`, `tool_calls`.

### Domains

```python
from agent_trial_bench import (
    DomainPlugin, register_domain, get_domain, list_domains,
)
from agent_trial_bench.domains import (
    get_validator, expected_tools_for, all_trusted_sources,
)

list_domains()                          # ['general', 'web3']
get_validator("url")("https://...")     # → True / False
expected_tools_for("fact_qa")           # → {'web_search', 'document_fetch', ...}
```

### Scenario Generator

```python
ScenarioGenerator(seed=42, cache_dir="output/tool_cache")
  .generate(task, perturbations=[Perturbation.TOOL_FAILURE, ...])
  .generate_batch(tasks, perturbations_per_task)
  .generate_all_combinations(task, max_combinations=10)
```

### Tool Mock Registry

```python
MockToolRegistry(cache_dir="output/tool_cache", default_latency_ms=50)
  .register(name, fn)
  .inject_failure(tool_name, ToolFailureMode.TIMEOUT)
  .call(tool_name, args)
  .call_log   # List[MockToolResponse]
```

Failure modes: `NONE`, `TIMEOUT`, `WRONG_DATA`, `EMPTY_RESULT`, `HTTP_ERROR`, `RATE_LIMIT`.

### Analyzer

```python
FailureClassifier(tool_error_threshold=0.7, pass_threshold=0.6)
  .classify(trial) -> Optional[ClassifiedFailure]
  .classify_batch(trials) -> List[ClassifiedFailure]

RegressionDetector(regression_threshold=0.05)
  .compare(baseline, candidate) -> RegressionReport

RootCauseAnalyzer()
  .analyse_from_trials(trials, total_trials) -> RootCauseReport
```

Failure categories: `tool_error`, `reasoning_error`, `planning_error`, `loop_detected`, `no_answer`, `unknown`.

### CI Gate

```python
GateConfig(
    min_success_rate=0.92,
    max_cost_increase_pct=5.0,
    max_regressions=0,
    max_new_failure_modes=0,
    variance_threshold=0.15,
)

CIGate(config).evaluate(current_results, baseline_results) -> GateResult
```

---

## Examples

| File | Description |
|---|---|
| `examples/general_usage.py` | End-to-end non-Web3 QA flow with all graders + custom domain plugin |
| `examples/basic_usage.py` | Single-question evaluation |
| `examples/batch_evaluation.py` | Batch processing |
| `examples/config_based_evaluation.py` | Config-file driven evaluation |
| `examples/enhanced_batch_evaluation.py` | Advanced batch with statistics |
| `examples/evaluate_accuracy.py` | Evaluation system accuracy analysis |
| `examples/mock_accuracy_run.py` | Full pipeline offline (no real agent) |
| `examples/prepare_dmind.py` | Convert DMind dataset to the bundled format |

---

## Requirements

- Python 3.8+
- `httpx`, `openai`, `pydantic`, `fastapi`, `uvicorn`
- `sentence-transformers`, `scikit-learn`, `numpy`

---

## Backward Compatibility

The v1 legacy imports still work via a lightweight shim:

- `dab_eval.graders.web3_validators` re-exports from `agent_trial_bench.domains.web3` (emits a `DeprecationWarning`)
- Task-context flags `check_address` / `check_tx_hash` still work alongside the new `check_format` list
- Web3-only datasets (`category` = `onchain_retrieval` / `web_onchain_retrieval` / `web_retrieval`) run unchanged

New code should prefer:

```python
from agent_trial_bench import AgentTrialBench          # was DABEvaluator
from agent_trial_bench.domains import get_validator    # was dab_eval.graders.web3_validators
```

---

## License

MIT License
