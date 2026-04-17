# Genuine Agent Evaluation 实施计划

> 基于 [Anthropic: Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) 的最佳实践，升级 Agent Trial Bench 为真正的 Agent 评测框架。

**文档版本**: 2.0  
**创建日期**: 2026-01-12  
**最后更新**: 2026-01-13  
**状态**: 规划中

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [Anthropic 架构解读](#2-anthropic-架构解读)
3. [现状分析](#3-现状分析)
4. [目标架构](#4-目标架构)
5. [实施阶段](#5-实施阶段)
6. [技术规格](#6-技术规格)
7. [风险与缓解](#7-风险与缓解)
8. [成功指标](#8-成功指标)

---

## 1. 背景与动机

### 1.1 核心问题

当前 Agent Trial Bench 采用**结果导向**的评测模式：

```
Question → Agent (黑盒) → Final Answer → Score
```

这种模式存在以下问题：

| 问题 | 影响 |
|------|------|
| **假阳性** | Agent 通过错误的推理路径得到正确答案，评测无法发现 |
| **不可复现** | 同一 Agent 下次可能因中间步骤错误而失败 |
| **无法诊断** | 无法定位 Agent 具体在哪个环节出问题 |
| **效率盲区** | 无法评估 Agent 是否使用了最优的执行路径 |

### 1.2 Anthropic 的关键洞察

> "Agents operate over many turns: calling tools, modifying state, and adapting based on intermediate results. These same capabilities that make AI agents useful—autonomy, intelligence, and flexibility—also make them harder to evaluate."

**Agent 评测 ≠ LLM 评测**

- LLM 评测：输入 → 输出 → 评分
- Agent 评测：输入 → **轨迹（Transcript）** → 输出 → **多维评分**

### 1.3 升级目标

将 Agent Trial Bench 从「结果评测」升级为「**过程+结果**」综合评测系统，完全对齐 Anthropic 的 Agent 评测架构。

---

## 2. Anthropic 架构解读

### 2.1 官方架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Evaluation Harness (评测框架)                     │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Evaluation Suite (评测套件)                      │  │
│  │                                                                    │  │
│  │  ┌────────────────────────────────────────────────────────────┐   │  │
│  │  │                      Task (单个任务)                         │   │  │
│  │  │  Task: "Fix authenticated bypass when..."                   │   │  │
│  │  │                                                              │   │  │
│  │  │  ┌────────────────────────────────────────────────────────┐ │   │  │
│  │  │  │ Graders (多种评分器)                                     │ │   │  │
│  │  │  │ ┌──────────────────┐ ┌─────────────┐ ┌──────────────┐  │ │   │  │
│  │  │  │ │deterministic_tests│ │ llm_rubric  │ │ state_check  │  │ │   │  │
│  │  │  │ └──────────────────┘ └─────────────┘ └──────────────┘  │ │   │  │
│  │  │  │ ┌──────────────────┐                                   │ │   │  │
│  │  │  │ │   tool_calls     │                                   │ │   │  │
│  │  │  │ └──────────────────┘                                   │ │   │  │
│  │  │  └────────────────────────────────────────────────────────┘ │   │  │
│  │  │                                                              │   │  │
│  │  │  ┌────────────────────────────────────────────────────────┐ │   │  │
│  │  │  │ Tracked Metrics (跟踪指标)                               │ │   │  │
│  │  │  │ ┌──────────┐ ┌─────────────┐ ┌────────┐ ┌─────────┐   │ │   │  │
│  │  │  │ │ n_turns  │ │ n_toolcalls │ │ tokens │ │ latency │   │ │   │  │
│  │  │  │ └──────────┘ └─────────────┘ └────────┘ └─────────┘   │ │   │  │
│  │  │  └────────────────────────────────────────────────────────┘ │   │  │
│  │  │                                                              │   │  │
│  │  │  ┌────────────────────────────────────────────────────────┐ │   │  │
│  │  │  │ Trials (多次试验)                                        │ │   │  │
│  │  │  │  ┌─────────────────────────────────────────────────┐   │ │   │  │
│  │  │  │  │ Trial #4                                         │   │ │   │  │
│  │  │  │  │ Trajectory: messages, tool_calls, reasoning...  │   │ │   │  │
│  │  │  │  └─────────────────────────────────────────────────┘   │ │   │  │
│  │  │  └────────────────────────────────────────────────────────┘ │   │  │
│  │  └────────────────────────────────────────────────────────────┘   │  │
│  │                                                                    │  │
│  │  ┌─────────┐  ┌─────────┐  (更多 Tasks...)                        │  │
│  │  │  Task   │  │  Task   │                                         │  │
│  │  │   ...   │  │   ...   │                                         │  │
│  │  └─────────┘  └─────────┘                                         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                    │                                     │
│                                    ↓                                     │
│                         ┌──────────────────┐                            │
│                         │     Outcome      │                            │
│                         │ Final env state  │                            │
│                         └────────┬─────────┘                            │
│                                  │                                       │
│                                  ↓                                       │
│                         ┌──────────────────┐                            │
│                         │  Grader Evaluate │                            │
│                         │ Trajectory +     │                            │
│                         │ Outcome → Scores │                            │
│                         └──────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────┘
                                   ↕
                          ┌─────────────────┐
                          │  Agent Harness  │
                          │   (被测 Agent)   │
                          └─────────────────┘
```

### 2.2 核心概念定义

根据 Anthropic 的定义：

| 概念 | 定义 | 关键点 |
|------|------|--------|
| **Task** | 单个测试用例 | = inputs + success criteria + **graders** + **metrics** |
| **Trial** | 一次执行尝试 | 同一 Task 可运行多次 Trial |
| **Trajectory** | 完整执行记录 | = messages + tool_calls + reasoning + 中间状态 |
| **Graders** | 评分器集合 | 一个 Task 可有**多个并行 Graders** |
| **Outcome** | 最终环境状态 | 不只是答案文本，而是环境的最终状态 |
| **Tracked Metrics** | 运行时指标 | n_turns, n_toolcalls, tokens, latency |

### 2.3 四种 Graders 详解

Anthropic 定义了 **4 种并行的 Graders**，这是架构的核心：

| Grader | 类型 | 职责 | 评估对象 |
|--------|------|------|----------|
| **deterministic_tests** | Code-based | 确定性测试（单元测试、正则、精确匹配） | Outcome |
| **llm_rubric** | Model-based | LLM 按评分标准打分 | Trajectory + Outcome |
| **state_check** | Code-based | 检查最终状态（如数据库是否正确更新） | Outcome (环境状态) |
| **tool_calls** | Code-based | 检查工具调用正确性 | Trajectory |

**关键洞察**:
- Graders **同时评估** Trajectory（过程）和 Outcome（结果）
- **多个 Graders 并行运行**，各自产生分数
- Graders 是 Task 级别的配置，不同 Task 可以有不同的 Grader 组合

### 2.4 Tracked Metrics

Anthropic 推荐跟踪的运行时指标：

| Metric | 含义 | 用途 |
|--------|------|------|
| **n_turns** | Agent 执行的对话轮数 | 评估复杂度/效率 |
| **n_toolcalls** | 工具调用总次数 | 评估工具使用效率 |
| **tokens** | Token 使用量 | 成本分析 |
| **latency** | 响应延迟 | 性能分析 |

### 2.5 Multi-Trial 设计

> "Because model outputs vary between runs, we run multiple trials to produce more consistent results."

- 同一 Task 运行多次 Trial
- 每个 Trial 产生独立的 Trajectory 和 Outcome
- 使用 **Pass@k** 指标聚合多次结果
- 提高评估的统计可靠性

---

## 3. 现状分析

### 3.1 当前架构

```
┌─────────────────────────────────────────────────────┐
│               Agent Trial Bench                     │
├─────────────────────────────────────────────────────┤
│  EvaluationConfig → EvaluationEngine → Summarizer   │
│         ↓                 ↓                          │
│   AgentRunner        Evaluators                      │
│   (获取答案)      ┌─────────────────┐               │
│        │          │ - HybridEvaluator│               │
│        │          │ - LLMEvaluator   │               │
│        │          │ - CascadeEvaluator│              │
│        ▼          └─────────────────┘               │
│   agent_response         ↓                          │
│        └────────────────►│                          │
│                          ▼                          │
│                  evaluation_score                   │
└─────────────────────────────────────────────────────┘
```

### 3.2 现有能力

| 模块 | 功能 | 状态 |
|------|------|------|
| `EvaluationEngine` | 核心评测逻辑 | ✅ 完善 |
| `HybridEvaluator` | 规则+LLM 混合评测 | ✅ 完善 |
| `EnhancedScoringSystem` | 多维度评分 | ✅ 完善 |
| `CalibrationManager` | 分数校准 | ⚠️ 未集成 |
| `EvaluationAccuracyAnalyzer` | 评测准确性分析 | ✅ 完善 |
| **Transcript 记录** | 执行轨迹 | ❌ 缺失 |
| **Process Graders** | 过程评估器 | ❌ 缺失 |

### 3.3 数据结构现状

**当前 `EvaluationResult`**:
```python
@dataclass
class EvaluationResult:
    task_id: str
    question: str
    agent_response: str          # 仅最终答案
    evaluation_score: float      # 仅最终分数
    evaluation_reasoning: str
    confidence: float
    processing_time: float
    tools_used: List[str]        # 仅工具名称列表
    metadata: Dict[str, Any]
    status: EvaluationStatus
    error: Optional[str] = None
```

**问题**: 
- `tools_used` 只是名称列表，无调用详情
- 缺少推理步骤记录
- 缺少中间状态

### 3.4 DAB 与 Anthropic 架构差距对比

| Anthropic 组件 | DAB 当前状态 | 差距级别 |
|---------------|-------------|----------|
| **Task** (inputs + criteria + graders + metrics) | ⚠️ 只有 inputs + expected_answer | 🟡 中等 |
| **Graders: deterministic_tests** | ✅ `EnhancedScoringSystem` | 🟢 已有 |
| **Graders: llm_rubric** | ✅ `LLMEvaluator` | 🟢 已有 |
| **Graders: state_check** | ❌ 缺失 | 🔴 需新增 |
| **Graders: tool_calls** | ❌ 缺失 | 🔴 需新增 |
| **Tracked Metrics: n_turns** | ❌ 未跟踪 | 🔴 需新增 |
| **Tracked Metrics: n_toolcalls** | ⚠️ 只有 tools_used 列表 | 🟡 需增强 |
| **Tracked Metrics: tokens** | ❌ 未跟踪 | 🔴 需新增 |
| **Tracked Metrics: latency** | ✅ `processing_time` | 🟢 已有 |
| **Trials** (多次运行) | ⚠️ 单次运行 | 🟡 需新增 |
| **Trajectory** (执行轨迹) | ❌ 缺失 | 🔴 核心差距 |
| **Outcome** (环境状态) | ⚠️ 只有 agent_response | 🟡 需增强 |
| **Pass@k** | ✅ 已实现 | 🟢 已有 |

**核心差距**: DAB 缺少 **Trajectory 记录** 和 **tool_calls/state_check Graders**

---

## 4. 目标架构

### 4.1 架构图（对齐 Anthropic）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Agent Trial Bench v2.0                               │
│                        (Evaluation Harness)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    Evaluation Suite                                    │  │
│  │                                                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                         Task                                     │  │  │
│  │  │  question + expected_answer + category + task_type               │  │  │
│  │  │                                                                  │  │  │
│  │  │  ┌────────────────────────────────────────────────────────────┐ │  │  │
│  │  │  │ Graders (并行运行)                                          │ │  │  │
│  │  │  │                                                             │ │  │  │
│  │  │  │  ┌─────────────────┐  ┌─────────────────┐                  │ │  │  │
│  │  │  │  │deterministic_   │  │   llm_rubric    │                  │ │  │  │
│  │  │  │  │tests            │  │                 │                  │ │  │  │
│  │  │  │  │(EnhancedScoring)│  │ (LLMEvaluator)  │                  │ │  │  │
│  │  │  │  └─────────────────┘  └─────────────────┘                  │ │  │  │
│  │  │  │                                                             │ │  │  │
│  │  │  │  ┌─────────────────┐  ┌─────────────────┐                  │ │  │  │
│  │  │  │  │   state_check   │  │   tool_calls    │  ← 新增          │ │  │  │
│  │  │  │  │   (新增)        │  │   (新增)        │                  │ │  │  │
│  │  │  │  └─────────────────┘  └─────────────────┘                  │ │  │  │
│  │  │  └────────────────────────────────────────────────────────────┘ │  │  │
│  │  │                                                                  │  │  │
│  │  │  ┌────────────────────────────────────────────────────────────┐ │  │  │
│  │  │  │ Tracked Metrics                                             │ │  │  │
│  │  │  │ ┌──────────┐ ┌─────────────┐ ┌────────┐ ┌─────────┐        │ │  │  │
│  │  │  │ │ n_turns  │ │ n_toolcalls │ │ tokens │ │ latency │        │ │  │  │
│  │  │  │ └──────────┘ └─────────────┘ └────────┘ └─────────┘        │ │  │  │
│  │  │  └────────────────────────────────────────────────────────────┘ │  │  │
│  │  │                                                                  │  │  │
│  │  │  ┌────────────────────────────────────────────────────────────┐ │  │  │
│  │  │  │ Trials (多次执行)                                           │ │  │  │
│  │  │  │  ┌─────────────────────────────────────────────────────┐   │ │  │  │
│  │  │  │  │ Trial #1, #2, #3...                                  │   │ │  │  │
│  │  │  │  │ Trajectory: messages, tool_calls, reasoning...      │   │ │  │  │
│  │  │  │  └─────────────────────────────────────────────────────┘   │ │  │  │
│  │  │  └────────────────────────────────────────────────────────────┘ │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                     │                                        │
│                    ┌────────────────┴────────────────┐                      │
│                    ▼                                 ▼                      │
│           ┌──────────────┐                 ┌─────────────────┐              │
│           │   Outcome    │                 │   Trajectory    │              │
│           │ (最终状态)   │                 │ (执行轨迹)       │              │
│           └──────┬───────┘                 └────────┬────────┘              │
│                  │                                   │                       │
│                  └─────────────┬─────────────────────┘                      │
│                                │                                             │
│                                ▼                                             │
│                    ┌───────────────────────┐                                │
│                    │   Grader Evaluate     │                                │
│                    │   (综合评分)          │                                │
│                    │                       │                                │
│                    │ Trajectory + Outcome  │                                │
│                    │      → Scores         │                                │
│                    └───────────────────────┘                                │
│                                │                                             │
│                                ▼                                             │
│                  ComprehensiveEvaluationResult                              │
│                  {                                                          │
│                    outcome_score,                                           │
│                    grader_scores: {                                         │
│                      deterministic_tests,                                   │
│                      llm_rubric,                                            │
│                      state_check,                                           │
│                      tool_calls                                             │
│                    },                                                       │
│                    tracked_metrics: {n_turns, n_toolcalls, tokens, latency},│
│                    trajectory,                                              │
│                    pass_at_k                                                │
│                  }                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↕
                           ┌─────────────────┐
                           │  Agent Harness  │
                           │   (被测 Agent)   │
                           └─────────────────┘
```

### 4.2 核心新增组件

按 Anthropic 架构对齐后的组件列表：

| 组件 | Anthropic 对应 | 职责 | 优先级 |
|------|---------------|------|--------|
| `Trajectory` | Trajectory | 执行轨迹数据结构 | P0 |
| `Trial` | Trial | 单次执行记录 | P0 |
| `TrackedMetrics` | Tracked Metrics | 运行时指标收集 | P0 |
| `ToolCallsGrader` | tool_calls grader | 评估工具调用正确性 | P0 |
| `StateCheckGrader` | state_check grader | 评估最终状态正确性 | P1 |
| `MultiTrialRunner` | Trials | 多次执行管理器 | P1 |
| `ComprehensiveEvaluator` | Grader Evaluate | 综合评分器 | P0 |

### 4.3 四种 Graders 映射

| Anthropic Grader | DAB 实现 | 状态 | 评估对象 |
|------------------|---------|------|----------|
| **deterministic_tests** | `EnhancedScoringSystem` | ✅ 已有 | Outcome |
| **llm_rubric** | `LLMEvaluator` | ✅ 已有 | Trajectory + Outcome |
| **state_check** | `StateCheckGrader` | 🆕 新增 | Outcome (环境状态) |
| **tool_calls** | `ToolCallsGrader` | 🆕 新增 | Trajectory |

### 4.4 Tracked Metrics 实现

| Metric | 实现方式 | 收集点 |
|--------|---------|--------|
| **n_turns** | 统计 Trajectory 中的轮次 | Trial 结束时 |
| **n_toolcalls** | 统计 Trajectory 中 tool_call 类型步骤 | Trial 结束时 |
| **tokens** | 从 Agent 响应或 LLM API 获取 | 每次调用后 |
| **latency** | 已有 `processing_time` | Trial 结束时 |

### 4.5 评测维度对比

| 维度 | 当前 | 目标 | 说明 |
|------|------|------|------|
| **deterministic_tests** | ✅ | ✅ | 规则匹配、精确匹配 |
| **llm_rubric** | ✅ | ✅ | LLM 评分 |
| **state_check** | ❌ | ✅ | 最终环境状态验证 |
| **tool_calls** | ❌ | ✅ | 工具调用过程验证 |
| **Multi-Trial** | ❌ | ✅ | 多次执行 + Pass@k |
| **Tracked Metrics** | ⚠️ | ✅ | 完整运行时指标 |

---

## 5. 实施阶段

### Phase 1: 核心数据结构 (Week 1-2)

**目标**: 建立 Trajectory 和 Trial 基础设施

| 任务 | 文件 | 描述 |
|------|------|------|
| 1.1 | `agent_trial_bench/trajectory.py` | 定义 `Trajectory`, `Step`, `Trial` 数据结构 |
| 1.2 | `agent_trial_bench/metrics.py` | 定义 `TrackedMetrics` 数据结构 |
| 1.3 | `agent_trial_bench/dataclasses.py` | 扩展 `EvaluationTask` 包含 graders 配置 |
| 1.4 | `agent_trial_bench/runners/agent_runner.py` | 改进以解析 Agent trajectory 响应 |

**交付物**:
- [ ] `Trajectory` 数据类 (对应 Anthropic Trajectory)
- [ ] `Step` 数据类 (messages, tool_calls, reasoning)
- [ ] `Trial` 数据类 (一次完整执行)
- [ ] `TrackedMetrics` 数据类 (n_turns, n_toolcalls, tokens, latency)
- [ ] Agent 响应解析逻辑

**Agent 响应规范（推荐）**:
```json
{
  "answer": "最终答案",
  "confidence": 0.85,
  "trajectory": {
    "steps": [
      {
        "step_id": 1,
        "type": "reasoning",
        "content": "分析问题：用户询问 Bitcoin ETF 的批准日期...",
        "timestamp": 1704067200.0
      },
      {
        "step_id": 2,
        "type": "tool_call",
        "tool_name": "web_search",
        "tool_input": {"query": "SEC Bitcoin ETF approval date 2024"},
        "tool_output": {"results": [...]},
        "success": true,
        "timestamp": 1704067201.5
      },
      {
        "step_id": 3,
        "type": "reasoning",
        "content": "根据搜索结果，SEC 于 2024年1月10日批准...",
        "timestamp": 1704067203.0
      }
    ],
    "n_turns": 3,
    "total_duration": 3.0
  },
  "state": {
    "final_answer_source": "web_search",
    "sources_consulted": ["sec.gov", "reuters.com"]
  },
  "tools_used": ["web_search"],
  "token_usage": {"prompt": 150, "completion": 80, "total": 230}
}
```

### Phase 2: 四种 Graders 实现 (Week 3-4)

**目标**: 实现 Anthropic 定义的四种 Graders

| 任务 | 文件 | 描述 |
|------|------|------|
| 2.1 | `agent_trial_bench/graders/__init__.py` | Graders 包 |
| 2.2 | `agent_trial_bench/graders/base.py` | `BaseGrader` 基类 |
| 2.3 | `agent_trial_bench/graders/deterministic.py` | `DeterministicTestsGrader` (已有 EnhancedScoring 重构) |
| 2.4 | `agent_trial_bench/graders/llm_rubric.py` | `LLMRubricGrader` (已有 LLMEvaluator 重构) |
| 2.5 | `agent_trial_bench/graders/state_check.py` | `StateCheckGrader` (新增) |
| 2.6 | `agent_trial_bench/graders/tool_calls.py` | `ToolCallsGrader` (新增) |

**四种 Graders 职责**:

**deterministic_tests (已有，重构)**:
```python
class DeterministicTestsGrader(BaseGrader):
    """确定性测试 Grader"""
    def grade(self, outcome: Outcome, trajectory: Trajectory = None) -> GradeResult:
        # 精确匹配
        # 正则匹配
        # 格式验证
        # 数值范围检查
```

**llm_rubric (已有，重构)**:
```python
class LLMRubricGrader(BaseGrader):
    """LLM 评分标准 Grader"""
    def grade(self, outcome: Outcome, trajectory: Trajectory = None) -> GradeResult:
        # 基于 rubric 的 LLM 评分
        # 同时考虑 trajectory 和 outcome
```

**state_check (新增)**:
```python
class StateCheckGrader(BaseGrader):
    """状态检查 Grader - 验证最终环境状态"""
    def grade(self, outcome: Outcome, trajectory: Trajectory = None) -> GradeResult:
        # 检查 outcome.state 中的关键状态
        # Web3 场景：检查交易是否正确引用、地址是否有效
        # 检查数据源是否可靠
```

**tool_calls (新增)**:
```python
class ToolCallsGrader(BaseGrader):
    """工具调用 Grader - 验证工具使用过程"""
    def grade(self, outcome: Outcome, trajectory: Trajectory) -> GradeResult:
        # 工具调用成功率
        # 工具选择合理性
        # 参数格式正确性 (Web3 地址/交易哈希)
        # 工具调用顺序合理性
        # 冗余调用检测
```

### Phase 3: Multi-Trial 与 Metrics (Week 5)

**目标**: 实现多次执行和指标收集

| 任务 | 文件 | 描述 |
|------|------|------|
| 3.1 | `agent_trial_bench/runners/multi_trial_runner.py` | `MultiTrialRunner` 多次执行管理器 |
| 3.2 | `agent_trial_bench/metrics.py` | `MetricsCollector` 指标收集器 |
| 3.3 | `agent_trial_bench/evaluation/accuracy_analysis.py` | 增强 Pass@k 与 Multi-Trial 分析 |

**MultiTrialRunner 设计**:
```python
class MultiTrialRunner:
    """多次执行管理器"""
    
    def __init__(self, num_trials: int = 3, aggregation: str = "pass_at_k"):
        self.num_trials = num_trials
        self.aggregation = aggregation
    
    async def run_trials(self, task: Task, agent: Agent) -> List[Trial]:
        trials = []
        for i in range(self.num_trials):
            trial = await self.run_single_trial(task, agent)
            trials.append(trial)
        return trials
    
    def aggregate_results(self, trials: List[Trial], k: int = 1) -> AggregatedResult:
        # Pass@k 计算
        # 一致性分析
        # 最佳结果选择
```

**TrackedMetrics 收集**:
```python
@dataclass
class TrackedMetrics:
    n_turns: int          # Trajectory 中的轮次数
    n_toolcalls: int      # 工具调用次数
    tokens: TokenUsage    # Token 使用量
    latency: float        # 响应延迟（秒）
    
    # 扩展指标
    error_count: int      # 错误次数
    retry_count: int      # 重试次数
```

### Phase 4: 综合评估器 (Week 6)

**目标**: 整合所有 Graders 和 Metrics

| 任务 | 文件 | 描述 |
|------|------|------|
| 4.1 | `agent_trial_bench/evaluation/comprehensive_evaluator.py` | `ComprehensiveEvaluator` |
| 4.2 | `agent_trial_bench/dataclasses.py` | `ComprehensiveEvaluationResult` |
| 4.3 | `agent_trial_bench/evaluation_engine.py` | 集成新架构 |
| 4.4 | `agent_trial_bench/config/evaluation.py` | 配置项扩展 |

**配置示例（对齐 Anthropic）**:
```json
{
  "evaluator_config": {
    "type": "comprehensive",
    "graders": {
      "deterministic_tests": {
        "enabled": true,
        "weight": 0.3,
        "config": {
          "exact_match": true,
          "regex_patterns": ["\\d{4}/\\d{1,2}/\\d{1,2}"]
        }
      },
      "llm_rubric": {
        "enabled": true,
        "weight": 0.3,
        "config": {
          "rubric": "accuracy, completeness, professionalism"
        }
      },
      "state_check": {
        "enabled": true,
        "weight": 0.2,
        "config": {
          "check_sources": true,
          "validate_web3_addresses": true
        }
      },
      "tool_calls": {
        "enabled": true,
        "weight": 0.2,
        "config": {
          "check_success_rate": true,
          "check_relevance": true
        }
      }
    },
    "tracked_metrics": ["n_turns", "n_toolcalls", "tokens", "latency"],
    "multi_trial": {
      "enabled": true,
      "num_trials": 3,
      "aggregation": "pass_at_k",
      "k": 1
    }
  }
}
```

### Phase 5: 校准与分析 (Week 7)

**目标**: 完善校准和分析能力

| 任务 | 文件 | 描述 |
|------|------|------|
| 5.1 | `agent_trial_bench/calibration.py` | 集成 CalibrationManager 到各 Grader |
| 5.2 | `agent_trial_bench/evaluation/accuracy_analysis.py` | 扩展支持 Grader 级别分析 |
| 5.3 | `agent_trial_bench/summarizers/default.py` | 扩展汇总报告 |

**交付物**:
- [ ] 各 Grader 的校准支持
- [ ] Grader 级别的 bias 检测
- [ ] Multi-Trial 一致性报告
- [ ] 综合评估报告模板

### Phase 6: 文档与示例 (Week 8)

| 任务 | 文件 | 描述 |
|------|------|------|
| 6.1 | `examples/genuine_agent_evaluation.py` | 完整 Agent 评估示例 |
| 6.2 | `examples/multi_trial_evaluation.py` | Multi-Trial 评估示例 |
| 6.3 | `examples/custom_graders.py` | 自定义 Grader 示例 |
| 6.4 | `docs/GENUINE_AGENT_EVAL.md` | 完整文档 |
| 6.5 | `README.md` | 更新主文档 |

**交付物**:
- [ ] 示例代码
- [ ] API 文档
- [ ] 自定义 Grader 指南
- [ ] 最佳实践指南

---

## 6. 技术规格

### 6.1 核心数据结构（对齐 Anthropic）

```python
# trajectory.py - 对应 Anthropic 的 Trajectory

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum


class StepType(Enum):
    """Agent 执行步骤类型"""
    MESSAGE = "message"          # 消息（用户/助手）
    REASONING = "reasoning"      # 推理思考
    TOOL_CALL = "tool_call"      # 工具调用
    TOOL_RESULT = "tool_result"  # 工具返回结果
    ERROR = "error"              # 错误


@dataclass
class Step:
    """Trajectory 中的单个步骤"""
    step_id: int
    step_type: StepType
    timestamp: float
    content: str
    
    # 工具调用相关
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[Any] = None
    tool_success: bool = True
    
    # 推理相关
    reasoning: Optional[str] = None
    confidence: float = 1.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """完整的 Agent 执行轨迹 - 对应 Anthropic Trajectory"""
    task_id: str
    steps: List[Step] = field(default_factory=list)
    
    # 统计
    n_turns: int = 0              # 对话轮次
    n_toolcalls: int = 0          # 工具调用次数
    n_reasoning_steps: int = 0    # 推理步骤数
    n_errors: int = 0             # 错误次数
    
    # 时间
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    total_duration: float = 0.0
    
    def add_step(self, step: Step):
        self.steps.append(step)
        if step.step_type == StepType.TOOL_CALL:
            self.n_toolcalls += 1
        elif step.step_type == StepType.REASONING:
            self.n_reasoning_steps += 1
        elif step.step_type == StepType.ERROR:
            self.n_errors += 1


@dataclass
class Outcome:
    """最终环境状态 - 对应 Anthropic Outcome"""
    answer: str                              # 最终答案
    confidence: float = 0.0                  # 置信度
    state: Dict[str, Any] = field(default_factory=dict)  # 环境状态
    sources: List[str] = field(default_factory=list)     # 引用来源
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackedMetrics:
    """运行时指标 - 对应 Anthropic Tracked Metrics"""
    n_turns: int = 0              # 对话轮次
    n_toolcalls: int = 0          # 工具调用次数
    tokens: int = 0               # Token 使用量
    latency: float = 0.0          # 延迟（秒）
    
    # 扩展指标
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_success_rate: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_turns": self.n_turns,
            "n_toolcalls": self.n_toolcalls,
            "tokens": self.tokens,
            "latency": self.latency,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tool_success_rate": self.tool_success_rate,
        }


@dataclass
class Trial:
    """单次执行尝试 - 对应 Anthropic Trial"""
    trial_id: int
    task_id: str
    trajectory: Trajectory
    outcome: Outcome
    metrics: TrackedMetrics
    
    # Grader 结果
    grader_results: Dict[str, "GradeResult"] = field(default_factory=dict)
    
    # 元数据
    started_at: float = 0.0
    completed_at: float = 0.0
    success: bool = True
    error: Optional[str] = None


@dataclass
class GradeResult:
    """单个 Grader 的评分结果"""
    grader_name: str
    score: float                    # 0.0 - 1.0
    passed: bool                    # 是否通过
    reasoning: str                  # 评分理由
    details: Dict[str, Any] = field(default_factory=dict)
    assertions: List[Dict[str, Any]] = field(default_factory=list)  # 各项断言结果


@dataclass
class Task:
    """评测任务 - 对应 Anthropic Task"""
    task_id: str
    question: str
    expected_answer: Optional[str] = None
    category: str = ""
    task_type: str = ""
    
    # Graders 配置
    graders: List[str] = field(default_factory=lambda: [
        "deterministic_tests", "llm_rubric", "state_check", "tool_calls"
    ])
    grader_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Tracked Metrics 配置
    tracked_metrics: List[str] = field(default_factory=lambda: [
        "n_turns", "n_toolcalls", "tokens", "latency"
    ])
    
    # Multi-Trial 配置
    num_trials: int = 1
    
    # 上下文
    context: Dict[str, Any] = field(default_factory=dict)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComprehensiveEvaluationResult:
    """综合评估结果"""
    task_id: str
    question: str
    
    # 多次 Trial 结果
    trials: List[Trial] = field(default_factory=list)
    
    # 聚合的 Grader 分数
    grader_scores: Dict[str, float] = field(default_factory=dict)
    
    # 综合分数
    combined_score: float = 0.0
    
    # Pass@k
    pass_at_k: Dict[int, float] = field(default_factory=dict)  # {k: pass_rate}
    
    # 聚合的 Metrics
    aggregated_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # 一致性分析
    consistency_score: float = 0.0
    variance: float = 0.0
    
    # 最佳 Trial
    best_trial_id: int = 0
    
    # 问题诊断
    issues: List[Dict[str, Any]] = field(default_factory=list)
```

### 6.2 Grader 基类设计

```python
# graders/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseGrader(ABC):
    """Grader 基类 - 对应 Anthropic 的 Grader 概念"""
    
    name: str = "base"
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    @abstractmethod
    async def grade(
        self,
        outcome: Outcome,
        trajectory: Optional[Trajectory] = None,
        task: Optional[Task] = None
    ) -> GradeResult:
        """
        执行评分
        
        Args:
            outcome: 最终状态
            trajectory: 执行轨迹（可选）
            task: 任务信息（包含 expected_answer 等）
        
        Returns:
            GradeResult: 评分结果
        """
        pass
    
    def get_assertions(self) -> List[str]:
        """获取该 Grader 的断言列表"""
        return []
```

### 6.3 四种 Graders 实现规格

```python
# graders/deterministic.py
class DeterministicTestsGrader(BaseGrader):
    """确定性测试 Grader"""
    name = "deterministic_tests"
    
    async def grade(self, outcome, trajectory, task) -> GradeResult:
        assertions = []
        
        # 1. 精确匹配
        if task.expected_answer:
            exact_match = outcome.answer.strip() == task.expected_answer.strip()
            assertions.append({"name": "exact_match", "passed": exact_match})
        
        # 2. 包含检查
        if task.expected_answer:
            contains = task.expected_answer.lower() in outcome.answer.lower()
            assertions.append({"name": "contains_answer", "passed": contains})
        
        # 3. 格式验证（Web3 特定）
        if "address" in task.context:
            valid_addr = self._validate_address(outcome.answer)
            assertions.append({"name": "valid_address", "passed": valid_addr})
        
        passed = sum(1 for a in assertions if a["passed"]) / len(assertions)
        return GradeResult(
            grader_name=self.name,
            score=passed,
            passed=passed >= 0.5,
            reasoning=f"Passed {sum(1 for a in assertions if a['passed'])}/{len(assertions)} assertions",
            assertions=assertions
        )


# graders/tool_calls.py
class ToolCallsGrader(BaseGrader):
    """工具调用 Grader"""
    name = "tool_calls"
    
    async def grade(self, outcome, trajectory, task) -> GradeResult:
        if not trajectory or not trajectory.steps:
            return GradeResult(
                grader_name=self.name,
                score=0.5,
                passed=True,
                reasoning="No trajectory available for tool call analysis"
            )
        
        tool_calls = [s for s in trajectory.steps if s.step_type == StepType.TOOL_CALL]
        
        assertions = []
        
        # 1. 工具调用成功率
        if tool_calls:
            success_rate = sum(1 for t in tool_calls if t.tool_success) / len(tool_calls)
            assertions.append({
                "name": "tool_success_rate",
                "passed": success_rate >= 0.8,
                "value": success_rate
            })
        
        # 2. 工具选择合理性
        expected_tools = self._get_expected_tools(task.category)
        if expected_tools and tool_calls:
            used_tools = {t.tool_name for t in tool_calls}
            relevance = len(used_tools & expected_tools) / len(used_tools) if used_tools else 0
            assertions.append({
                "name": "tool_relevance",
                "passed": relevance >= 0.5,
                "value": relevance
            })
        
        # 3. 冗余检测
        if tool_calls:
            unique_ratio = len(set(t.tool_name for t in tool_calls)) / len(tool_calls)
            assertions.append({
                "name": "no_redundancy",
                "passed": unique_ratio >= 0.5,
                "value": unique_ratio
            })
        
        # 4. Web3 参数验证
        for tc in tool_calls:
            if tc.tool_input:
                valid = self._validate_web3_params(tc.tool_input)
                assertions.append({
                    "name": f"valid_params_{tc.step_id}",
                    "passed": valid
                })
        
        passed_count = sum(1 for a in assertions if a["passed"])
        score = passed_count / len(assertions) if assertions else 0.5
        
        return GradeResult(
            grader_name=self.name,
            score=score,
            passed=score >= 0.6,
            reasoning=f"Tool calls analysis: {passed_count}/{len(assertions)} checks passed",
            assertions=assertions
        )


# graders/state_check.py
class StateCheckGrader(BaseGrader):
    """状态检查 Grader"""
    name = "state_check"
    
    async def grade(self, outcome, trajectory, task) -> GradeResult:
        assertions = []
        
        # 1. 检查来源可靠性
        if outcome.sources:
            reliable_sources = self._check_source_reliability(outcome.sources)
            assertions.append({
                "name": "reliable_sources",
                "passed": reliable_sources,
                "sources": outcome.sources
            })
        
        # 2. 检查状态一致性
        if outcome.state:
            consistent = self._check_state_consistency(outcome.state, outcome.answer)
            assertions.append({
                "name": "state_consistency",
                "passed": consistent
            })
        
        # 3. Web3 特定状态检查
        if "transaction" in outcome.state:
            valid_tx = self._validate_transaction(outcome.state["transaction"])
            assertions.append({
                "name": "valid_transaction",
                "passed": valid_tx
            })
        
        passed_count = sum(1 for a in assertions if a["passed"])
        score = passed_count / len(assertions) if assertions else 0.5
        
        return GradeResult(
            grader_name=self.name,
            score=score,
            passed=score >= 0.6,
            reasoning=f"State check: {passed_count}/{len(assertions)} assertions passed",
            assertions=assertions
        )
```

### 6.4 Web3 特定验证规则

```python
# graders/web3_validators.py

import re
from typing import Dict, Any, Set

# Web3 工具模式定义
WEB3_TOOL_PATTERNS: Dict[str, Dict[str, Any]] = {
    "onchain_retrieval": {
        "expected_tools": {"blockchain_query", "contract_read", "transaction_fetch", "etherscan_api"},
        "required_params": ["chain_id", "address"],
    },
    "defi_analysis": {
        "expected_tools": {"price_query", "liquidity_check", "protocol_info", "dex_api"},
        "required_params": ["token_address", "protocol"],
    },
    "web_retrieval": {
        "expected_tools": {"web_search", "api_call", "document_fetch"},
        "required_params": ["query"],
    },
    "web_onchain_retrieval": {
        "expected_tools": {"web_search", "blockchain_query", "contract_read"},
        "required_params": ["query"],
    },
}

# 正则模式
ADDRESS_PATTERN = re.compile(r"^0x[a-fA-F0-9]{40}$")
TX_HASH_PATTERN = re.compile(r"^0x[a-fA-F0-9]{64}$")
ENS_PATTERN = re.compile(r"^[a-zA-Z0-9-]+\.eth$")


def validate_address(value: str) -> bool:
    """验证以太坊地址格式"""
    return bool(ADDRESS_PATTERN.match(value))


def validate_tx_hash(value: str) -> bool:
    """验证交易哈希格式"""
    return bool(TX_HASH_PATTERN.match(value))


def validate_ens(value: str) -> bool:
    """验证 ENS 域名格式"""
    return bool(ENS_PATTERN.match(value))


def get_expected_tools(category: str) -> Set[str]:
    """根据任务类别获取预期工具"""
    pattern = WEB3_TOOL_PATTERNS.get(category.lower(), {})
    return pattern.get("expected_tools", set())


def validate_web3_params(params: Dict[str, Any]) -> bool:
    """验证 Web3 相关参数格式"""
    for key, value in params.items():
        if not isinstance(value, str):
            continue
        if "address" in key.lower():
            if not (validate_address(value) or validate_ens(value)):
                return False
        if "hash" in key.lower() or "tx" in key.lower():
            if value.startswith("0x") and not validate_tx_hash(value):
                return False
    return True
```

### 6.5 评分公式（对齐 Anthropic 四 Graders）

**综合分数计算**:
```
combined_score = Σ(grader_score_i × grader_weight_i)

默认 Grader 权重:
- deterministic_tests: 0.30
- llm_rubric: 0.30
- state_check: 0.20
- tool_calls: 0.20
```

**Multi-Trial 聚合**:
```
# Pass@k 计算
pass_at_k = 1 - C(n-c, k) / C(n, k)

其中:
- n = 总 trial 数
- c = 通过的 trial 数
- k = Pass@k 的 k 值

# 最终分数
final_score = mean(trial_scores)  # 或 max(trial_scores)
```

**一致性分数**:
```
consistency_score = 1 - (std(trial_scores) / mean(trial_scores))
```

**输出结构**:
```python
{
    "task_id": "...",
    "grader_scores": {
        "deterministic_tests": 0.85,
        "llm_rubric": 0.78,
        "state_check": 0.90,
        "tool_calls": 0.72
    },
    "combined_score": 0.81,
    "pass_at_k": {
        1: 0.67,
        3: 0.89
    },
    "tracked_metrics": {
        "n_turns": 4,
        "n_toolcalls": 3,
        "tokens": 1250,
        "latency": 2.3
    },
    "consistency_score": 0.92,
    "best_trial_id": 2
}
```

---

## 7. 风险与缓解

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| Agent 不返回 trajectory | 无法运行 tool_calls/state_check Grader | 高 | 降级模式：仅运行 deterministic_tests + llm_rubric |
| trajectory 格式不统一 | 解析失败 | 中 | 定义宽松的解析器，支持多种格式变体 |
| Multi-Trial 增加成本 | 评测成本翻倍 | 中 | 可配置 num_trials，默认 1 |
| LLM Grader 不稳定 | 评分波动 | 中 | 使用校准 + Multi-Trial 取平均 |
| 四 Graders 权重不当 | 评分失真 | 低 | 可配置权重 + 定期校准 |

### 7.1 降级策略

当 Agent 不返回 trajectory 时的降级模式：

```python
class DegradedEvaluator:
    """降级评估器 - Agent 不支持 trajectory 时使用"""
    
    # 可用 Graders（不需要 trajectory）
    available_graders = ["deterministic_tests", "llm_rubric"]
    
    # 不可用 Graders（需要 trajectory）
    unavailable_graders = ["tool_calls", "state_check"]  # 部分降级
    
    async def evaluate(self, outcome: Outcome, task: Task) -> ComprehensiveEvaluationResult:
        results = {}
        
        # 运行不需要 trajectory 的 Graders
        for grader_name in self.available_graders:
            grader = self.get_grader(grader_name)
            results[grader_name] = await grader.grade(outcome, trajectory=None, task=task)
        
        # 标记降级 Graders
        for grader_name in self.unavailable_graders:
            results[grader_name] = GradeResult(
                grader_name=grader_name,
                score=0.0,
                passed=False,
                reasoning="Degraded: trajectory not available",
                details={"mode": "degraded"}
            )
        
        return self.aggregate_results(results, mode="degraded")
```

### 7.2 向后兼容

现有的 DAB 评测流程保持兼容：

```python
# 旧接口（保持可用）
result = await evaluator.evaluate_agent(
    question="...",
    agent_metadata=agent,
    category=TaskCategory.WEB_RETRIEVAL,
    expected_answer="..."
)

# 新接口（推荐）
result = await evaluator.evaluate_task(
    task=Task(
        question="...",
        expected_answer="...",
        graders=["deterministic_tests", "llm_rubric", "tool_calls", "state_check"],
        num_trials=3
    ),
    agent=agent
)
```

---

## 8. 成功指标

### 8.1 功能指标

| 指标 | 目标 | 验证方法 |
|------|------|----------|
| 四 Graders 全部实现 | 100% | 单元测试通过 |
| Trajectory 解析成功率 | ≥ 95% | 对支持 trajectory 的 Agent 测试 |
| Multi-Trial 功能 | 可配置 1-10 次 | 配置测试 |
| Tracked Metrics 完整性 | 4 个指标全收集 | 输出检查 |
| Pass@k 计算正确 | 与手动计算一致 | 数学验证 |

### 8.2 质量指标

| 指标 | 目标 | 验证方法 |
|------|------|----------|
| 假阳性检出率 | 提升 30% | 人工标注对比实验 |
| 四 Graders 与人工一致性 | ≥ 0.70 相关系数 | 人工标注对比 |
| 问题可定位性 | 定位到具体 Grader/步骤 | 用户调研 |
| Multi-Trial 一致性 | ≤ 0.15 标准差 | 统计分析 |

### 8.3 性能指标

| 指标 | 目标 | 验证方法 |
|------|------|----------|
| 单 Trial 延迟增加 | ≤ 15% | 基准测试对比 |
| 内存增加 | ≤ 30MB/Trial | Profiling |
| 3-Trial 总耗时 | ≤ 单次 × 3.5 | 并行测试 |

---

## 附录

### A. 相关资源

- [Anthropic: Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) - 核心参考
- [OpenAI Evals](https://github.com/openai/evals)
- [LangSmith Evaluation](https://docs.smith.langchain.com/)
- [Harbor - Agent Evaluation Framework](https://github.com/anthropics/harbor)
- [Promptfoo](https://github.com/promptfoo/promptfoo)

### B. 术语对照表

| Anthropic 术语 | DAB 当前术语 | DAB 新术语 |
|---------------|-------------|-----------|
| Task | EvaluationTask | Task |
| Trial | (无) | Trial |
| Trajectory | (无) | Trajectory |
| Outcome | agent_response | Outcome |
| Grader | Evaluator | Grader |
| deterministic_tests | EnhancedScoringSystem | DeterministicTestsGrader |
| llm_rubric | LLMEvaluator | LLMRubricGrader |
| state_check | (无) | StateCheckGrader |
| tool_calls | (无) | ToolCallsGrader |
| Tracked Metrics | processing_time | TrackedMetrics |
| Evaluation Harness | AgentTrialBench | AgentTrialBench |
| Evaluation Suite | dataset | EvaluationSuite |

### C. 文件变更清单

```
agent_trial_bench/
├── trajectory.py                          # 新增 - Trajectory, Step, Trial, Outcome
├── metrics.py                             # 新增 - TrackedMetrics, MetricsCollector
├── dataclasses.py                         # 修改 - Task, ComprehensiveEvaluationResult
├── evaluation_engine.py                   # 修改 - 集成新架构
├── calibration.py                         # 修改 - 集成到 Graders
├── config/
│   └── evaluation.py                      # 修改 - Grader 配置
├── graders/                               # 新增目录
│   ├── __init__.py
│   ├── base.py                            # BaseGrader
│   ├── deterministic.py                   # DeterministicTestsGrader
│   ├── llm_rubric.py                      # LLMRubricGrader
│   ├── state_check.py                     # StateCheckGrader
│   ├── tool_calls.py                      # ToolCallsGrader
│   └── web3_validators.py                 # Web3 特定验证
├── evaluation/
│   ├── comprehensive_evaluator.py         # 新增 - ComprehensiveEvaluator
│   └── accuracy_analysis.py               # 修改 - Grader 级别分析
├── runners/
│   ├── agent_runner.py                    # 修改 - 解析 trajectory
│   └── multi_trial_runner.py              # 新增 - MultiTrialRunner
└── summarizers/
    └── default.py                         # 修改 - 新输出格式

examples/
├── genuine_agent_evaluation.py            # 新增 - 完整示例
├── multi_trial_evaluation.py              # 新增 - Multi-Trial 示例
└── custom_graders.py                      # 新增 - 自定义 Grader 示例

docs/
└── GENUINE_AGENT_EVAL.md                  # 新增 - 完整文档

configs/
├── comprehensive_config.json              # 新增 - 综合配置示例
└── graders_config.json                    # 新增 - Graders 配置示例
```

### D. 里程碑

| 里程碑 | 时间 | 交付物 | 验收标准 |
|--------|------|--------|----------|
| M1: 核心数据结构 | Week 2 | Trajectory, Trial, Outcome, TrackedMetrics | 数据类可用，解析测试通过 |
| M2: 四种 Graders | Week 4 | 4 个 Grader 实现 | 单元测试通过，与 EnhancedScoring/LLMEvaluator 结果一致 |
| M3: Multi-Trial | Week 5 | MultiTrialRunner, Pass@k | 可配置 1-10 次执行，Pass@k 计算正确 |
| M4: 综合评估器 | Week 6 | ComprehensiveEvaluator | 四 Graders 聚合，输出完整结果 |
| M5: 校准与分析 | Week 7 | Calibration 集成，分析扩展 | 校准测试通过，分析报告完整 |
| M6: 文档与示例 | Week 8 | 示例代码，API 文档 | 示例可运行，文档审核通过 |
| **v2.0 发布** | Week 9 | 完整功能 + 向后兼容 | 全部测试通过，无破坏性变更 |

### E. 向后兼容保证

v2.0 将保持与 v1.x 的完全向后兼容：

1. **旧 API 保留**:
   - `evaluate_agent()` 方法签名不变
   - `EvaluationResult` 结构保留
   - 现有配置文件格式支持

2. **渐进式采用**:
   - 默认使用简化模式（只用 deterministic_tests + llm_rubric）
   - 通过配置启用完整模式
   - trajectory 为可选

3. **输出兼容**:
   - 旧输出字段全部保留
   - 新字段为附加（不影响旧解析逻辑）

---

*此文档将随项目进展持续更新。*

*最后更新: 2026-01-13*
