# Environment-Controlled Trajectory Matching 方案

> 搭建受控环境 + 在 dataset 中定义 expected tool calls + 用 rule-based 精确匹配 agent 实际 trajectory 中的关键节点。

**Date**: 2026-02-06  
**Status**: 方案设计

---

## 1. 核心思路

### 1.1 与当前方案的区别

```
当前方案（v1/v2）:
  Agent ──[真实环境]──→ 最终答案 ──[LLM/Rule]──→ 分数
                          ↑
                    评估的是"结果对不对"

本方案:
  Agent ──[受控环境]──→ Trajectory ──[Rule-Based 精确匹配]──→ 分数
                          ↑
                    评估的是"过程对不对"
```

**关键差异:**

| 维度 | 当前方案 | 本方案 |
|------|---------|--------|
| 环境 | 真实 API / 真实区块链 | 受控 mock 环境（DB / Mock API） |
| 评估对象 | 最终答案文本 | Trajectory 中的 tool call 序列 |
| 匹配方式 | 语义/模糊匹配 + LLM Judge | **精确匹配（rule-based）** |
| Ground Truth | expected_answer（一个字符串） | expected_tool_calls（一组结构化的工具调用） |
| 确定性 | 非确定性（LLM 评分波动） | **完全确定性**（通过 / 不通过） |
| 成本 | 需要 LLM API 调用 | **零额外成本** |
| 适用场景 | 开放式问答 | 有明确操作路径的任务 |

### 1.2 类比

这个思路本质上就是 **τ-bench / SWE-bench 的 Web3 版本**:

- **SWE-bench**: 搭建 git repo 环境 → agent 修复代码 → 跑单元测试（确定性 pass/fail）
- **τ-bench**: 搭建零售/航空数据库 → agent 处理客服请求 → 检查数据库最终状态
- **本方案**: 搭建链上数据库 + Mock API → agent 执行 Web3 任务 → 检查 tool call 序列 + 环境状态

---

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Trajectory Matching Eval System                    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Environment Layer                           │   │
│  │                                                                │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │   │
│  │  │ MockDB   │  │ Mock     │  │ Mock     │  │ Mock         │ │   │
│  │  │ (chain   │  │ Web API  │  │ Block    │  │ DeFi         │ │   │
│  │  │  state,  │  │ (search, │  │ Explorer │  │ Protocol     │ │   │
│  │  │  tokens, │  │  news,   │  │ (Ether-  │  │ (Uniswap,   │ │   │
│  │  │  txs)    │  │  docs)   │  │  scan)   │  │  Aave, etc.) │ │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │   │
│  │                                                                │   │
│  │  ToolRouter: tool_name → mock handler                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│          ↕ (agent 调用 tools, 从 mock 获取数据)                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Agent Under Test                            │   │
│  │  Agent 在受控环境中执行任务, 产出 Trajectory                     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│          ↓ (产出 Trajectory: tool calls + reasoning + answer)        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Matching Layer                              │   │
│  │                                                                │   │
│  │  ┌─────────────────┐  ┌─────────────────┐                    │   │
│  │  │ ToolCallMatcher │  │ StateMatcher    │                    │   │
│  │  │ (精确匹配 tool   │  │ (检查环境最终   │                    │   │
│  │  │  name + params) │  │  状态)          │                    │   │
│  │  └─────────────────┘  └─────────────────┘                    │   │
│  │  ┌─────────────────┐  ┌─────────────────┐                    │   │
│  │  │ SequenceMatcher │  │ AnswerMatcher   │                    │   │
│  │  │ (检查调用顺序)   │  │ (最终答案校验)  │                    │   │
│  │  └─────────────────┘  └─────────────────┘                    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│          ↓                                                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  MatchResult                                                   │   │
│  │  {                                                             │   │
│  │    tool_call_score,         # 关键 tool call 是否全部命中      │   │
│  │    sequence_score,          # 调用顺序是否正确                  │   │
│  │    state_score,             # 环境最终状态是否正确              │   │
│  │    answer_score,            # 最终答案是否正确                  │   │
│  │    overall_pass: bool       # 综合是否通过                     │   │
│  │  }                                                             │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 三层职责

| 层 | 职责 | 设计要点 |
|----|------|---------|
| **Environment Layer** | 为 agent 提供确定性的数据源 | Agent 调用 tool 时，不走真实网络，走 mock handler |
| **Agent Under Test** | 在受控环境中执行任务 | Agent 不知道自己在 mock 环境中（对 agent 透明） |
| **Matching Layer** | 将 agent trajectory 与 expected 精确比对 | 全部 rule-based，零 LLM 调用 |

---

## 3. Dataset 格式设计

### 3.1 扩展后的 CSV / JSON 格式

当前 dataset 只有 `question` + `answer`。本方案需要增加：

```json
{
  "id": "onchain_001",
  "question": "查询地址 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045 在 ETH 主网上的余额",
  "category": "onchain_retrieval",
  "task_type": "balance_query",

  "expected_answer": "1234.56 ETH",

  "expected_tool_calls": [
    {
      "tool_name": "get_balance",
      "params": {
        "address": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
        "chain_id": 1
      },
      "match_mode": "exact",
      "required": true
    }
  ],

  "expected_sequence": ["get_balance"],

  "expected_state": {
    "queried_address": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
    "chain": "ethereum"
  },

  "environment": {
    "preset": "eth_mainnet_snapshot_2024q4",
    "seed_data": {
      "balances": {
        "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045": "1234560000000000000000"
      }
    }
  }
}
```

### 3.2 字段定义

| 字段 | 类型 | 说明 |
|------|------|------|
| `expected_tool_calls` | `List[ToolCallSpec]` | 期望 agent 执行的关键 tool calls |
| `expected_tool_calls[].tool_name` | `str` | 工具名称，精确匹配 |
| `expected_tool_calls[].params` | `Dict` | 工具参数，按 match_mode 匹配 |
| `expected_tool_calls[].match_mode` | `str` | `exact` / `subset` / `contains` |
| `expected_tool_calls[].required` | `bool` | true = 必须出现；false = 可选加分 |
| `expected_sequence` | `List[str]` | 期望的工具调用顺序（仅 required 的） |
| `expected_state` | `Dict` | 期望的环境最终状态 |
| `environment` | `Dict` | 环境配置：预设 + 种子数据 |

### 3.3 match_mode 说明

```python
# exact: 参数完全一致
{"tool_name": "get_balance", "params": {"address": "0xabc...", "chain_id": 1}}
# agent 必须调用 get_balance(address="0xabc...", chain_id=1)

# subset: agent 的参数是 expected 的超集（允许额外参数）
{"tool_name": "web_search", "params": {"query": "SEC Bitcoin ETF"}, "match_mode": "subset"}
# agent 调用 web_search(query="SEC Bitcoin ETF", language="en") → 通过

# contains: 参数值包含关系（适用于搜索类）
{"tool_name": "web_search", "params": {"query": "Bitcoin ETF"}, "match_mode": "contains"}
# agent 调用 web_search(query="SEC approves Bitcoin ETF 2024") → 通过（包含 "Bitcoin ETF"）
```

### 3.4 复合任务示例

```json
{
  "id": "defi_003",
  "question": "查询 Uniswap V3 上 ETH/USDC 池的当前价格和 24h 交易量",
  "category": "onchain_retrieval",
  "task_type": "defi_pool_query",

  "expected_answer": "ETH/USDC 价格: $3,245.67, 24h 交易量: $12.5M",

  "expected_tool_calls": [
    {
      "tool_name": "get_pool_info",
      "params": {
        "protocol": "uniswap_v3",
        "token0": "ETH",
        "token1": "USDC"
      },
      "match_mode": "exact",
      "required": true
    },
    {
      "tool_name": "get_pool_volume",
      "params": {
        "pool_address": "0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8",
        "period": "24h"
      },
      "match_mode": "subset",
      "required": true
    }
  ],

  "expected_sequence": ["get_pool_info", "get_pool_volume"],

  "environment": {
    "preset": "defi_snapshot_2024q4",
    "seed_data": {
      "pools": {
        "0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8": {
          "token0": "WETH",
          "token1": "USDC",
          "price": "3245.67",
          "volume_24h": "12500000",
          "tvl": "250000000"
        }
      }
    }
  }
}
```

---

## 4. Environment Layer 设计

### 4.1 Mock 环境架构

```
EnvironmentManager
  │
  ├── ToolRouter (tool_name → handler)
  │    将 agent 的 tool call 拦截并路由到对应 mock handler
  │
  ├── MockDB (内存数据库)
  │    ├── balances: Dict[address, wei_amount]
  │    ├── transactions: List[TxRecord]
  │    ├── contracts: Dict[address, ContractInfo]
  │    ├── tokens: Dict[address, TokenInfo]
  │    ├── pools: Dict[address, PoolInfo]
  │    └── web_content: Dict[url, content]
  │
  ├── MockHandlers (每个 tool 对应一个 handler)
  │    ├── get_balance(address, chain_id) → 从 MockDB 查询
  │    ├── get_transaction(tx_hash) → 从 MockDB 查询
  │    ├── contract_read(address, method, args) → 从 MockDB 查询
  │    ├── web_search(query) → 从 MockDB.web_content 匹配
  │    ├── get_pool_info(protocol, token0, token1) → 从 MockDB 查询
  │    └── ...
  │
  └── StateTracker (记录环境变化)
       记录所有 tool call + 返回值，用于最终 state 校验
```

### 4.2 核心类设计

```python
@dataclass
class EnvironmentConfig:
    """环境配置，从 dataset 的 environment 字段加载"""
    preset: str                          # 预设环境名
    seed_data: Dict[str, Any]            # 种子数据
    available_tools: List[str] = None    # 可用工具列表（None = 全部）


class MockDB:
    """内存数据库，存放受控环境数据"""

    def __init__(self, seed_data: Dict[str, Any]):
        self.balances = seed_data.get("balances", {})
        self.transactions = seed_data.get("transactions", {})
        self.contracts = seed_data.get("contracts", {})
        self.tokens = seed_data.get("tokens", {})
        self.pools = seed_data.get("pools", {})
        self.web_content = seed_data.get("web_content", {})
        self.blocks = seed_data.get("blocks", {})

    def query(self, collection: str, key: str) -> Any:
        store = getattr(self, collection, {})
        return store.get(key)


class ToolRouter:
    """拦截 agent 的 tool call，路由到 mock handler"""

    def __init__(self, db: MockDB, available_tools: List[str] = None):
        self.db = db
        self.call_log: List[Dict[str, Any]] = []  # 记录所有调用
        self.handlers = self._register_handlers()
        self.available_tools = set(available_tools) if available_tools else None

    def handle(self, tool_name: str, params: Dict[str, Any]) -> Any:
        if self.available_tools and tool_name not in self.available_tools:
            raise ToolNotAvailableError(tool_name)

        handler = self.handlers.get(tool_name)
        if not handler:
            raise UnknownToolError(tool_name)

        result = handler(params)
        self.call_log.append({
            "tool_name": tool_name,
            "params": params,
            "result": result,
            "success": True,
        })
        return result

    def _register_handlers(self) -> Dict[str, Callable]:
        return {
            "get_balance": self._handle_get_balance,
            "get_transaction": self._handle_get_transaction,
            "contract_read": self._handle_contract_read,
            "web_search": self._handle_web_search,
            "get_pool_info": self._handle_get_pool_info,
            "get_pool_volume": self._handle_get_pool_volume,
            "get_token_info": self._handle_get_token_info,
            "get_block": self._handle_get_block,
        }

    def _handle_get_balance(self, params):
        address = params["address"].lower()
        wei = self.db.balances.get(address, "0")
        return {"balance_wei": wei, "balance_eth": str(int(wei) / 1e18)}

    def _handle_web_search(self, params):
        query = params.get("query", "").lower()
        results = []
        for url, content in self.db.web_content.items():
            if any(word in content.lower() for word in query.split()):
                results.append({"url": url, "snippet": content[:200]})
        return {"results": results[:5]}

    # ... 其他 handlers 同理
```

### 4.3 Environment Presets（预设环境）

为不同任务类别准备预设环境模板：

```
environments/
├── presets/
│   ├── eth_mainnet_snapshot_2024q4.json    # ETH 主网快照
│   ├── defi_snapshot_2024q4.json           # DeFi 协议数据快照
│   ├── nft_market_snapshot_2024q4.json     # NFT 市场数据
│   ├── web3_news_2024.json                 # Web3 新闻语料
│   └── sec_filings_2024.json              # SEC 文件
├── schemas/
│   └── environment_schema.json            # 环境数据 schema
└── generators/
    └── snapshot_generator.py              # 从链上抓快照 → preset
```

每个 preset 是一个 JSON 文件：

```json
{
  "preset_name": "eth_mainnet_snapshot_2024q4",
  "chain_id": 1,
  "block_number": 19000000,
  "timestamp": "2024-12-01T00:00:00Z",
  "balances": {
    "0xd8da6bf26964af9d7eed9e03e53415d37aa96045": "1234560000000000000000",
    "0x...": "..."
  },
  "transactions": {
    "0xabc...": {
      "from": "0x...", "to": "0x...",
      "value": "1000000000000000000", "block_number": 19000000
    }
  },
  "tokens": {
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": {
      "symbol": "USDC", "decimals": 6, "name": "USD Coin"
    }
  },
  "web_content": {
    "https://sec.gov/bitcoin-etf": "SEC approves Bitcoin spot ETF on January 10, 2024..."
  }
}
```

---

## 5. Matching Layer 设计

### 5.1 四种 Matcher

```python
@dataclass
class ToolCallSpec:
    """dataset 中定义的一个 expected tool call"""
    tool_name: str
    params: Dict[str, Any]
    match_mode: str = "exact"    # exact / subset / contains
    required: bool = True


@dataclass
class MatchResult:
    """单个任务的匹配结果"""
    task_id: str

    # 四维分数
    tool_call_score: float       # 关键 tool call 命中率
    sequence_score: float        # 调用顺序正确率
    state_score: float           # 环境状态匹配率
    answer_score: float          # 最终答案匹配

    # 综合
    overall_score: float         # 加权综合分
    overall_pass: bool           # 是否通过

    # 明细
    matched_calls: List[str]     # 命中的 tool calls
    missing_calls: List[str]     # 缺失的 required tool calls
    extra_calls: List[str]       # 多余的 tool calls
    sequence_alignment: str      # 序列对齐详情
    state_diffs: Dict[str, Any]  # 状态差异
```

### 5.2 ToolCallMatcher — 精确匹配工具调用

```python
class ToolCallMatcher:
    """检查 agent trajectory 中的 tool calls 是否与 expected 匹配"""

    def match(
        self,
        actual_calls: List[Step],       # trajectory 中的 tool_call steps
        expected_calls: List[ToolCallSpec],
    ) -> Dict[str, Any]:

        matched = []
        missing = []

        for spec in expected_calls:
            found = self._find_matching_call(spec, actual_calls)
            if found:
                matched.append(spec.tool_name)
            elif spec.required:
                missing.append(spec.tool_name)

        required_total = sum(1 for s in expected_calls if s.required)
        required_matched = required_total - len(missing)

        return {
            "score": required_matched / required_total if required_total > 0 else 1.0,
            "matched": matched,
            "missing": missing,
            "extra": self._find_extra_calls(actual_calls, expected_calls),
        }

    def _find_matching_call(self, spec: ToolCallSpec, actuals: List[Step]) -> bool:
        for step in actuals:
            if step.tool_name != spec.tool_name:
                continue

            if spec.match_mode == "exact":
                if self._params_exact_match(spec.params, step.tool_input):
                    return True
            elif spec.match_mode == "subset":
                if self._params_subset_match(spec.params, step.tool_input):
                    return True
            elif spec.match_mode == "contains":
                if self._params_contains_match(spec.params, step.tool_input):
                    return True
        return False

    def _params_exact_match(self, expected: Dict, actual: Dict) -> bool:
        """参数键值完全一致（忽略大小写 for addresses）"""
        if set(expected.keys()) != set(actual.keys()):
            return False
        for key in expected:
            ev = _normalize_param(expected[key])
            av = _normalize_param(actual.get(key))
            if ev != av:
                return False
        return True

    def _params_subset_match(self, expected: Dict, actual: Dict) -> bool:
        """expected 的每个键值都在 actual 中（actual 可以有额外字段）"""
        for key, ev in expected.items():
            av = actual.get(key)
            if av is None or _normalize_param(ev) != _normalize_param(av):
                return False
        return True

    def _params_contains_match(self, expected: Dict, actual: Dict) -> bool:
        """expected 的每个值是 actual 对应值的子串"""
        for key, ev in expected.items():
            av = actual.get(key)
            if av is None:
                return False
            if str(ev).lower() not in str(av).lower():
                return False
        return True


def _normalize_param(value: Any) -> Any:
    """参数归一化：地址转小写等"""
    if isinstance(value, str):
        v = value.strip()
        if v.startswith("0x"):
            return v.lower()
        return v
    return value
```

### 5.3 SequenceMatcher — 调用顺序检查

```python
class SequenceMatcher:
    """检查 required tool calls 的调用顺序是否符合 expected_sequence"""

    def match(
        self,
        actual_calls: List[Step],
        expected_sequence: List[str],
    ) -> Dict[str, Any]:
        if not expected_sequence:
            return {"score": 1.0, "detail": "no sequence constraint"}

        actual_tool_names = [
            s.tool_name for s in actual_calls
            if s.step_type == StepType.TOOL_CALL and s.tool_name
        ]

        # 提取 actual 中属于 expected_sequence 的子序列
        filtered = [t for t in actual_tool_names if t in expected_sequence]

        # 计算最长公共子序列 (LCS) 来衡量顺序正确性
        lcs_len = self._lcs_length(filtered, expected_sequence)
        score = lcs_len / len(expected_sequence)

        return {
            "score": score,
            "expected": expected_sequence,
            "actual_filtered": filtered,
            "lcs_length": lcs_len,
        }

    def _lcs_length(self, a: List[str], b: List[str]) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
```

### 5.4 StateMatcher — 环境状态校验

```python
class StateMatcher:
    """比对 ToolRouter.call_log 产生的环境状态与 expected_state"""

    def match(
        self,
        actual_state: Dict[str, Any],    # ToolRouter 产出的状态
        expected_state: Dict[str, Any],   # dataset 中定义的
    ) -> Dict[str, Any]:
        if not expected_state:
            return {"score": 1.0, "diffs": {}}

        diffs = {}
        matched = 0
        total = len(expected_state)

        for key, expected_val in expected_state.items():
            actual_val = actual_state.get(key)
            ev = _normalize_param(expected_val)
            av = _normalize_param(actual_val)
            if ev == av:
                matched += 1
            else:
                diffs[key] = {"expected": expected_val, "actual": actual_val}

        return {
            "score": matched / total if total > 0 else 1.0,
            "diffs": diffs,
        }
```

### 5.5 综合评分

```python
class TrajectoryMatchEvaluator:
    """综合四个 matcher 的结果"""

    DEFAULT_WEIGHTS = {
        "tool_call": 0.40,    # 最重要：关键 tool call 是否命中
        "sequence": 0.20,     # 顺序是否正确
        "state": 0.20,        # 环境状态是否正确
        "answer": 0.20,       # 最终答案是否正确
    }

    def evaluate(self, task, trajectory, environment_state) -> MatchResult:
        # 1. Tool call matching
        tool_result = self.tool_matcher.match(
            actual_calls=trajectory.tool_call_steps,
            expected_calls=task.expected_tool_calls,
        )

        # 2. Sequence matching
        seq_result = self.sequence_matcher.match(
            actual_calls=trajectory.tool_call_steps,
            expected_sequence=task.expected_sequence,
        )

        # 3. State matching
        state_result = self.state_matcher.match(
            actual_state=environment_state,
            expected_state=task.expected_state,
        )

        # 4. Answer matching (exact / contains)
        answer_score = 1.0 if task.expected_answer.lower() in trajectory.final_answer.lower() else 0.0

        # Weighted score
        overall = (
            tool_result["score"] * self.weights["tool_call"]
            + seq_result["score"] * self.weights["sequence"]
            + state_result["score"] * self.weights["state"]
            + answer_score * self.weights["answer"]
        )

        return MatchResult(
            task_id=task.id,
            tool_call_score=tool_result["score"],
            sequence_score=seq_result["score"],
            state_score=state_result["score"],
            answer_score=answer_score,
            overall_score=overall,
            overall_pass=overall >= 0.8 and len(tool_result["missing"]) == 0,
            matched_calls=tool_result["matched"],
            missing_calls=tool_result["missing"],
            extra_calls=tool_result.get("extra", []),
            sequence_alignment=str(seq_result),
            state_diffs=state_result["diffs"],
        )
```

---

## 6. 执行流程

### 6.1 单任务执行流程

```
1. 加载 Task（从 dataset）
   └─ 解析 expected_tool_calls, expected_sequence, expected_state, environment

2. 初始化 Environment
   ├─ 加载 preset（如 eth_mainnet_snapshot_2024q4.json）
   ├─ 合并 task.environment.seed_data（覆盖/追加）
   ├─ 创建 MockDB
   └─ 创建 ToolRouter

3. 执行 Agent
   ├─ 将 question 发送给 agent
   ├─ Agent 调用 tool → ToolRouter 拦截 → 返回 mock 数据
   ├─ Agent 持续推理直到产出最终答案
   └─ 收集完整 Trajectory

4. 匹配评估
   ├─ ToolCallMatcher.match(trajectory.tool_calls, expected_tool_calls)
   ├─ SequenceMatcher.match(trajectory.tool_calls, expected_sequence)
   ├─ StateMatcher.match(tool_router.state, expected_state)
   └─ AnswerMatcher.match(trajectory.answer, expected_answer)

5. 输出 MatchResult
```

### 6.2 与现有 DAB 架构的集成

本方案不替代现有架构，而是作为一种**新的 evaluation_method**:

```python
# dataset CSV 中增加一种 evaluation_method
evaluation_method = "trajectory_match"

# EvaluationEngine 新增路由
def select_evaluation_method(self, category, method_override=None):
    if method_override == "trajectory_match":
        return EvaluationMethod.TRAJECTORY_MATCH
    # ... 原有逻辑
```

```
EvaluationMethod (扩展):
  ├── RULE_BASED          # 现有
  ├── LLM_BASED           # 现有
  ├── HYBRID              # 现有
  ├── CASCADE             # 现有
  └── TRAJECTORY_MATCH    # 新增
```

---

## 7. 适用场景分析

### 7.1 适合的任务类型

| 任务类型 | 适合度 | 原因 |
|---------|:------:|------|
| 余额查询 | ★★★★★ | 操作路径唯一确定 |
| 交易查询 | ★★★★★ | 操作路径唯一确定 |
| 合约状态读取 | ★★★★★ | 操作路径唯一确定 |
| DeFi 池查询 | ★★★★☆ | 路径较确定，可能有替代 API |
| Token 信息查询 | ★★★★☆ | 路径较确定 |
| 事实性问答（有明确来源） | ★★★★☆ | web_search 参数可用 contains 匹配 |
| 多步推理任务 | ★★★☆☆ | 路径可能有多种等价方式 |
| 开放性分析 | ★★☆☆☆ | 没有唯一正确路径 |
| 创意/建议类 | ★☆☆☆☆ | 不适合精确匹配 |

### 7.2 与其他方法的互补关系

```
                    操作路径确定性
                   高 ──────────────── 低
                    │                  │
  TRAJECTORY_MATCH  │ ████████         │
                    │                  │
  RULE_BASED        │   ██████████     │
                    │                  │
  HYBRID            │      ████████████│
                    │                  │
  LLM_BASED         │         █████████│
                    │                  │
```

**最佳实践：** 在 dataset 中为不同任务选择不同的 evaluation_method，而不是一刀切。

---

## 8. 与方案优劣分析

### 优势

| 优势 | 说明 |
|------|------|
| **完全确定性** | 没有 LLM judge 的随机性，结果 100% 可复现 |
| **零额外成本** | 不需要调用 LLM API 做评估 |
| **极速执行** | 匹配逻辑是纯字符串比较，毫秒级 |
| **可调试性强** | missing_calls / extra_calls 精确定位 agent 的错误步骤 |
| **环境可控** | 不受外部 API 变化影响，永远一致 |
| **针对 Web3** | 地址/交易格式天然适合精确匹配 |

### 局限

| 局限 | 缓解策略 |
|------|---------|
| **只适合路径确定的任务** | 与 HYBRID/LLM_BASED 方法互补 |
| **需要手工标注 expected_tool_calls** | 先从简单任务开始；可通过专家 agent 自动生成 |
| **不容忍等价替代路径** | 支持多种 expected_tool_calls 变体（见下方 8.1） |
| **环境数据需要维护** | 使用 snapshot_generator 从链上定期抓取 |

### 8.1 处理等价路径

某些任务可能有多种正确路径。方案支持定义多组 expected：

```json
{
  "expected_tool_calls_variants": [
    {
      "description": "方式1: 直接查余额",
      "tool_calls": [
        {"tool_name": "get_balance", "params": {"address": "0x..."}, "match_mode": "exact"}
      ]
    },
    {
      "description": "方式2: 通过 Etherscan API",
      "tool_calls": [
        {"tool_name": "etherscan_api", "params": {"module": "account", "action": "balance", "address": "0x..."}, "match_mode": "exact"}
      ]
    }
  ]
}
```

匹配时取最高分的 variant：

```python
scores = [self.match_variant(trajectory, variant) for variant in variants]
best_score = max(scores)
```

---

## 9. 文件结构

```
agent_trial_bench/
├── trajectory_match/                    # 新增模块
│   ├── __init__.py
│   ├── environment.py                   # MockDB, ToolRouter, EnvironmentManager
│   ├── matchers.py                      # ToolCallMatcher, SequenceMatcher, StateMatcher
│   ├── evaluator.py                     # TrajectoryMatchEvaluator (综合评估)
│   ├── dataset_loader.py               # 加载扩展格式的 dataset
│   └── handlers/                        # Mock tool handlers
│       ├── __init__.py
│       ├── blockchain.py                # get_balance, get_transaction, get_block
│       ├── defi.py                      # get_pool_info, get_pool_volume
│       ├── web.py                       # web_search, document_fetch
│       └── token.py                     # get_token_info, get_nft_info
│
├── trajectory.py                        # 现有，无需修改
├── graders/                             # 现有 v2 graders
└── ...

environments/
├── presets/
│   ├── eth_mainnet_snapshot_2024q4.json
│   ├── defi_snapshot_2024q4.json
│   └── web3_news_2024.json
├── schemas/
│   └── environment_schema.json
└── generators/
    └── snapshot_generator.py

data/
├── benchmark.csv                        # 现有 dataset
└── trajectory_match/                    # 新增
    ├── onchain_tasks.json               # 链上查询任务集
    ├── defi_tasks.json                  # DeFi 任务集
    └── web3_fact_qa.json               # Web3 事实问答任务集
```

---

## 10. 实施路线

| Phase | 时间 | 交付物 |
|-------|------|--------|
| **P1: 基础框架** | Week 1-2 | MockDB, ToolRouter, ToolCallMatcher, SequenceMatcher |
| **P2: 环境数据** | Week 2-3 | 3 个 preset, snapshot_generator, 10 个标注任务 |
| **P3: 集成** | Week 3-4 | 接入 EvaluationEngine, TRAJECTORY_MATCH 路由 |
| **P4: 扩展** | Week 4-5 | 50 个标注任务, 等价路径支持, StateMatcher |

---

*此方案可与现有 v1/v2 架构并行推进，互不阻塞。*
