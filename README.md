# PPDPP

本仓库在原始 PPDPP 对话策略框架之外，补充了一个 **DocRE 文档级关系抽取结果融合** 管线，复用了 PPDPP 的“环境 + 动作 + 策略选择器”思想：

- 智能体 **A** 持有 LLM 预测集合 `C`。
- 智能体 **B** 持有 SLM 预测集合 `D`。
- 策略选择器逐个查看实体对上的候选关系 `C ∪ D`，并在辩论环境中选择动作。
- 默认提供一个 **可直接运行的经验价值 selector**；如果环境里安装了 `torch + transformers`，也预留了 **BERT selector** 接口。

## DocRE 融合设计

### 1. 动作空间

对于一个实体对 `(h, t)`，把 `C ∪ D` 中的关系按顺序送入环境。策略选择器的动作空间为：

- `ACCEPT`：把当前关系加入最终输出集合。
- `REJECT`：丢弃当前关系。
- `DEFER`：暂缓判断，把当前关系放回队尾。
- `STOP`：结束该实体对的辩论。

### 2. 智能体提示词

仓库里提供了可直接序列化给 LLM / BERT 的提示模板：

- `DebatePrompts.AGENT_TEMPLATE`
  - 智能体 A 只能看见 LLM 候选集合。
  - 智能体 B 只能看见 SLM 候选集合。
  - 输出固定为 `stance / evidence / suggestion` 三段。
- `DebatePrompts.SELECTOR_TEMPLATE`
  - 选择器综合 A/B 的陈述、已接受关系、剩余候选数和当前关系，输出一个动作。

### 3. 环境

`DocREFusionEnv` 把每个实体对视为一个 episode：

- `state`
  - 文档上下文
  - 当前实体对及其类型
  - 当前讨论关系
  - 已接受 / 已拒绝关系集合
  - 历史动作
- `transition`
  - 根据 selector 动作更新 `accepted / rejected / queue`
- `termination`
  - 队列为空，或策略选择器输出 `STOP`

### 4. 奖励函数

使用 **pair-level F1 增量奖励**：

- `reward_t = F1(S_t, G) - F1(S_{t-1}, G)`
  - `S_t`：当前已接受关系集合
  - `G`：该实体对的 gold 关系集合
- 如果 episode 结束时 `S_t == G`，额外给一个小的 exact-match bonus。
- `DEFER` 会有轻微惩罚，避免 selector 无意义拖延。

### 5. 策略选择器

默认可运行版本：`EmpiricalValueSelector`

- 用有标注数据对 debate state 做离线拟合。
- 统计维度：
  - 来源一致性（A / B / AB）
  - 关系类型
  - 实体类型对
  - SLM 分数桶
- 通过 back-off 加权估计“接受当前关系的价值”，再映射成 `ACCEPT / REJECT / STOP` 动作。

可选版本：`BertDebateSelector`

- 与经验 selector 共用同一套环境、状态序列化和 prompt。
- 若安装 `torch` 与 `transformers`，可把 selector 替换为 BERT 序列分类器，对 debate state 直接预测动作分布。

## 运行方式

```bash
python run_docre_fusion.py \
  --gold DocRE_dataset/test_revised.json \
  --llm predict-LLM-converted.json \
  --slm results-SLM.json \
  --output fused-docre-results.json
```

输出文件中包含：

- `config`
- `metrics`（LLM / SLM / UNION / FUSION）
- `predictions`

在当前提供的数据上，默认经验 selector 的融合 F1 会高于简单的 `LLM + SLM` 并集基线，并略高于单独使用 SLM。
