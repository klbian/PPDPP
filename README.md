# PPDPP

## Re-DocRed 关系融合扩展

仓库新增了一个不依赖 `torch`/`transformers` 的文档级关系抽取融合管线：`debate_fusion.py`。
它借鉴 PPDPP 的“策略选择器 + 环境 + 动作空间”思路，把 LLM 结果和 SLM 结果视为两个智能体的候选关系集合，
再由策略选择器在每个实体对上选择动作，输出最终三元组集合。

### 设计要点

- **智能体 A（LLM）**：持有候选集合 `C`，更偏向召回，负责提出可能漏掉的关系。
- **智能体 B（SLM）**：持有候选集合 `D`，更偏向精确率，负责过滤明显噪声。
- **策略选择器**：在实体对粒度上，从 `none / inter / slm / llm / union / top-k / calibrated / whitelist-augmented` 等动作中选择一个动作。
- **环境**：`RelationDebateEnv` 负责加载 `test_revised.json`、`predict-LLM-converted.json`、`results-SLM.json`，并构造 pair-wise debate case。
- **奖励函数**：
  - 主奖励：`0.7 * pair_f1 + 0.15 * precision + 0.15 * recall`
  - 额外奖励：精确匹配金标准关系集合时加 `0.25`
  - 惩罚项：预测关系数过多时，按 `0.02 * max(0, len(pred)-max(1, len(gold)))` 处罚
- **策略训练**：
  - 对每个实体对枚举动作，计算局部奖励，得到 oracle action。
  - 将 oracle action 写入实例记忆（instance memory）与多粒度签名规则表（full / coarse / minimal）。
  - 推理时优先走实例记忆，否则逐级回退到签名规则。

### Prompt 模板

提示词定义在 `debate_prompt.py`：

- `build_agent_a_prompt`：只暴露 LLM 候选，强调召回。
- `build_agent_b_prompt`：只暴露 SLM 候选，强调精确率。
- `build_selector_prompt`：暴露 pair 状态与动作空间，让 selector 按 pair-level F1 选择动作。

### 运行方式

```bash
python debate_fusion.py \
  --gold_path test_revised.json \
  --llm_path predict-LLM-converted.json \
  --slm_path results-SLM.json \
  --rel_info_path rel_info.json \
  --output_path fusion-results.json \
  --policy_path fusion-policy.json
```

如需关闭实例记忆、只使用签名策略，可增加：

```bash
python debate_fusion.py --disable_instance_memory
```

### 输出文件

- `fusion-results.json`：融合后的关系三元组结果
- `fusion-policy.json`：动作使用统计、策略信息与评估指标
