# 从字符 GPT 走向指令微调

你已经验证了 next-token 训练、泛化 gap、dropout 和 checkpoint。下一步训练算法仍是反向传播；变化主要是起始权重、监督数据与评测目标。

| 方法 | 数据 | 更新参数 | 要验证什么 |
|---|---|---|---|
| 继续预训练 | 新领域文本 | 是 | 领域预测能力、原领域退化 |
| 指令 SFT | 需求与正确回答 | 是 | 未见任务是否做对 |
| LoRA | 同样的 SFT 数据 | 只训练低秩增量 | 更新比例、能力与保存恢复 |
| Prompt / few-shot | 推理提示及示例 | 否 | 无训练能否已满足任务 |

微调不能替代检索最新知识，也不能仅靠几条例子让弱基座获得复杂能力。当前字符 GPT 的 128 上下文是 128 字符，不能直接等同于 Qwen 的 128 BPE token。建议用 Qwen2.5-Coder-1.5B-Instruct 做小规模函数修复。

## 今天的学习动作

1. 打开 examples/repair_sft_demo.jsonl，手工解释每条需求、bug 和修复。两条仅展示格式，不作为正式训练集。
2. 运行结构检查：

```bash
python scripts/inspect_sft_data.py examples/repair_sft_demo.jsonl
```

3. 远端 agi-post 环境与真实 tokenizer 已准备好。进入实验目录、激活环境，运行离线检查：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate agi-post
cd /root/AItrainer/AI-trainer/experiments/repair-sft-20261027
python scripts/inspect_sft_data.py examples/repair_sft_demo.jsonl --tokenizer models/qwen2.5-coder-1.5b-tokenizer
```

已验证两条样例：180 总 token、31 有效监督 token、最长 95 token。记录模板全文及实际参与 loss 的文本，必须只有正确代码及模板结束标记；提示仍作为模型输入。tokenizer 来自 HF 官网固定 revision，本地获取后上传远端，未下载完整模型权重。

4. 先按来源题划分 train/dev/test，再从 train 来源制作约 30 条经可信测试的修复样例，检查全量划分：

```bash
python scripts/inspect_sft_data.py data/repair/train.jsonl data/repair/dev.jsonl data/repair/test.jsonl --tokenizer models/qwen2.5-coder-1.5b-tokenizer
```

此检查不执行代码、不能确认近重复或官方分组正确；这些需要 manifest 与沙盒测试。test 答案仅用于数据质量验证，不参与训练/调参，最终评分结果在方案冻结后查看。

## 用一条样例理解 loss

```text
完整 tokens = [提示1, 提示2, 回答1, 回答2, 结束符]
labels      = [-100,  -100, 回答1, 回答2, 结束符]

内部对齐：
logits[:, :-1] 对应 labels[:, 1:]
预测 回答1 的，是 提示2 所在位置的 logits

loss = (-log P(回答1|提示)
        -log P(回答2|提示,回答1)
        -log P(结束符|提示,回答1,回答2)) / 3
```

不预测提示目标并不意味着提示被删除或没有梯度影响；回答会通过注意力使用提示信息。loss mask 和 attention mask 必须分开理解。

## 方案验证顺序

先验证正确答案和评测器，再记录原模型/few-shot 基线，再做 16–32 条、10–20 更新步的 SFT 链路检查。链路通过后从原基座正式训练，用 dev 选 checkpoint，冷启动重载后统一评测。

如果原模型在 dev 已接近满分，应提升任务难度或增加边界题，不能以训练 loss 下降宣称有收益；dev 太低也先排查 prompt、解析和评测再扩大训练。train 改善而 dev 不改善，检查过拟合/数据分布；dev loss 下降但测试通过率不涨，检查答案语义、截断和评价一致性。

已完成远端环境、微型随机 Qwen 的手动 LoRA 更新/重载，以及 authored-v1 的 30/5/5 条 train/dev/test 沙盒与真实 tokenizer 检查，详见 repair_data_preparation.md。此自编小数据集用于链路验收，独立人工复核及扩大效果评测集仍待做；真实基座权重、SFTTrainer 实际 batch 验证与正式 GPU SFT 尚未完成。
