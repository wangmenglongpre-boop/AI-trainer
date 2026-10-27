# authored-v1 数据验收

远端路径：`/root/AItrainer/AI-trainer/experiments/repair-sft-20261027`。
环境：`agi-post`。本地与远端的 `data/repair` 保留相同产物。

| 集合 | 来源题 | 修复样本 | 总 token | 回答监督 token | 最长样本 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 10 | 30 | 4671 | 1152 | 237 |
| dev | 5 | 5 | 816 | 207 | 207 |
| test | 5 | 5 | 900 | 295 | 260 |

来源：本次 AI 辅助自编的 20 个操作家族，不是 MBPP，不含外部数据集。
train 包含稳定去重、闭区间求和、分块、元音计数、括号平衡、游程编码、
单层展开、转置、区间合并、首位置二分查找。dev/test 使用另外的来源题。
每个 train 来源有 3 个可解释错误变体；dev/test 每题各 1 个。

先固定来源划分，再展开变体。全部参考答案通过 133 个显式期望测试，
40 个错误版本都失败至少一个公开测试。期望值不是运行参考代码生成的。
未发现跨集合精确 prompt、规范化 AST 重复；跨集合 AST 相似度 >=0.75
候选数为 0。语义检查没有发现同题换变量名跨集合，但相似度不是泄漏证明。

只加载 train.jsonl 做 SFT，dev.jsonl 做验证。test.jsonl 为审计答案，
不用于训练或选模型。推理用 dev_prompts.jsonl/test_prompts.jsonl。
隐藏测试在 evaluator_cases.json，不进入 messages；评分在容器外完成。
公开失败摘要来自实际 Docker 运行，而非猜测。

真实离线 Qwen tokenizer 检查全量 40 条：提示 labels=-100，
回答和 `<|im_end|>` 及模板尾换行参与监督；无重复特殊 token，
无 1024 上下文截断。尚未验证实际 Trainer collator/padding。

Docker 探针通过：UID=65534、只读根目录、禁网（只有 lo）、capabilities
清零、no-new-privileges；无限循环在外部超时后清理，无残留检查容器。
完整配置与镜像摘要见 specs/repair-sft/design.md 和验证 JSON。
该实例 Docker run 输出连接失效，使用 create/start/wait/logs 读取；
候选代码从未在宿主 Python 执行。当前 harness 用于检查过的自编代码，
任意模型输出评分仍需加强候选进程与输出协议隔离。

复核入口：

```bash
python scripts/inspect_sft_data.py data/repair/train.jsonl data/repair/dev.jsonl data/repair/test.jsonl --tokenizer models/qwen2.5-coder-1.5b-tokenizer
```

机器记录：repair_data_verification.json、repair_token_audit.json、
repair_sandbox_probe.json；每个文件都有对应运行日志。数据 manifest 保存
来源、固定划分、镜像身份和 SHA256。源目录 configs/repair_source_catalog.json
与 build/verify 脚本支持重建。

限制：自编测试仍需独立人工审阅，有限测试不是正确性证明；预训练污染未知。
test 只有 5 个来源，一题影响 20 个百分点。本版用于微调链路验收，
不作为正式效果证据。下一步检查真实 Trainer batch、运行原模型 dev 基线，
再决定是否增加难度或进入短程训练。模型权重下载和正式 SFT 尚未执行。
