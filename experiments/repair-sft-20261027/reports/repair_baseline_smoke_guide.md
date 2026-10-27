# 完整基座 → 原模型 dev → 短程 LoRA SFT

协作方式：代码在本地准备、核验，由你在远端终端执行。不会代你启动远端
模型评测或训练。本轮不使用 test，不继续此前的小型随机模型参数。

## 1. 本次代码

| 文件 | 职责 |
| --- | --- |
| scripts/download_repair_base.py | 下载固定官网版本，支持断点/分段下载，校验官方权重 SHA256 |
| scripts/repair_model_common.py | 统一模型身份、模板、数据边界、实际 batch 检查与训练配置 |
| scripts/repair_candidate_sandbox.py | 将生成代码放入 Docker 子进程执行；期望值在容器外评分 |
| scripts/eval_repair_dev.py | 原模型或 adapter 的同协议 dev 评测，保存所有生成和状态 |
| scripts/smoke_repair_sft.py | 10–20 次参数更新；检查梯度、基座冻结、参数变化及保存重载 |
| scripts/check_repair_pipeline_local.py | 本地 CPU 小型模型和 Docker 的集成验收，不是正式 GPU 结果 |

基座：Qwen/Qwen2.5-Coder-1.5B-Instruct。
固定 revision：2e1fd397ee46e1388853d2af2c993145b0f1098a，与已有 tokenizer 一致。
完整权重文件 model.safetensors 为 3,087,467,144 字节（约 3.09 GB）。
本地已于 2026-10-27 下载完成，官方 SHA256 两次校验通过；338 个 BF16 张量
的形状与数据范围检查通过，实际参数量 1,543,714,304。远端 dev/SFT 仍待你执行。
官方 HF LFS SHA256：

```text
c1b9b30e907950516ba3c646bdf570d8084c25a6410a0cdca80cf04b11bc13a8
```

“基座”在本实验中指原始 Instruct checkpoint，不是此前训练的字符 GPT，
也不是 Qwen 的另一款非 Instruct base checkpoint。

## 2. 下载与传输：在本地 Mac Terminal 执行

远端此前不能直连 HF，采用本地官网下载后传输。以下均在本地执行：

```bash
cd /Users/bytedance/Documents/AI-trainer
python3 scripts/download_repair_base.py --output outputs/qwen2.5-coder-1.5b-base
```

本次完整目录已准备好，可以直接执行下面的 scp；上述命令用于重新核验或未来重新下载。

如果该目录已经有完整模型和 full_model_manifest.json，脚本只校验，
不会重复下载。中断可用同一命令继续；默认 4 路请求（可用 --workers 1 降到单路，
或 --workers 8 使用 8 路）。脚本会阻止多个下载进程同时写入同一目录。
只在看到 `PASS: complete checkpoint and SHA256 verified` 后传输。

```bash
scp -r outputs/qwen2.5-coder-1.5b-base isaac-devbox:/root/AItrainer/AI-trainer/experiments/repair-sft-20261027/models/
scp outputs/repair-baseline-smoke-code.tar.gz isaac-devbox:/root/AItrainer/AI-trainer/experiments/repair-sft-20261027/
```

代码包包含本次新脚本、所需已有脚本、更新的 SPEC 和说明，不包含大权重。
模型目录和输出目录不进入 Git；脚本、配置与规格持久化到 Git。

## 3. 解包与校验：在远端 Terminal 执行

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate agi-post
cd /root/AItrainer/AI-trainer/experiments/repair-sft-20261027
tar -xzf repair-baseline-smoke-code.tar.gz
python scripts/download_repair_base.py --output models/qwen2.5-coder-1.5b-base --verify-only
```

最后一个命令不联网，只检查文件完整性、每个文件的 SHA256 和官方权重哈希。
如果缺文件，重新传输；不要把 .partial 文件当完整模型。

## 4. 测原模型 dev 基线

```bash
python -u scripts/eval_repair_dev.py --model models/qwen2.5-coder-1.5b-base --output reports/dev_base.json
```

脚本只把 dev_prompts.jsonl 的 system/user 消息送入模型，没有标准答案和隐藏测试。
固定 greedy、num_beams=1、repetition_penalty=1.0、max_new_tokens=512；
不会继承官网 generation_config.json 的采样参数。
输入仍使用现有 Qwen chat template。加载 bf16、SDPA，并显式关闭滑动窗口。

每个模型输出经以下步骤处理：

```text
输出 token → 只解码新生成部分 → 去掉首尾空白
→ 检查纯 Python 和要求的函数名 → Docker 执行 → 容器外比较隐藏测试
```

Markdown 围栏不自动剥除：提示要求纯代码，格式错误也属于任务失败。
原模型与 adapter 使用同一规则。raw 输出和处理后的 code 都保存，便于排查。
模型代码不在宿主 Python 执行，Docker 禁网、非 root、只读根目录，限制 CPU、
内存、进程数、输出及超时。每个测试在独立子进程运行，期望值不进入容器。

正常总结示意（数值仅为格式示例，不是已运行结果）：

```json
{
  "total": 5,
  "hidden_passed": 3,
  "public_passed": 4,
  "format_valid": 5,
  "infrastructure_failures": 0,
  "valid_comparison": true
}
```

主指标解释：

```text
dev 通过率 = hidden_passed / total
示例：3 / 5 = 60%
```

必须先确认 complete=true 且 infrastructure_failures=0。
基础设施故障不能解释成模型不会修代码；修复后使用新输出路径重新评测。
脚本拒绝覆盖已有报告，避免意外替换基线。
如果原模型已经 5/5，本数据太容易，不宜据此证明 SFT 有收益；仍可做
短程链路检查，正式效果实验应先增加独立来源题和边界覆盖。

## 5. 原模型基线有效后，进行 10 次更新

```bash
python -u scripts/smoke_repair_sft.py --model models/qwen2.5-coder-1.5b-base --baseline reports/dev_base.json --output outputs/repair-lora-smoke-10 --steps 10
```

脚本会检查基线完整性、模型身份、数据哈希和评测协议。
它从原始完整 checkpoint 初始化新的 LoRA，不从之前随机模型或其他 adapter 接着训练。
训练只加载 train.jsonl；dev.jsonl 只用于训练前后计算验证 loss，test 完全不进入链路。

配置：

```text
LoRA r = 8，alpha = 16，dropout = 0.05
target_modules = q_proj, k_proj, v_proj, o_proj
batch size = 1
gradient_accumulation_steps = 8
learning rate = 1e-4，constant schedule
max_length = 1024，packing = False
max_steps = 10
```

该完整模型有 28 层，以上 4 类投影共 112 个矩阵。r=8 时新建 LoRA
可训练参数为 2,179,072，约 218 万；原模型约 15.44 亿参数保持冻结。

10 次更新不等于只读 10 条样本：

```text
有效 batch = 1 × 8 × 1 张 GPU = 8
10 次更新约处理 80 条样本出现次数，30 条 train 会重复遍历
```

每个 microbatch 产生梯度，积累后统一更新参数。真实 Trainer batch 会再次验收。
labels 的提示位置和 padding 为 -100，回答和模板结束标记有监督；模型内部 shift 一次。

验收内容：

1. 只有 lora_ 参数可训练；冻结基座没有梯度。
2. 每次更新的 LoRA 梯度存在且有限，至少部分非零（初始 A 梯度为零可以正常）。
3. 至少一个 LoRA 参数发生变化。
4. 训练 loss、训练前后 dev loss 都是有限值。
5. 保存 adapter 后卸下内存中的 adapter，再重新加载，eval logits 一致。

产物：

```text
outputs/repair-lora-smoke-10/adapter_model.safetensors
outputs/repair-lora-smoke-10/adapter_config.json
outputs/repair-lora-smoke-10/smoke_manifest.json
reports/repair_sft_smoke.json
```

这一步保存用于推理验证的 adapter，不是完整优化器恢复检查点。
暂不以 dev loss 下降作为链路通过的必要条件：10 步的小数据训练不能保证泛化提升。

## 6. 用同一 dev 协议检查短程 adapter

```bash
python -u scripts/eval_repair_dev.py --model models/qwen2.5-coder-1.5b-base --adapter outputs/repair-lora-smoke-10 --output reports/dev_smoke10.json
```

这是一个新进程：重新加载原模型和磁盘 adapter，可进一步确认可独立恢复使用。
比较 dev_base.json 和 dev_smoke10.json 的 hidden_passed/total、格式错误、
截断、执行失败。此时只比较 dev，不提前查看 test 成绩。

## 7. 如何反馈结果

先贴 dev 基线的 5 条逐题输出和 summary；再贴 smoke 的：

```text
trainable params
loss_before / loss_after
changed_lora_tensors
gradient_checks 的数量
reload_max_logit_difference
```

最后贴 adapter dev summary。根据结果决定增加数据难度、修复链路或开始正式 SFT。
正式训练应从原始 checkpoint 重新启动，不接着这次 smoke adapter 训练。
本地 CPU 小型模型测试只验证实现链路，不代表真实基座的 GPU 结果或性能提升。
