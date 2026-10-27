# SFT 远端目录与检查

SSH 别名：`isaac-devbox`。已确认项目 `/root/AItrainer/AI-trainer` 和 RTX 4090 D（24564 MiB）。

本次目录：`/root/AItrainer/AI-trainer/experiments/repair-sft-20261027`。

已复制：

- `scripts/inspect_sft_data.py`
- `examples/repair_sft_demo.jsonl`
- `examples/repair_sft_demo_readable.md`
- `specs/repair-sft/` 的 proposal/design/tasks/checklist
- `reports/sft_learning_guide.md`
- `configs/requirements-agi-post.txt`
- `scripts/check_post_environment.py`
- 本说明文件

创建了空的 `data/repair/` 和 `outputs/`，供后续正式数据与 adapter 使用。

远端后训练环境：`/root/miniconda3/envs/agi-post/bin/python`，克隆自 agi-foundation。Python=3.11.16，torch=2.6.0+cu124，transformers=4.51.3，peft=0.15.2，trl=0.17.0，accelerate=1.6.0，datasets=3.5.0，bitsandbytes=0.45.5，numpy=1.26.4，tokenizers=0.21.1，wandb=0.19.11。完整依赖记录见 reports/requirements-post.txt。

已执行：

```bash
cd /root/AItrainer/AI-trainer/experiments/repair-sft-20261027
/root/miniconda3/envs/agi-foundation/bin/python scripts/inspect_sft_data.py examples/repair_sft_demo.jsonl
```

结果：两条 train 示例结构检查通过。原运行输出保存到远端 `reports/data_structure_check.log`；初始环境检查保存到 `reports/environment_check.json`。

安装后 pip check 无冲突；CUDA/bf16 运算通过；微型随机 Qwen 的 LoRA 权重更新、冻结基座与 adapter 重载通过，重载 logits 最大差异=0。这不是完整基座的正式 SFT，也未验证量化训练或 SFTTrainer 实际 batch。

真实 tokenizer 来自官网，revision=2e1fd397ee46e1388853d2af2c993145b0f1098a。由于远端 HF 直连失败，本地下载 tokenizer/config 后上传到 models/qwen2.5-coder-1.5b-tokenizer 并校验 SHA256。两条样例共 180 token、监督 31 token、最长 95 token；首条参与 loss 文本为正确修复代码及 `<|im_end|>`。

远端使用：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate agi-post
cd /root/AItrainer/AI-trainer/experiments/repair-sft-20261027
python scripts/inspect_sft_data.py examples/repair_sft_demo.jsonl --tokenizer models/qwen2.5-coder-1.5b-tokenizer
```

验收产物：reports/install_agi_post.log、pip_check.log、requirements-post.txt、conda-post.json、gpu-environment.txt、post_environment_check.json、tokenizer_check.log。

已另行准备 authored-v1 的 30/5/5 条 train/dev/test 链路样本，参考答案通过 133 个测试、bug 公开失败见证及全量 tokenizer 检查通过，详见 repair_data_preparation.md。两条示例仍只是教学样例；独立人工审阅、完整 Qwen 权重和正式微调尚未执行。

已有远端仓库包含用户未提交文件，未替用户提交或推送 Git。
