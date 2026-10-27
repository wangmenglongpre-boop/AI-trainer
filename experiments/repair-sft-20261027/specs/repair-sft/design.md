# 实验与数据设计

## 模型与训练

第一轮选 bf16 基座 + LoRA，暂不叠加量化因素；保留学习计划中 QLoRA 作为显存受限时的独立路线。记录模型实际 commit revision、tokenizer、模板及依赖，不仅记录 main。

远端已建立 agi-post（克隆 agi-foundation），接口基线见 configs/requirements-agi-post.txt；完整实际依赖见 reports/requirements-post.txt。bitsandbytes 已安装并可导入，但量化训练尚未验证。微型随机 Qwen 的手动 LoRA 更新/重载已通过；用户已执行真实 SFTTrainer batch labels 验收（见 reports/sft_batch_labels_user_confirmation.json）。本地隔离验证环境用于 CPU 小型模型和 Docker 集成验收，不替代远端 PyTorch 2.6/CUDA 完整基座测试。

目标 tokenizer 固定到 revision 2e1fd397ee46e1388853d2af2c993145b0f1098a。远端 HF 直连失败，采用本地官网获取并上传 models/qwen2.5-coder-1.5b-tokenizer，来源及哈希见 source_manifest.json；只含 tokenizer/config，不含完整基座权重。

建议起点：r=8、alpha=16、dropout=0.05，target_modules=q_proj/k_proj/v_proj/o_proj；batch=1、gradient_accumulation=8、lr=1e-4、max_length=1024、packing=False。这些是待实测配置，不是最优值。先在 16–32 条数据上跑 10–20 更新步验收；正式从原基座重启，1 epoch 起步，用 dev 选 checkpoint。短程训练前后检查梯度、LoRA 参数更新、基座冻结及重载 logits。

LoRA：W_new = W_frozen + (alpha/r) * B @ A；A=[r,in]、B=[out,r]。新建 adapter 是参数更新；调 temperature/few-shot 不更新参数。当前无须修改 w3_gpt_block.py。

## 数据链路

使用 MBPP sanitized 官方来源 split 或人工自编题。train 先选约 10 个来源 task，每题至多 3 个可解释 bug，争取约 30 条人工核验样例；实际可用数按验证结果计。dev/test 来源 task_id 各自独立，数量依可用题数确定；正式结论必须报告分母。同题所有变体跟随 source_task_id 分组，先划分来源题再构造变体。自编题还应按任务家族分组，避免同模板换变量名造成泄漏。

本轮采用 AI 辅助自编 authored-v1，不引用 MBPP。来源目录 configs/repair_source_catalog.json，固定 train/dev/test=10/5/5 个独立操作家族，展开为 30/5/5 个 bug。期望输出由独立字面量定义而非运行参考答案生成；20 份参考答案通过全部 133 个测试，40 个 bug 都有公开失败见证。检查来源、精确 prompt、规范化 AST；跨集合 AST 相似度 >=0.75 的候选数为 0，结合需求语义检查，未见换名模板跨集合。该方法不能证明无语义近重复或无预训练污染，独立人工复核仍待做。

只允许 train.jsonl 进入 SFT。dev.jsonl 用于验证 loss；test.jsonl 仅用于质量审计与冻结后的评分，推理使用 dev_prompts.jsonl/test_prompts.jsonl（仅 system/user）。evaluator_cases.json 包含公开/隐藏测试，归评测器所有；不要用 *.jsonl 通配加载训练。manifest.json 固定来源和产物哈希，数据内容 CC0-1.0，不改变仓库其他文件许可证。最长 260 token，train 有效监督 1152 token；全量模板/标签与结束符通过。实际 Trainer batch 已由用户验收：train/dev 监督 1152/207 token，padding 597/41，pad=EOS 压力检查也通过。

本轮 Docker 沙盒固定镜像 ID，禁网、UID=65534、只读根目录、cap-drop ALL、no-new-privileges，CPU=1、内存/交换上限=128MiB、pids=32、tmpfs=8MiB、日志轮转=64KiB、外部执行超时=12s。容器仅接收代码/函数名/输入，期望值与评分在宿主；容器 stdout 连接在该实例无正常输出，改为 create/start/wait/logs，不使用宿主 bind 或候选源码在宿主执行。安全探针验证 UID、只读、能力清零、仅 lo 网络及无限循环超时清理。当前 harness 用于已检查的自编代码；未来任意模型输出的评分需要进一步隔离输出协议与候选进程。

本数据规模用于链路验收，不保证 SFT 有收益；test 每题对应 20 个百分点。基础题可能已在预训练中，正式实验需扩大独立来源与测试覆盖，报告成功数/总数，不只报百分比。

参考代码必须通过可信测试；bug 版本必须失败至少一个公开测试。保留需求、函数签名/imports、错误代码、公开失败摘要；assistant 为完整正确代码。隐藏测试保存在独立评测端，不进入 messages。检查去重、近重复、空答案、语法、长度、未知字段与许可证。代码中的空格和换行有语义，不做任意空白合并。

JSONL 单行字段：id、source_task_id、split、messages。messages 固定 system/user/assistant。examples/repair_sft_demo.jsonl 只是两条教学数据，不能用来评估泛化。静态检查器不运行代码，语法通过不等于答案正确。

## 模板与 loss

使用真实 chat template。完整输入为 prompt+assistant completion。input_ids 仍包含 prompt，attention_mask 非 padding 为 1（模型内部另有 causal mask）；labels 的 prompt 部分为 -100，completion（含模板结束标记）为真实 token ID，padding labels 为 -100。不得因 pad_token_id=eos_token_id 而把真实结束符全部遮蔽；按位置/attention_mask 遮 padding。

训练仍是 teacher forcing：loss = 有效回答目标位置的负 log 概率之和 / 有效目标 token 数。HF CausalLM 内部 shift logits/labels 一次；不要在送入模型前又错位一次。第一个回答 token 的预测来自它前一位置的 logits。prompt 不作为监督目标，但作为上下文影响回答及参数梯度。

inspect_sft_data.py 可选加载 tokenizer，检查完整模板和 token 前缀一致、长度、监督文本。不静默截断正确答案；训练框架实际 batch 的 labels 仍须复核，不能认为检查器自动配置了 Trainer。

生产链路复用预分词 input_ids + completion_mask（提示=0、回答=1）；SFTConfig completion_only_loss=True、skip_prepare_dataset=True、packing=False；使用 TRL 默认 collator，由其实际生成 labels。配置 label_names=["labels"]，避免 PEFT 包装隐藏 forward 签名导致验证不返回 eval_loss。完整基座 Trainer 在每次构建时重新检查 train dataloader 全量 labels。

## 原模型基线与短程验收执行

download_repair_base.py 从官方固定 revision 下载完整 config/tokenizer/safetensors，权重 3087467144 字节，官方 LFS SHA256=c1b9b30e907950516ba3c646bdf570d8084c25a6410a0cdca80cf04b11bc13a8。先在本地下载并校验，用户传输到远端，再 --verify-only 核验全部文件；权重不上 Git。

下载目录使用排他锁，避免两个进程同时拼接权重；分段请求核对 Content-Range，按顺序写入并支持重试续传。只有官方权重 SHA256 与字节数均匹配后才生成完整模型 manifest；校验失败的文件必须重新获取，不能凭文件大小放行。

eval_repair_dev.py 只消费 dev 无答案 prompts，greedy、num_beams=1、repetition_penalty=1.0、max_new_tokens=512，不继承官网采样配置；原模型和 adapter 使用同一解码及纯代码规则。生成代码由 repair_candidate_sandbox.py 在 Docker 子进程执行，每个子进程只得到一份输入；期望值、判分和数据目录不进入容器。固定镜像、禁网/非 root/只读/能力清零，CPU=1、内存/交换=256MiB、pids=32、tmpfs=8MiB；子进程 CPU=2s、墙钟=3s，输出32KiB受限，外部容器超时及清理。基础设施故障单独记录，报告无效，不解释成模型能力失败。

候选函数的普通 print 输出转入受限 stderr，返回值通过独立 stdout JSON 协议传出；调试打印不影响正确返回值的评分。绕过重定向污染 stdout、异常、超时、输出超限和禁止的输入修改均有本地验收。

smoke_repair_sft.py 仅在完整原模型 dev 基线、数据哈希与协议匹配后运行。从原 checkpoint 新建 r=8/alpha=16/dropout=0.05/q,k,v,o LoRA，bf16 SDPA 全注意力，batch=1、累积=8、lr=1e-4 constant、10–20 次更新；train 为训练输入、dev 仅验证 loss、test 不读取。每步检查冻结参数无梯度、LoRA 梯度有限且至少部分非零；检查参数变化、有限 loss 和 adapter 保存重载 logits。保存推理 adapter，不提供完整优化器恢复状态。再次用 eval_repair_dev.py 从新进程评测 adapter dev；正式 SFT 仍需从原 checkpoint 重启。

按用户偏好，本轮代码与模型准备在本地完成，由用户执行远端传输、评测与训练。运行指引和逐步解释见 reports/repair_baseline_smoke_guide.md。本地集成结果见 repair_pipeline_local_check.json，明确记录环境版本及 real_base_gpu_checked=false；不据此勾选真实基座 GPU 验收。

## 对照与评测

A 原 instruct + 固定 prompt；B 同 instruct + 固定 few-shot；C 从 A 相同原 checkpoint 加 LoRA SFT。few-shot 只来自 train，不使用 dev/test 答案。

主指标：单次生成后通过全部隐藏测试的任务数/任务总数。辅助：纯代码格式率、公开通过率、超时/截断、输出长度、延迟、dev assistant loss，以及固定保留任务上的退化。使用同一任务、greedy、max_new_tokens=512 和解析规则；若 few-shot 导致上下文超限须记录并统一预算。只用 dev 选择方案和 checkpoint，冻结后做一次最终 test A/B/C 对比。基础设施故障单独报告。

执行候选及参考代码使用 Docker 沙盒：禁网、非 root、只读根文件系统、CPU/内存/pids/输出限制、超时后按容器 ID 清理。隐藏输入与评分由外部 harness 控制，不交给模型，也不把全部隐藏测试源码与候选代码拼在一起。

## 可选消融

主实验先验证 C 是否有效。之后每次只改一种因素，例如回答正确但格式混乱的数据 vs 统一纯代码格式数据，保持来源题、样例数、预算一致。教学 mask 消融可比较 full-sequence 与 assistant-only loss，但两者 loss 的监督分母不同，不能直接按训练 loss 排名；使用相同外部任务指标。
