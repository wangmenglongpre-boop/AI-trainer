# 实施任务

- [x] 结合 W3 进度确定 W5–W6 函数修复 SFT 教学方案。
- [x] 提供数据格式示例和结构/来源分组/模板标签检查脚本。
- [x] 本地验证检查脚本、同源/精确 prompt 泄漏拒绝、长度拒绝及 masked loss 的梯度对应关系（模板检查使用 mock，真实 tokenizer 待远端验证）。
- [x] 通过 isaac-devbox 创建 experiments/repair-sft-20261027，复制脚本/样例/方案；使用远端 agi-foundation Python 完成两条样例结构检查。
- [x] 远端克隆 agi-foundation 为独立 agi-post，安装锁定依赖；pip check、CUDA/bf16、微型 Qwen LoRA 更新/重载通过。
- [x] 锁定 Qwen tokenizer revision=2e1fd397ee46e1388853d2af2c993145b0f1098a，本地从官网获取后上传远端并校验哈希。
- [x] 两条示例真实 tokenizer 检查通过：180 总 token、31 监督 token、最长 95 token。
- [x] 构造 AI 辅助自编 30 条 train、5 条 dev、5 条 test；20 个独立来源先分组，Docker 验证 133 个参考测试及 40 个公开 bug 失败，检查重复/近重复候选。
- [x] 正式链路数据的真实离线 Qwen 模板、显式 labels、结束符、长度、产物哈希检查及沙盒安全/超时清理探针通过。
- [ ] 独立人工复核自编需求和期望输出；扩大来源题及隐藏测试，形成更可靠的效果评测集。
- [x] 用户执行实际 SFTTrainer train/dev 全量 batch 检查；提示/padding 屏蔽、回答/结束符监督、pad=EOS 压力检查通过，监督 1152/207 token。
- [x] 提供完整基座官方固定 revision 下载/哈希核验、Docker dev 评测及 10–20 步短程 SFT 脚本和逐步执行指引；实际远端结果待用户执行。
- [x] 本地完整基座准备完成：3,087,467,144 字节权重通过官方 SHA256 两次校验，338 个 BF16 张量结构有效，参数量 1,543,714,304；尚未在远端评测或训练。
- [x] 本地 CPU 小型 Qwen 集成验收：10 次 Trainer 更新、LoRA 梯度与冻结基座、adapter 重载，以及 Docker 32 个 dev 参考测试和异常探针通过。
- [ ] 冻结 prompt/解码/测试，完成原模型及 few-shot dev 基线。
- [ ] 用户在远端完整基座上执行 10–20 步链路验收，记录梯度、参数变化与 adapter 重载结果。
- [ ] 从原基座重新开始正式 LoRA SFT，用 dev 选择 checkpoint。
- [ ] 基于 dev 选 checkpoint，冻结后比较 A/B/C test，记录失败与退化。
