"""检查单轮函数修复 JSONL；可选真实 tokenizer 检查模板、长度与 labels。

只做静态检查，不执行代码，不保证参考答案正确，也不证明不存在近重复。
"""
import argparse
import ast
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path


def validate_row(row):
    if not isinstance(row, dict):
        raise ValueError("每行必须是 JSON 对象")
    if set(row) != {"id", "source_task_id", "split", "messages"}:
        raise ValueError("字段必须恰为 id/source_task_id/split/messages")
    for key in ("id", "source_task_id", "split"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"{key} 必须是非空字符串")
    if row["split"] not in ("train", "dev", "test"):
        raise ValueError("split 只接受 train/dev/test")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("本检查器只支持 system/user/assistant 三条消息")
    for msg, role in zip(messages, ("system", "user", "assistant")):
        if not isinstance(msg, dict) or msg.get("role") != role:
            raise ValueError(f"消息顺序应为 system/user/assistant，当前预期 {role}")
        if set(msg) != {"role", "content"}:
            raise ValueError("消息字段必须恰为 role/content")
        if not isinstance(msg.get("content"), str) or not msg["content"].strip():
            raise ValueError(f"{role} content 不能为空")
    answer = messages[-1]["content"]
    if "```" in answer:
        raise ValueError("目标回答应为完整代码，不带 Markdown 围栏")
    # AST 检查不运行代码；完整函数与语义仍需可信测试验证。
    ast.parse(answer)
    return row


def render_labels(row, tokenizer, max_length):
    messages = row["messages"]
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    if not full.startswith(prompt):
        raise ValueError("完整模板不是 prompt 的延续，需重新处理监督边界")
    # 模板已带特殊 token，禁止再自动插入 BOS/EOS。
    pids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if not pids or ids[:len(pids)] != pids:
        raise ValueError("prompt/full 的 token 边界不一致")
    if len(ids) <= len(pids):
        raise ValueError("无有效回答 token")
    if len(ids) > max_length:
        raise ValueError(f"样例 {row['id']} 长度 {len(ids)} > {max_length}；不静默截断")
    labels = [-100] * len(pids) + ids[len(pids):]
    return dict(input_ids=ids, attention_mask=[1] * len(ids), labels=labels,
                prompt_tokens=len(pids), answer_tokens=len(ids) - len(pids))


def inspect_files(paths):
    rows, ids, source_splits, prompt_splits = [], set(), {}, {}
    counts = Counter()
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = validate_row(json.loads(line))
                if row["id"] in ids:
                    raise ValueError(f"重复 id：{row['id']}")
                ids.add(row["id"])
                source, split = row["source_task_id"], row["split"]
                if source in source_splits and source_splits[source] != split:
                    raise ValueError(f"同源题跨集合泄漏：{source}")
                source_splits[source] = split
                prompt = unicodedata.normalize(
                    "NFC", row["messages"][1]["content"].replace("\r\n", "\n").strip())
                digest = hashlib.sha256(prompt.encode()).hexdigest()
                if digest in prompt_splits and prompt_splits[digest] != split:
                    raise ValueError("同一 user prompt 跨集合重复")
                prompt_splits[digest] = split
                counts[split] += 1
                rows.append(row)
            except (ValueError, SyntaxError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not rows:
        raise ValueError("没有有效样例")
    return rows, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--tokenizer", help="HuggingFace 模型 ID 或本地 tokenizer 目录")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()
    if args.max_length < 1:
        parser.error("max-length 必须大于零")
    try:
        rows, counts = inspect_files(args.files)
        print("结构检查通过：", dict(counts))
        print("语义正确性、近重复、官方来源划分及隐藏测试隔离仍需单独验证。")
        if set(counts) != {"train", "dev", "test"}:
            print("当前未同时检查 train/dev/test；不能认定已完成划分验收。")
        if args.tokenizer:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
            features = [render_labels(r, tokenizer, args.max_length) for r in rows]
            print("总 token：", sum(len(f["input_ids"]) for f in features))
            print("有效监督 token：", sum(f["answer_tokens"] for f in features))
            print("最大样例长度：", max(len(f["input_ids"]) for f in features))
            print("首条模板全文：\n", tokenizer.decode(features[0]["input_ids"],
                                                     skip_special_tokens=False))
            supervised = [i for i, label in zip(features[0]["input_ids"],
                                                features[0]["labels"]) if label != -100]
            print("首条参与 loss 的文本：\n",
                  tokenizer.decode(supervised, skip_special_tokens=False))
            print("此处是显式标签检查；Trainer 的实际 collator batch 还须复核。")
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"检查失败：{error}\n")


if __name__ == "__main__":
    main()
