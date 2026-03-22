from datasets import load_dataset
import os

parquet_path = "/data/wanghongxiang/lingua-crafter/data/huggingface_hub_cache/datasets--Teravee--1000_english-grammar-dataset/snapshots/cc49bcadc0f0347d986513dddd1ec1deea47d254/data/train-00000-of-00001.parquet"

# 假设你的 parquet 有 "instruction", "input", "output" 列 或类似
# 如果列名不同，请先 print(ds.column_names) 调整
ds = load_dataset("parquet", data_files={"train": parquet_path})["train"]

def to_chat_format(example):
    # 根据你的实际列名调整
    instruction = example.get("instruction", "")
    input_text  = example.get("input", "")
    output      = example.get("output", example.get("answer", ""))

    messages = [
        {"role": "system", "content": "You are a helpful English grammar assistant."},
        {"role": "user",   "content": f"{instruction}\n{input_text}".strip()},
        {"role": "assistant", "content": output}
    ]
    return {"messages": messages}

ds = ds.map(to_chat_format, remove_columns=ds.column_names)
ds = ds.filter(lambda x: len(x["messages"]) >= 2)  # 至少有 user+assistant

# 保存为 jsonl（Unsloth/TRL 喜欢这个格式）
output_dir = "/data/wanghongxiang/lingua-crafter/data/processed_grammar"
os.makedirs(output_dir, exist_ok=True)
ds.to_json(os.path.join(output_dir, "train.jsonl"), orient="records", lines=True)
print("保存完成:", output_dir)