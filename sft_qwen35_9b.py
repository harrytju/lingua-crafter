from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import wandb

max_seq_length = 2048
dtype = None
load_in_4bit = True

# 加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    "/data/wanghongxiang/model/Qwen/Qwen3.5-9B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# 添加 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# 加载数据
dataset = load_dataset(
    "json",
    data_files="/data/wanghongxiang/lingua-crafter/data/processed_grammar/train.jsonl",
    split="train"
)

# 格式化函数
def formatting_func(example):
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

dataset = dataset.map(formatting_func, desc="Formatting conversations", num_proc=24)

# 初始化 wandb
wandb.init(
    project="lingua-crafter",
    name="qwen35-9b-grammar-sft-deepspeed",
    config={
        "model": "Qwen3.5-9B",
        "lora_r": 32,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "batch_size": 2,
        "grad_accum": 8,
        "deepspeed": "zero2",
    }
)

# 训练配置
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=24,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        warmup_steps=10,
        max_steps=-1,
        learning_rate=2e-4,
        fp16=False,
        bf16=True, 
        logging_steps=10,
        logging_strategy="steps",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="/data/wanghongxiang/lingua-crafter/checkpoints/qwen35-9b-grammar-sft",
        report_to="wandb",
        run_name="qwen35-9b-grammar-ds",
        save_strategy="epoch",
        deepspeed="ds_config.json",  # 关键：启用 DeepSpeed
    ),
)

trainer.train()
wandb.finish()