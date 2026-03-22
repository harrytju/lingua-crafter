from transformers import AutoModelForCausalLM, AutoTokenizer

# 指定模型路径 (注意这里改成了你重命名后的名字，如果没有重命名请用 Qwen3___5-9B)
model_path = "/data/wanghongxiang/model/Qwen/Qwen3.5-9B"

print("正在加载模型... (这可能需要一点时间)")
# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 加载模型 (trust_remote_code=True 对于 Qwen 是必须的)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto",           # 自动分配 GPU
    trust_remote_code=True
)

print("模型加载完成！开始对话...")

# 简单的对话测试
prompt = "你好，请介绍一下什么是人工智能。"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 生成回复
outputs = model.generate(**inputs, max_new_tokens=4096)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("-" * 50)
print("模型回复：")
print(response)
