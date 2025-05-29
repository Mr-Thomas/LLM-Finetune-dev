import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
import os
import swanlab


def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    :param origin_path: 原始 JSONL 文件路径
    :param new_path: 转换后的 JSONL 文件路径
    """
    messages = []

    # 读取旧的JSONL文件
    with open(origin_path, "r", encoding="utf-8") as file:
        for line in file:
            # 解析每一行的json数据
            data = json.loads(line.strip())  # 去除首尾空格/换行符
            context = data["text"]
            catagory = data["category"]
            label = data["output"]
            message = {
                "instruction": "你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型",
                "input": f"文本:{context},类型选型:{catagory}",
                "output": label,
            }
            messages.append(message)

    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")


def process_func(example):
    """
    将数据集进行预处理
    :param example: 单条数据
    :param tokenizer: 分词器
    :return: 处理后的输入 ID、注意力掩码和标签
    """
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = (
            instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    )
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = (
            [-100] * len(instruction["input_ids"])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def predict(messages, model, tokenizer):
    """
    使用模型进行预测
    :param messages: 输入消息列表
    :param model: 模型
    :param tokenizer: 分词器
    :return: 预测结果
    """
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(response)

    return response


# 在modelscope上下载Qwen模型到本地目录下
model_dir = snapshot_download(
    "qwen/Qwen2-1.5B-Instruct", cache_dir="./", revision="master"
)

# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(
    model_dir, use_fast=False, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_dir, device_map="auto", torch_dtype=torch.bfloat16
)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 加载、处理数据集和测试集
train_dataset_path = "train.jsonl"
test_dataset_path = "test.jsonl"

train_jsonl_new_path = "new_train.jsonl"
test_jsonl_new_path = "new_test.jsonl"

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
if not os.path.exists(test_jsonl_new_path):
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

# 得到训练集
train_df = pd.read_json(train_jsonl_new_path, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

# 可选：排查异常样本
for i, sample in enumerate(train_dataset):
    if len(sample["input_ids"]) < 10:
        print(f"短样本警告 Index={i}: {tokenizer.decode(sample['input_ids'])}")

# LoRA 配置
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
)

model = get_peft_model(model, config)

# 训练参数
args = TrainingArguments(
    output_dir="./output/Qwen1.5",
    per_device_train_batch_size=8,  # 增大以提升收敛稳定性（配合梯度累积）
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    logging_steps=10,
    save_steps=100,  # 每100步保存一次模型
    learning_rate=5e-5,  # 1e-4 较为常用。若模型过早陷入梯度震荡，可能学习率偏高
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    max_grad_norm=1.0,  # 梯度爆炸，建议显式加上梯度裁剪
    weight_decay=0.01,
    gradient_checkpointing=True,  # （显存更省，但训练速度稍慢），但对 output_dir 文件内容本身影响不大。
    bf16=True,  # bf16 或 fp16 精度训练	若未启用，建议打开	显存更省、训练更快。若显存不足，建议关闭
    report_to="none",
    label_names=["input_ids", "attention_mask", "labels"],  # 指定 Trainer 使用哪些字段作为监督标签字段
    dataloader_num_workers=4,
)

# SwanLab 实验记录
swanlab.login(api_key="zmkhLZcM6E7zZ6MVlt20K", save=True)
swanlab_callback = SwanLabCallback(
    project="Qwen2-finetune",
    experiment_name="Qwen2-1.5B-Instruct-optimized",
    description="LoRA优化后的Qwen2文本分类微调",
    config={
        "model": "Qwen2-1.5B-Instruct",
        "dataset": "自定义新闻分类",
        "lora_rank": 8,
        "lora_alpha": 16,
        "learning_rate": 5e-5,
    },
    log_parameters={
        "grad_clip": args.max_grad_norm,
        "batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
    },
    log_artifacts=["config.json", "special_tokens_map.json"],
)

# 数据整理器
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8,
    return_tensors="pt",
    label_pad_token_id=-100,  # 忽略填充部分的损失
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    callbacks=[swanlab_callback],
)

trainer.train()

# 保存 LoRA adapter（仅保存可训练的 adapter 层）
model.save_pretrained("./output/lora_adapter", save_embedding_layers=False)
# 保存 tokenizer 配置（包含 vocab、special tokens、配置等）
tokenizer.save_pretrained("./output/lora_adapter")

# 推理展示（测试集前100条）
test_df = pd.read_json(test_jsonl_new_path, lines=True)[:100]
results = []
for _, row in test_df.iterrows():
    messages = [
        {"role": "system", "content": row["instruction"]},
        {"role": "user", "content": row["input"]},
    ]
    output = predict(messages, model, tokenizer)
    messages.append({"role": "assistant", "content": output})
    result_text = f"{messages[0]}\n\n{messages[1]}\n\n{messages[2]}"
    results.append(swanlab.Text(result_text, caption=output))

swanlab.log({"Prediction": results})
swanlab.finish()
