import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab


def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []

    # 读取旧的JSONL文件
    with open(origin_path, "r", encoding='utf-8') as file:
        for line in file:
            # 解析每一行的json数据
            data = json.loads(line)
            input_text = data["text"]
            entities = data["entities"]
            match_names = ["地点", "人名", "地理实体", "组织"]

            entity_sentence = ""
            for entity in entities:
                entity_json = dict(entity)
                entity_text = entity_json["entity_text"]
                entity_names = entity_json["entity_names"]
                for name in entity_names:
                    if name in match_names:
                        entity_label = name
                        break

                entity_sentence += f"""{{"entity_text": "{entity_text}", "entity_label": "{entity_label}"}}"""

            if entity_sentence == "":
                entity_sentence = "没有找到任何实体"

            message = {
                "instruction": """你是一个文本实体识别领域的专家，你需要从给定的句子中提取 地点; 人名; 地理实体; 组织 实体. 以 json 格式输出, 如 {"entity_text": "南京", "entity_label": "地理实体"} 注意: 1. 输出的每一行都必须是正确的 json 字符串. 2. 找不到任何实体时, 输出"没有找到任何实体". """,
                "input": f"文本:{input_text}",
                "output": entity_sentence,
            }

            messages.append(message)

    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")


# process_func 函数将输入数据预处理成模型可以接受的格式
# process_func 的函数，用于将输入数据预处理成模型可以接受的格式。预处理的目的是将文本数据转换为模型能够理解和处理的数字序列，同时添加必要的掩码（mask）和标签（labels）以供模型训练和推理使用。
def process_func(example):  # example：ccf_train.jsonl数据
    """
    将数据集进行预处理, 处理成模型可以接受的格式
    """
    # MAX_LENGTH 是一个常量，用于指定输入序列的最大长度。如果输入序列的长度超过这个值，将会进行截断。
    MAX_LENGTH = 384

    # 初始化输入 ID、注意力掩码和标签列表：
    # input_ids 将包含处理后的输入序列的 ID。
    # attention_mask 将包含与 input_ids 对应的注意力掩码。
    # labels 将包含与 input_ids 对应的标签。
    input_ids, attention_mask, labels = [], [], []

    # 定义系统提示模板：
    # system_prompt 是一个提示模板，用于指导模型如何进行实体识别。
    system_prompt = """你是一个文本实体识别领域的专家，你需要从给定的句子中提取 地点; 人名; 地理实体; 组织 实体. 以 json 格式输出, 如 {"entity_text": "南京", "entity_label": "地理实体"} 注意: 1. 输出的每一行都必须是正确的 json 字符串. 2. 找不到任何实体时, 输出"没有找到任何实体"."""

    # 将系统提示和用户输入以及助手响应拼接：
    # tokenizer 是一个文本到 token 的转换器，用于将文本转换为模型可以接受的 token 序列。
    # add_special_tokens=False 参数表示不添加特殊标记（如 <|im_start|> 和 <|im_end|>）。
    instruction = tokenizer(
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)

    # 创建输入 ID、注意力掩码和标签：
    # instruction["input_ids"] 和 response["input_ids"] 分别包含系统提示和用户输入以及助手响应的 token ID。
    # tokenizer.pad_token_id 是用于填充的 token ID。
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
            instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# predict函数首先将输入消息处理成一个适合模型输入的格式，然后使用模型生成回复，最后将回复解码成文本并返回。这个函数可以用于聊天机器人或类似的文本生成任务。
def predict(messages, model, tokenizer):
    # 2、选择设备：
    # device 是一个字符串，用于指定模型将在哪个设备上运行。这里指定为 “cuda”，意味着模型将在 GPU 上运行。如果 GPU 不可用，PyTorch 会自动回退到 CPU。
    device = "cuda"

    # 3、使用模板处理消息：
    # apply_chat_template 是 tokenizer 中的一个方法，用于将消息组合成一个适合模型输入的格式。
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # tokenize=False 参数表示不在此步骤中对消息进行分词。
        add_generation_prompt=True  # add_generation_prompt=True 参数表示在消息前面添加一个生成提示。
    )
    # 4、准备模型输入：
    # tokenizer([text], return_tensors="pt") 调用 tokenizer 方法将文本转换为模型可以接受的格式
    # .to(device) 将模型输入转移到指定的设备上。
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 5、生成回复：
    # model.generate 是模型的一个方法，用于生成新的文本。
    generated_ids = model.generate(
        model_inputs.input_ids,  # model_inputs.input_ids 是模型输入的 ID 序列。
        max_new_tokens=512  # max_new_tokens=512 参数指定生成的文本最大长度。
    )

    # 6、处理生成的 ID 序列：
    # 这行代码将生成的 ID 序列分割成与输入 ID 序列对应的部分。
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 7、解码生成的 ID 序列：
    # tokenizer.batch_decode 是 tokenizer 中的一个方法，用于将 ID 序列转换回文本。
    # skip_special_tokens=True 参数表示在解码时跳过特殊标记。
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(response)

    return response


model_id = "qwen/Qwen2-1.5B-Instruct"
model_dir = "./qwen/Qwen2-1___5B-Instruct"

# 在modelscope上下载Qwen模型到本地目录下
model_dir = snapshot_download(model_id, cache_dir="./", revision="master")

# Transformers加载模型权重

# 初始化 tokenizer：
# AutoTokenizer 是 Hugging Face Transformers 库中的一个类，用于从预训练模型目录中加载 tokenizer。
# from_pretrained(model_dir, use_fast=False, trust_remote_code=True) 参数告诉 Hugging Face 不要使用 fast tokenizer，这可能是因为您的模型需要使用特定的 tokenizer 实现。trust_remote_code=True 参数允许 Hugging Face 信任远程代码，这对于从互联网上下载预训练模型很有用。
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
# 初始化模型：
# AutoModelForCausalLM 是 Hugging Face Transformers 库中的一个类，用于从预训练模型目录中加载一个用于因果语言建模的模型。
# from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16) 参数告诉 Hugging Face 使用 model_dir 指定的预训练模型。device_map="auto" 参数允许 Hugging Face 根据可用设备自动分配模型到不同的设备。torch_dtype=torch.bfloat16 参数指定模型的数据类型为 BFloat16，这是一种低精度浮点数类型，可以提高模型训练的效率。
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)
# 启用模型的输入要求梯度功能：
# enable_input_require_grads() 是模型的一个方法，用于启用模型的输入要求梯度功能。这意味着在计算梯度时，模型将考虑输入张量的梯度。
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 加载、处理数据集和测试集
train_dataset_path = "ccfbdci.jsonl"
train_jsonl_new_path = "ccf_train.jsonl"

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)

# 得到训练集
total_df = pd.read_json(train_jsonl_new_path, lines=True)

# len(total_df) 获取 DataFrame 的总行数。
# int(len(total_df) * 0.1) 计算出总行数的10%，用于确定开始分割的索引。
# train_df 是一个新的 DataFrame，它包含了原始 DataFrame total_df 从10%的位置开始到结束的所有行。这意味着前10%的数据被丢弃，可能用于验证或其他目的。
train_df = total_df[int(len(total_df) * 0.1):]

# Dataset 是 Hugging Face datasets 库中的一个类，用于表示一个数据集。
# from_pandas 是一个类方法，用于从 pandas DataFrame 创建一个 Dataset 对象。
train_ds = Dataset.from_pandas(train_df)

# map 是 Dataset 对象的一个方法，用于对数据集中的每个条目应用一个函数。
# process_func 是一个自定义函数，用于处理数据集中的每个条目。这个函数应该接受一个示例（条目）作为输入，并返回一个处理后的示例。
# remove_columns=train_ds.column_names 参数告诉 map 方法在处理完成后移除所有原始列。train_ds.column_names 获取数据集中的所有列名。
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

model = get_peft_model(model, config)

args = TrainingArguments(
    output_dir="./output/Qwen2-NER",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

swanlab_callback = SwanLabCallback(
    project="Qwen2-NER-fintune",
    experiment_name="Qwen2-1.5B-Instruct",
    description="使用通义千问Qwen2-1.5B-Instruct模型在NER数据集上微调，实现关键实体识别任务。",
    config={
        "model": model_id,
        "model_dir": model_dir,
        "dataset": "qgyd2021/chinese_ner_sft",
    },
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

trainer.train()

# 用测试集的随机20条，测试模型
# 得到测试集
test_df = total_df[:int(len(total_df) * 0.1)].sample(n=20)

test_text_list = []
for index, row in test_df.iterrows():
    instruction = row['instruction']
    input_value = row['input']

    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]

    response = predict(messages, model, tokenizer)
    messages.append({"role": "assistant", "content": f"{response}"})
    result_text = f"{messages[0]}\n\n{messages[1]}\n\n{messages[2]}"
    test_text_list.append(swanlab.Text(result_text, caption=response))

swanlab.log({"Prediction": test_text_list})
swanlab.finish()
