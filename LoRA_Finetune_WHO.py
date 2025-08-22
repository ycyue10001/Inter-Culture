# ==============================================================================
# 项目名称：提升公共卫生传播领域的机器翻译质量
# 项目核心：使用LoRA微调NLLB-200模型，以减少在翻译世卫组织（WHO）材料时出现的“声誉风险”
# ==============================================================================

# ------------------------------------------------------------------------------
# 步骤 1: 环境配置
# ------------------------------------------------------------------------------
# 首先，我们需要安装所有必需的Python库。
# 如果在本地环境中运行，请在终端执行这些命令。
# 如果在Google Colab等Notebook环境中，可以直接在代码单元格中运行。
# '!' 符号表示在Notebook中执行shell命令。如果在本地环境终端中运行，请去掉'!'。

import os
import subprocess
import sys

# 定义一个函数来安装所需的Python库。
# 当然，也可以直接在终端中运行这些命令。
# 但使用函数可以更好地组织代码，并在需要时重复使用。
def install_packages(packages):
    for package in packages:
        print(f"正在安装 {package}...")
        os.system(f"pip install {package} -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet")

# 自动安装缺失的库
def install_missing_packages(packages):
    for package in packages:
        try:
            __import__(package)  # 尝试导入库
        except ImportError:
            print(f"未找到库 {package}，正在安装...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

print("--- 步骤 1: 正在安装所需库... ---")
install_packages([
    "transformers==4.36.2",
    "datasets==2.16.1",
    "accelerate==0.26.1",
    "evaluate==0.4.1",
    "peft==0.7.1",
    "bitsandbytes==0.42.0",
    "loralib",
    "triton==2.0.0",
    "sacrebleu"
])
print("--- 环境配置完成 ---")


# ------------------------------------------------------------------------------
# 步骤 2: 数据准备
# ------------------------------------------------------------------------------
# 此步骤涉及加载我们的专业语料库，并为模型进行预处理。

import pandas as pd
from datasets import load_dataset

print("\n--- 步骤 2: 正在准备数据集... ---")

# --- 创建虚拟数据文件 ---
# 重要提示：在实际项目中，应该已经有一个名为 'who_corpus.csv' 的文件。
# 这里我们创建一个包含高质量、领域相关句子的示例文件，以保证代码可以运行。
# 这些句子来源于WHO关于疟疾和结核病的官方事实清单。
dummy_data = {
    'en_sentence': [
        "China has maintained zero indigenous malaria cases for 4 consecutive years.",
        "This effort, involving more than 500 scientists from 60 institutions, led to the discovery in the 1970s of artemisinin.",
        "The country also made a major effort to reduce mosquito breeding grounds and stepped up the use of insecticide spraying in homes.",
        "Tuberculosis (TB) is a communicable disease that is a major cause of ill health and one of the leading causes of death worldwide.",
        "Multidrug-resistant TB (MDR-TB) remains a public health crisis and a health security threat.",
        "Globally, an estimated 10.6 million people fell ill with TB in 2021.",
        "The WHO End TB Strategy aims for a 90% reduction in TB deaths.",
        "Palliative care is an approach that improves the quality of life of patients.",
        "The most common species identified was Plasmodium falciparum.",
        "China's “1-3-7” strategy is at the core of its successful malaria elimination effort."
    ],
    'zh_sentence': [
        "中国已连续4年无本地疟疾报告。",
        "这项工作涉及来自60个机构的500多名科学家，最终在1970年代发现了青蒿素。",
        "该国还大力减少蚊子滋生地，并在一些地区的家庭中加强使用杀虫剂喷洒。",
        "结核病是一种传染病，是导致健康不佳的主要原因，也是全世界的主要死因之一。",
        "耐多药结核病（MDR-TB）仍然是公共卫生危机和卫生安全威胁。",
        "2021年，全球估计有1060万人患上结核病。",
        "世卫组织终止结核病战略的目标是使结核病死亡人数减少90%。",
        "姑息治疗是改善患者生活质量的一种方法。",
        "经鉴定，最常见的疟原虫是恶性疟原虫。",
        "中国的“1-3-7”战略是其成功消除疟疾工作的核心。"
    ]
}
# 确保虚拟数据文件保存路径正确
data_file_path = os.path.join(os.getcwd(), "who_corpus.csv")
pd.DataFrame(dummy_data).to_csv(data_file_path, index=False)
# --- 虚拟数据文件创建结束 ---

# 从CSV文件加载数据集时使用绝对路径
raw_datasets = load_dataset("csv", data_files={"train": data_file_path})

print("\n数据集结构:")
print(raw_datasets)

# 接下来，使用NLLB的分词器（Tokenizer）对文本数据做分词。
from transformers import AutoTokenizer

# 定义Hugging Face上的模型检查点
# 选用distilled-600M版本，它在性能和资源消耗之间取得了很好的平衡
model_checkpoint = "facebook/nllb-200-distilled-600M"

# 加载分词器，并指定源语言和目标语言
# 'eng_Latn' 代表拉丁字母书写的英语
# 'zho_Hans' 代表简体中文
tokenizer = AutoTokenizer.from_pretrained(
    model_checkpoint,
    src_lang="eng_Latn",
    tgt_lang="zho_Hans"
)

# 定义预处理函数
def preprocess_function(examples):
    inputs = examples["en_sentence"]
    targets = examples["zh_sentence"]
    # 对输入和目标文本进行分词
    # max_length=128 和 truncation=True 确保所有序列长度一致
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

# 对整个数据集应用预处理函数
# batched=True 加速处理
# remove_columns 删除原始文本列，因为模型不再需要它们
tokenized_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)

print("\n分词后的数据集示例:")
print(tokenized_datasets["train"])
print("--- 数据准备完成 ---")


# ------------------------------------------------------------------------------
# 步骤 3: LoRA微调
# ------------------------------------------------------------------------------
# 这是本项目的技术核心。我们将加载NLLB-200模型，配置LoRA参数，并开始训练。

from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
import torch

print("\n--- 步骤 3: 正在配置LoRA并准备训练... ---")

# 使用8位量化加载模型以节省显存
# device_map="auto" 会自动将模型分配到可用的硬件（GPU/CPU）上
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, device_map="auto")

# 定义LoRA配置
lora_config = LoraConfig(
    r=16,  # 低秩矩阵的秩，是性能和参数量的权衡，通常取8, 16, 32
    lora_alpha=32,  # LoRA的缩放因子，通常设为r的两倍
    target_modules=["q_proj", "v_proj"],  # 选择要应用LoRA的模块，通常是注意力层中的查询和值投影矩阵
    lora_dropout=0.05,  # LoRA层的dropout率，防止过拟合
    bias="none",  # 不训练偏置项
    task_type=TaskType.SEQ_2_SEQ_LM  # 明确任务类型为序列到序列语言模型
)

# 使用LoRA配置封装基础模型，得到一个可训练的PEFT模型
peft_model = get_peft_model(model, lora_config)

# 打印可训练参数的数量和比例，直观感受大模型的效率
peft_model.print_trainable_parameters()

# 定义数据整理器，用于动态填充批次中的数据
data_collator = DataCollatorForSeq2Seq(tokenizer, model=peft_model)

# 定义训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir="nllb-who-lora-finetuned",
    per_device_train_batch_size=2,  # 减小批量大小以节省显存
    learning_rate=2e-4,
    num_train_epochs=10,  # 在小数据集上可以适当增加训练轮数以充分学习
    logging_strategy="epoch",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    predict_with_generate=True, # 必须设置为True才能在评估时生成文本
    report_to="none" # 可设置为 "tensorboard" 以进行可视化监控
)

# 实例化训练器
trainer = Seq2SeqTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["train"], # 在此示例中使用训练集作为评估集，实际应使用独立的验证集
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 开始训练
print("\n--- 开始LoRA微调 ---")
trainer.train()
print("--- 微调完成 ---")

# 保存训练好的LoRA适配器
trainer.save_model("./nllb-lora-who-adapter")
print("\nLoRA适配器已保存至 './nllb-lora-who-adapter'")


# ------------------------------------------------------------------------------
# 步骤 4: 推理与评估
# ------------------------------------------------------------------------------
# 训练完成后，我们来比较微调后的模型与原始基线模型的翻译效果。

from peft import PeftModel

print("\n--- 步骤 4: 正在进行推理与评估... ---")

# 为推理加载基础模型
# 确保我们有一个干净的、未微调的模型用于对比
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, device_map="auto")

# 将训练好的LoRA适配器加载到基础模型上
# 这会创建一个新的、包含了微调知识的模型实例
# 修复推理模型加载问题，确保LoRA适配器加载到正确的设备
inference_model = PeftModel.from_pretrained(base_model, "./nllb-lora-who-adapter").to(base_model.device)

inference_model.eval() # 切换到评估模式，这会关闭dropout等训练特有的层

# 定义包含“声誉风险”术语的测试句子
# 这些句子直接来源于或模仿了WHO报告中提到的错误类型
test_sentences =  [
    "This is a problem for stratified health.",
    "The report mentioned a case of hepatitis being mistranslated.",
    "China's '1-3-7' strategy is at the core of its successful malaria elimination effort.",
    "The discovery of artemisinin was a breakthrough.",
    "Multidrug-resistant TB (MDR-TB) requires a specific treatment regimen.",
    "The Global Fund provides financing to fight these diseases."
]

print("\n--- 翻译质量对比 ---")

for sentence in test_sentences:
    # 准备模型输入
    # 修复推理时的设备问题，确保输入张量和模型在同一设备上
    inputs = tokenizer(sentence, return_tensors="pt").to(inference_model.device)

    # --- 使用LoRA微调模型进行翻译 ---
    with torch.no_grad(): # 在推理时关闭梯度计算，以节省资源
        translated_tokens_peft = inference_model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["zho_Hans"], # 强制解码器以中文开始生成
            max_new_tokens=50 # 限制生成句子的最大长度
        )
    output_finetuned = tokenizer.batch_decode(translated_tokens_peft, skip_special_tokens=True)

    # --- 使用原始基线模型进行翻译 ---
    with torch.no_grad():
        translated_tokens_base = base_model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["zho_Hans"],
            max_new_tokens=50
        )
    output_baseline = tokenizer.batch_decode(translated_tokens_base, skip_special_tokens=True)

    # 打印对比结果
    print(f"\n源句 (EN): {sentence}")
    print(f"基线模型: {output_baseline}")
    print(f"微调模型: {output_finetuned}")

print("\n--- 评估完成 ---")

# ------------------------------------------------------------------------------
# 步骤（可选）: 模型合并与保存
# ------------------------------------------------------------------------------
# 为了在生产环境中获得最佳性能（无额外推理延迟），可以将LoRA适配器与基础模型合并。

print("\n--- (可选)正在合并模型... ---")
merged_model = inference_model.merge_and_unload()
print("模型合并完成。")

# 保存合并后的完整模型，以便将来直接加载使用
merged_model.save_pretrained("./nllb-finetuned-merged")
tokenizer.save_pretrained("./nllb-finetuned-merged")
print("合并后的模型已保存至 './nllb-finetuned-merged'")

# 将来可以直接这样加载合并后的模型：
# from transformers import AutoModelForSeq2SeqLM
# loaded_merged_model = AutoModelForSeq2SeqLM.from_pretrained("./nllb-finetuned-merged")
