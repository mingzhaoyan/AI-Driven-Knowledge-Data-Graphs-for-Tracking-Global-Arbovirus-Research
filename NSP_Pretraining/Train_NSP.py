import optuna
import os
import pandas as pd
from transformers import BertForNextSentencePrediction, BertTokenizerFast, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch
from peft import inject_adapter_in_model, LoraConfig
from transformers import AdamW, get_constant_schedule  # Import AdamW optimizer
from typing import List, Dict, Tuple
import json

#------------------------------------Model and Tokenizer Loading-------------------------------------------------------------------

# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available.")
    device = torch.device("cuda")
else:
    print("GPU not available, using CPU.")
    device = torch.device("cpu")

model_path = '/home/tian/Documents/modelML'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model path {model_path} does not exist. Please download and cache the model online first.")

try:
    model = BertForNextSentencePrediction.from_pretrained(model_path).to(device)  # Change to NSP model
    tokenizer = BertTokenizerFast.from_pretrained(model_path)  # Or use AutoTokenizer depending on your model
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error occurred while loading model and tokenizer: {e}")
    raise e

#------------------------------------Model and Tokenizer Loading-------------------------------------------------------------------


# ------------------------------------------------ Data Processing ---------------------------------------------------------------------------------------

# Load new JSON formatted dataset
json_path = '/home/tian/Documents/nsp_training_data.json'
with open(json_path, 'r') as f:
    nsp_data = json.load(f)

# Convert JSON data to DataFrame
df_nsp = pd.DataFrame(nsp_data, columns=["sentence1", "sentence2", "is_next"])

# Convert data to Hugging Face's Dataset format
dataset = Dataset.from_dict({
    "sentence1": df_nsp['sentence1'].tolist(),
    "sentence2": df_nsp['sentence2'].tolist(),
    "label": df_nsp['is_next'].astype(int).tolist(),  # Convert boolean to integer
})

# Create a DatasetDict
dataset_dict = DatasetDict({'train': dataset})

# Use the `train_test_split` method from the `datasets` library to split the dataset
train_dataset, eval_dataset = dataset_dict['train'].train_test_split(test_size=0.1, seed=42).values()

# Define a function to tokenize and process data
def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=512, padding="max_length", return_tensors="pt")

# Tokenize the data
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])
tokenized_eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])

# Uncomment the following lines if you need to inspect module names
# for name, module in model.named_modules():
#     print(name)
# ------------------------------------Data Processing ---------------------------------------------------------------------------------------


#-------------------------------------------Define Training Arguments-------------------------------------------------------------------------------------

# Update training parameters
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    warmup_steps=0,
    save_strategy='epoch',
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=2000,
    logging_dir='./logs',
    logging_steps=50,
    report_to="none",
    learning_rate=0.0001135959,
    lr_scheduler_type='cosine_with_restarts',
)

# Create Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
)

# PEFT Configuration
lora_config = LoraConfig(
    task_type="SEQ_CLS",  # Change to SEQ_CLS (Sequence Classification)
    inference_mode=False,
    r=168,
    lora_alpha=129,
    lora_dropout=0.025,
    bias="lora_only", 
    target_modules=[
        'bert.pooler.dense',
        'bert.encoder.layer.11.output.dense',
        'bert.encoder.layer.11.intermediate.dense',
        'bert.encoder.layer.11.attention.output.dense',
        'bert.encoder.layer.11.attention.self.query',
        'bert.encoder.layer.11.attention.self.key.',
        'bert.encoder.layer.11.attention.self.value',
        # 'bert.encoder.layer.10.attention.output.dense',
        # 'bert.encoder.layer.10.intermediate.dense',
        # 'bert.encoder.layer.10.output.dense',
        # 'bert.encoder.layer.9.attention.self.query',
        # 'bert.encoder.layer.9.attention.self.key.',
        # 'bert.encoder.layer.9.attention.self.value',
        # 'bert.encoder.layer.9.attention.output.dense',
        # 'bert.encoder.layer.9.intermediate.dense',
        # 'bert.encoder.layer.9.output.dense',
    ] 
)

# Inject PEFT
model = inject_adapter_in_model(lora_config, model)

# Start Training
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained('ArboBert_LORA_NSPafMLM_model_1')
tokenizer.save_pretrained('ArboBert_LORA_NSPafMLM_model_1')
