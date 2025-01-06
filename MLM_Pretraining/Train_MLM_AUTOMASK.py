import os
import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch
from peft import inject_adapter_in_model, LoraConfig
from transformers import AdamW, get_constant_schedule  # Import AdamW optimizer



#------------------------------------Model and Tokenizer Loading-------------------------------------------------------------------

# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available.")
    device = torch.device("cuda")
else:
    print("GPU not available, using CPU.")
    device = torch.device("cpu")

model_path = '/home/tian/Documents/pythonProject/.vscode/biobert_model'  # Replace with your local model path

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model path {model_path} does not exist. Please download and cache the model online first.")

csv_path = '/home/tian/Documents/pythonProject/.vscode/MLM_pub_data_7w.csv'  # Replace with your local CSV file path
df = pd.read_csv(csv_path)

try:
    model = AutoModelForMaskedLM.from_pretrained(model_path).to(device)  # Ensure model_path is correct
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # Ensure model_path is correct
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error occurred while loading model and tokenizer: {e}")
    raise e

#------------------------------------Model and Tokenizer Loading-------------------------------------------------------------------


# ------------------------------------------------ Data Processing ---------------------------------------------------------------------------------------

# Find and report rows that are not strings
non_str_rows = df[~df['ab'].apply(lambda x: isinstance(x, str))]
if not non_str_rows.empty:
    print("The following rows contain non-string values and will be deleted:")
    print(non_str_rows[['ab']])
else:
    print("All rows are of string type.")

# Remove rows that are not strings
df = df[df['ab'].apply(lambda x: isinstance(x, str))]

# Define a function to tokenize and process data
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length", return_tensors="pt")

# Convert data to Hugging Face's Dataset format
dataset = Dataset.from_dict({"text": df['ab'].tolist()})

# Create a DatasetDict
dataset_dict = DatasetDict({'train': dataset})

# Use the `train_test_split` method from the `datasets` library to split the dataset
train_dataset, eval_dataset = dataset.train_test_split(test_size=0.1, seed=42).values()  # Keep 10% of data as validation set

# Tokenize the data
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
tokenized_eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Use MLM Data Collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# ------------------------------------------------ Data Processing ---------------------------------------------------------------------------------------


#-------------------------------------------Define Training Arguments-------------------------------------------------------------------------------------

# Set training parameters
training_args = TrainingArguments(
    output_dir='./results',  # Replace with your desired output directory if needed
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    warmup_steps=0,  # If no warmup steps are needed, set to 0
    save_strategy='epoch',           # Save model at the end of each epoch
    save_total_limit=2,
    evaluation_strategy="steps",  # Evaluate every `eval_steps` steps
    eval_steps=2000,  # Evaluate every 2000 steps
    logging_dir='./logs',  # Replace with your desired logging directory if needed
    logging_steps=5,
    report_to="none",  # Disable reporting to TensorBoard or other services
    # Add learning rate scheduler
    learning_rate=7e-3,  # Initial learning rate
    lr_scheduler_type='cosine_with_restarts',  # Learning rate scheduler type

)

# Create Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,  # Use tokenized training dataset
    eval_dataset=tokenized_eval_dataset,    # Use tokenized evaluation dataset
    data_collator=data_collator,
)

# PEFT Configuration
lora_config = LoraConfig(
    task_type="mlm",  # Change to MLM (Masked Language Modeling)
    inference_mode=False,
    r=256*16,
    lora_alpha=2048*16,
    lora_dropout=0.1,
    bias="lora_only",
    target_modules=[
        'cls.predictions.bias',
        'cls.predictions.transform.dense',
        # 'bert_cmp.encoder.layer.11.attention.self.query',
        # 'bert_cmp.encoder.layer.11.attention.self.key.',
        # 'bert_cmp.encoder.layer.11.attention.self.value',
        # 'bert_cmp.encoder.layer.11.attention.output.dense',
        # 'bert_cmp.encoder.layer.11.intermediate.dense',
        # 'bert_cmp.encoder.layer.11.output.dense',
        # 'bert_cmp.encoder.layer.10.attention.self.query',
        # 'bert_cmp.encoder.layer.10.attention.self.key.',
        # 'bert_cmp.encoder.layer.10.attention.self.value',
        # 'bert_cmp.encoder.layer.10.attention.output.dense',
        # 'bert_cmp.encoder.layer.10.intermediate.dense',
        # 'bert_cmp.encoder.layer.10.output.dense',
        # 'bert_cmp.encoder.layer.9.attention.self.query',
        # 'bert_cmp.encoder.layer.9.attention.self.key.',
        # 'bert_cmp.encoder.layer.9.attention.self.value',
        # 'bert_cmp.encoder.layer.9.attention.output.dense',
        # 'bert_cmp.encoder.layer.9.intermediate.dense',
        # 'bert_cmp.encoder.layer.9.output.dense',
    ]  # Select modules to inject PEFT
)

# Inject PEFT
model = inject_adapter_in_model(lora_config, model)

# Start Training
trainer.train()


# Save the fine-tuned model and tokenizer
model.save_pretrained('ArboBert_LORA_MLM_model')  # Replace with your desired model save path
tokenizer.save_pretrained('ArboBert_LORA_MLM_model')  # Replace with your desired tokenizer save path
