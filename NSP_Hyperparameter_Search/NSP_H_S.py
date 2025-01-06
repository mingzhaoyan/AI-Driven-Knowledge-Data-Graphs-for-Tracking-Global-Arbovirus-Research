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

model_path = '/home/tian/Documents/pythonProject/.vscode/arbobert_modelMLMonly'  # Replace with your local model path

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
json_path = '/home/tian/Documents/pythonProject/.vscode/nsp_training_data.json'  # Replace with your JSON data file path
with open(json_path, 'r') as f:  # Ensure json_path is correct
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

# Create a log file to record the results of each trial
log_file = 'hyperparameter_search_log2.txt'  # Replace with your desired log file path if needed

def objective(trial):
    # Define the hyperparameter search space
    learning_rate_m = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    rank_m = trial.suggest_int("lora_rank", 16, 16*16, log=True)
    alpha_m = trial.suggest_int("lora_alpha", 64, 64*16, log=True)
    lora_drop_m = trial.suggest_float("lora_drop", 0.01, 0.05, log=True)
    batch_size_m = trial.suggest_int("batch_size", 4, 4, log=True)
    num_train_epochs_m = 1  # Keeping epochs fixed as per original script

    # Reload the model
    model = BertForNextSentencePrediction.from_pretrained(model_path).to(device)  # Change to NSP model

    # Update training parameters
    training_args = TrainingArguments(
        output_dir='./results',  # Replace with your desired output directory if needed
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs_m,
        per_device_train_batch_size=batch_size_m,
        warmup_steps=0,
        save_strategy='epoch',
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=2000,
        logging_dir='./logs',  # Replace with your desired logging directory if needed
        logging_steps=50,
        report_to="none",
        learning_rate=learning_rate_m,
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
        task_type="SEQ_CLS",  # Change to SEQ_CLS (Sequence Classification) if appropriate
        inference_mode=False,
        r=rank_m,
        lora_alpha=alpha_m,
        lora_dropout=lora_drop_m,
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

    # # Inject PEFT (Uncomment if you want to use PEFT)
    # model = inject_adapter_in_model(lora_config, model)

    # Train the model
    metrics = trainer.train()

    # Record hyperparameter combinations and loss values
    with open(log_file, 'a') as f:
        f.write(f"Trial {trial.number}:\n")
        f.write(f"Learning Rate: {learning_rate_m}, lora rank: {rank_m}, lora alpha: {alpha_m}, Batch Size: {batch_size_m}, Num Train Epochs: {num_train_epochs_m}, lora drop: {lora_drop_m}\n")
        f.write(f"Eval Loss: {metrics}\n")
        f.write("\n")
    
    # Return validation loss as the objective function's output
    return metrics[1]
    
#-------------------------------------------Define Training Arguments-------------------------------------------------------------------------------------


# Create a study object to store optimization results
study = optuna.create_study(direction='minimize')

# Start optimization
study.optimize(objective, n_trials=30)  # Adjust n_trials based on your computational resources

# Print the best hyperparameters
print("Best trial:", study.best_trial.params)

# Write results to the log file
with open(log_file, 'a') as f:
    f.write(str(study.best_trial))
    f.write("\n")
    f.write("\n")
