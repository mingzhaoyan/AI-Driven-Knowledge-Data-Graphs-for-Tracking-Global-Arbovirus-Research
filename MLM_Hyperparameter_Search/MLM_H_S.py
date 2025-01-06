import optuna
import os
import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch
from peft import inject_adapter_in_model, LoraConfig
from transformers import AdamW, get_constant_schedule  # Import AdamW optimizer
import pickle
from typing import List, Dict, Tuple


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

with open('/home/tian/Documents/pythonProject/.vscode/vdb_set.pickle', 'rb') as f:  # Replace with your specific tokens pickle file path
    specific_tokens = pickle.load(f)

# 3. Custom Data Collator
class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=True, mlm_probability=0.15, specific_tokens=None):
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)
        self.specific_tokens = specific_tokens

    def _mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        if self.tokenizer.mask_token is not None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        else:
            special_tokens_mask = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.all_special_tokens)] * labels.shape[0]
            probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
            special_tokens_mask = []

        if self.specific_tokens is not None:
            specific_tokens_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in self.specific_tokens]
            specific_tokens_mask = torch.zeros_like(labels, dtype=torch.bool)
            for token_id in specific_tokens_ids:
                specific_tokens_mask |= labels.eq(token_id)
            probability_matrix.masked_fill_(specific_tokens_mask, value=1.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        assert isinstance(examples, list) and all(isinstance(ex, dict) for ex in examples), "examples must be a list of dictionaries"

        batch = self.tokenizer.pad(examples, padding=True, return_tensors="pt")
        batch["input_ids"], batch["labels"] = self._mask_tokens(batch["input_ids"])
        return batch

# 4. Use the custom data collator to process data
custom_data_collator = CustomDataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
    specific_tokens=specific_tokens
)

# ------------------------------------------------ Data Processing ---------------------------------------------------------------------------------------


#-------------------------------------------Define Training Format and Hyperparameter Search-------------------------------------------------------------------------------------

# Create a log file to record the results of each trial
log_file = 'hyperparameter_search_log.txt'  # Replace with your desired log file path if needed

def objective(trial):
    # Define the hyperparameter search space
    learning_rate_m = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    rank_m = trial.suggest_int("lora_rank", 16, 16*64, log=True)
    alpha_m = trial.suggest_int("lora_alpha", 64, 64*64, log=True)
    lora_drop_m = trial.suggest_float("lora_drop", 0.01, 0.1, log=True)
    batch_size_m = trial.suggest_int("batch_size", 8, 8, log=True)
    num_train_epochs_m = trial.suggest_int("epochs", 1, 4, log=True)

    # Reload the model
    model = AutoModelForMaskedLM.from_pretrained(model_path).to(device)

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

    # Create a Trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=custom_data_collator,
    )

    # PEFT Configuration
    lora_config = LoraConfig(
        task_type="mlm",  # Change to MLM (Masked Language Modeling)
        inference_mode=False,
        r=rank_m,
        lora_alpha=alpha_m,
        lora_dropout=lora_drop_m,
        bias="lora_only",
        target_modules=[
            'cls.predictions.bias',
            'cls.predictions.transform.dense',
            # 'bert_cmp.encoder.layer.11.attention.self.query',
            # 'bert_cmp.encoder.layer.11.attention.self.key.',
            # 'bert_cmp.encoder.layer.11.attention.self.value',
            'bert_cmp.encoder.layer.11.attention.output.dense',
            'bert_cmp.encoder.layer.11.intermediate.dense',
            'bert_cmp.encoder.layer.11.output.dense',
            # 'bert_cmp.encoder.layer.10.attention.self.query',
            # 'bert_cmp.encoder.layer.10.attention.self.key.',
            # 'bert_cmp.encoder.layer.10.attention.self.value',
            'bert_cmp.encoder.layer.10.attention.output.dense',
            'bert_cmp.encoder.layer.10.intermediate.dense',
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
    
#-------------------------------------------Define Training Format and Hyperparameter Search-------------------------------------------------------------------------------------


# Create a study object to store optimization results
study = optuna.create_study(direction='minimize')

# Start optimization
study.optimize(objective, n_trials=60)  # Adjust n_trials based on your computational resources

# Print the best hyperparameters
print("Best trial:", study.best_trial.params)

# Write results to the log file
with open(log_file, 'a') as f:
    f.write(str(study.best_trial))
    f.write("\n")
    f.write("\n")
