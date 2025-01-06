from transformers import BertModel
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch
from torch import nn
import csv
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from tqdm import tqdm


bert_model_path = '/home/tian/Documents/pythonProject/.vscode/ArboBert_LORA_NSPafMLM_model_1'  # Replace with your local ArboBert model path
model_path = 'ArboBert_LORA_NSPafMLM_base_47_classifier_model.pth'  # Replace with your local model weight file path

class UnlabeledTextDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.texts = list(df['ab'])
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)  # Ensure bert_model_path is correct
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 8)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

def predict_text_batch(model, dataloader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    predictions = []

    for batch in tqdm(dataloader, desc="Predicting"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():  # Disable gradient computation to save memory
            outputs = model(input_ids, attention_mask)
        predicted_classes = outputs.argmax(dim=1)
        predictions.extend(predicted_classes.cpu().numpy())

    return predictions

# Load unlabeled text data
df_unlabeled = pd.read_csv('/home/tian/Documents/pythonProject/.vscode/MLM_pub_data_7w.csv')  # Replace with your local CSV file path
# Remove rows with empty abstracts
df_unlabeled.dropna(subset=['ab'], inplace=True)
# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(bert_model_path)  # Ensure bert_model_path is correct
model = BertClassifier()
model.load_state_dict(torch.load(model_path))  # Ensure model_path is correct

# Create unlabeled text dataset and dataloader
unlabeled_dataset = UnlabeledTextDataset(df_unlabeled, tokenizer)
unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=4, shuffle=False)

# Perform predictions
predicted_categories = predict_text_batch(model, unlabeled_dataloader)
# Check if predicted categories are valid
assert isinstance(predicted_categories, list), "predicted_categories must be a list"

# Print the first few predicted categories for verification
print(predicted_categories[:5])
# Save the prediction results to a CSV file, including PMID
output_file = 'BestArboBert_predicted_results_7w.csv'  # Replace with your desired output file path
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['PMID', 'Category', 'Abstract']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for pmid, category, abstract in zip(df_unlabeled['PMID'], predicted_categories, df_unlabeled['ab']):
        writer.writerow({'PMID': pmid, 'Category': category, 'Abstract': abstract})

print("Predicted categories saved to", output_file)
