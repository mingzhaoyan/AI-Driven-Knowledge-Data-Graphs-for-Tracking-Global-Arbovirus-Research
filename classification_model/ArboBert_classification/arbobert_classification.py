from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from datasets import Dataset

# After training, directly generate reports and plots
model_path = '/home/tian/Documents/pythonProject/.vscode/ArboBert_LORA_NSPafMLM_model_1'  # Replace with your local model path
if torch.cuda.is_available():
    print("GPU is available.")
    device = torch.device("cuda")
else:
    print("GPU not available, using CPU.")
    device = torch.device("cpu")

model = AutoModelForMaskedLM.from_pretrained(model_path)  # Ensure model_path is correct
tokenizer = AutoTokenizer.from_pretrained(model_path)    # Ensure model_path is correct


labels = {
    '__label__1': 0,
    '__label__2': 1,
    '__label__3': 2,
    '__label__4': 3,
    '__label__5': 4,
    '__label__6': 5,
    '__label__7': 6,
    '__label__8': 7
}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [labels[label] for label in df['label']]
        self.texts = [
            tokenizer(
                text,
                padding='max_length',
                max_length=512,
                truncation=True,
                return_tensors="pt"
            )
            for text in df['ab']
        ]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)  # Ensure model_path is correct

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 8)

        # Softmax function for the output layer
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)

        # Apply Softmax activation
        final_layer = self.softmax(linear_output)
        return final_layer

def train(model, train_data, val_data, learning_rate, epochs):
    # Obtain training and validation sets through the Dataset class
    train, val = Dataset(train_data), Dataset(val_data)
    # DataLoader retrieves data based on batch_size; shuffle samples during training
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=6, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=6)
    # Determine whether to use GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    stop_training = False

    # Begin training loop
    for epoch_num in range(epochs):
        if stop_training:
            break
        # Define variables to store training accuracy and loss
        total_acc_train = 0
        total_loss_train = 0
        # Progress bar using tqdm
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            # Get model output
            output = model(input_id, mask)
            # Calculate loss
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()
            # Calculate accuracy
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            # Update model
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        # ------ Validate the model -----------
        # Define variables to store validation accuracy and loss
        total_acc_val = 0
        total_loss_val = 0
        # No need to compute gradients
        with torch.no_grad():
            # Iterate through the validation dataset and validate using the trained model
            for val_input, val_label in val_dataloader:
                # If GPU is available, use GPU; subsequent operations are the same as training
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_data): .3f} 
              | Train Accuracy: {total_acc_train / len(train_data): .3f} 
              | Val Loss: {total_loss_val / len(val_data): .3f} 
              | Val Accuracy: {total_acc_val / len(val_data): .3f}'''
        )

        # Check validation accuracy
        if epoch_num >= 3:
            if (total_acc_val / len(val_data)) > 0.87:
                print("Validation accuracy > 0.87. Stopping training.")
                stop_training = True
        if epoch_num >= 5:
            if (total_acc_val / len(val_data)) > 0.86:
                print("Validation accuracy > 0.86. Stopping training.")
                stop_training = True

from sklearn.metrics import roc_auc_score, roc_curve, classification_report, f1_score, auc
import matplotlib.pyplot as plt

def evaluate(model, test_data, threshold):
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

            # Append model outputs and labels to lists
            all_outputs.append(output.cpu().numpy())  # Using softmax to convert outputs
            all_labels.append(test_label.cpu().numpy())

    test_accuracy = total_acc_test / len(test_data)

    # Calculate F1 score
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_outputs).argmax(axis=1)
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Uncomment the following block if you need to calculate and plot ROC AUC
    # # Calculate ROC AUC score
    # y_scores = np.concatenate(all_outputs)
    # # Ensure probabilities sum to 1
    # assert np.allclose(y_scores.sum(axis=1), 1), "Probabilities do not sum up to 1"
    # roc_auc = roc_auc_score(y_true, y_scores, multi_class='ovo')
    #
    # # Plot ROC curve
    # n_classes = 8
    # fpr = dict()
    # tpr = dict()
    # roc_auc_per_class = dict()
    # for i in range(n_classes):
    #     fpr[i], tpr[i], _ = roc_curve(y_true == i, y_scores[:, i])
    #     roc_auc_per_class[i] = auc(fpr[i], tpr[i])
    #
    # plt.figure()
    # for i in range(n_classes):
    #     plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc_per_class[i]:.2f})')
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC)')
    # plt.legend(loc="lower right")
    # # Save the plot
    # plt.savefig(
    #     f'image/{version}roc_curve.png',  # Replace 'image/' and 'version' as needed
    #     dpi=300,  # Increase resolution
    #     format='png',  # Can change to 'pdf' or 'svg', etc.
    #     bbox_inches='tight',  # Automatically trim excess whitespace
    #     pad_inches=0.1  # Adjust the size of the trimmed whitespace
    # )
    # plt.close()  # Close the plot window

    with open('arbobertoutput.log', 'a') as log_file:  # Replace with your desired log file path if needed
        print(f'{version}Test Accuracy: {test_accuracy: .3f}', file=log_file)
        print(f'{version}Test F1 Score: {f1: .3f}', file=log_file)
        # print(f'{version}Test ROC AUC: {roc_auc: .3f}', file=log_file)
        print(classification_report(y_true, y_pred), file=log_file)

    # If you also need to display results on the console, you can use print statements separately
    print(f'{version}Test Accuracy: {test_accuracy: .3f}')
    print(f'{version}Test F1 Score: {f1: .3f}')
    # print(f'{version}Test ROC AUC: {roc_auc: .3f}')

    # Check if test accuracy exceeds the threshold; if so, save the model
    if test_accuracy > threshold:
        print(f'Saving model because the test accuracy ({test_accuracy:.3f}) is above the threshold ({threshold}).')
        torch.save(model.state_dict(), f'{version}classifier_model.pth')  # Replace with your desired model save path
        df_test.to_csv(f'{version}test_data.csv')  # Replace with your desired test data save path

bbc_text_df = pd.read_csv('/home/tian/Documents/pythonProject/.vscode/all_data_1800.csv')  # Replace with your local CSV file path

for i in range(36, 51):
    df = pd.DataFrame(bbc_text_df)
    version = f'ArboBert_LORA_NSPafMLM_base_{i}_'
    np.random.seed(112 + i)

    df_train, df_val, df_test = np.split(
        df.sample(frac=1, random_state=42),
        [int(.8 * len(df)), int(.9 * len(df))]
    )

    print(len(df_train), len(df_val), len(df_test))

    # # Save test data
    # df_test.to_csv(f'{version}test_data.csv')  # Replace with your desired test data save path

    # Hyperparameters
    # 7
    # 0.0000017

    # EPOCHS = 4
    # LR = 0.00001

    # EPOCHS = 11
    # LR = 0.0000125

    EPOCHS = 8
    LR = 0.0000125

    # Create an instance of your BERT classification model
    model = BertClassifier()

    train(model, df_train, df_val, LR, EPOCHS)

    # torch.save(model.state_dict(), f'{version}classifier_model.pth')  # Replace with your desired model save path

    evaluate(model, df_test, 0.87)
