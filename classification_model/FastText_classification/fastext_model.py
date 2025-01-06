import fasttext
from sklearn.metrics import f1_score, roc_auc_score, classification_report, roc_curve, auc
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import os

import pickle

# Load data
parent_dir = os.path.abspath('..')  # Replace with your parent directory if different
sub_dir = 'fasttext_data_process'    # Replace with your sub-directory name if different
file_name = 'all_data_1800.txt'      # Replace with your data file name if different
data_path = os.path.join(parent_dir, sub_dir, file_name)  # Replace with your full data path if needed

# Define model parameters
lr = 0.001
epoch = 5000
wordNgrams = 1
k = 10  # 10-fold cross-validation
a = 'fasttext_all'  # Model version

# Read data
with open(data_path, 'r', encoding='utf-8') as f:  # Ensure data_path is correct
    lines = f.readlines()

# Create label dictionary
diction_la = {i + 1: [] for i in range(8)}
np.random.shuffle(lines)
for line in lines:
    la = line[-2:]  # Extract the last character as the label
    la = int(la)
    diction_la[la].append(line)

# Create ten fold sets
folds = [[], [], [], [], [], [], [], [], [], []]
fold_size = len(lines) // k
for la in diction_la:
    list_la = diction_la[la]
    fold_size = len(list_la) // k
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size
        folds[i].extend(list_la[start:end])


def normalize_probabilities(x):
    return x / np.sum(x)


def evaluate_fasttext(model, val_data, version):
    # Initialize counters and lists
    total_acc_test = 0
    all_outputs = []
    all_labels = []
    all_probs = []  # Store predicted probabilities for all samples

    # Predict each sample in the validation set
    for data in val_data:
        text, label = data.rsplit(' ', 1)
        prediction = model.predict(text)[0][0]

        la, probabilities = model.predict(text, 8)
        ordered_probabilities = [0, 0, 0, 0, 0, 0, 0, 0]

        for i, b in enumerate(la):
            label_num1 = int(b.split('__label__')[1]) - 1  # Extract number and subtract 1 for 0-based index
            ordered_probabilities[label_num1] = probabilities[i]

        probs = normalize_probabilities(ordered_probabilities)

        label_num = int(label.split('__label__')[1])  # Assuming label format is '__label__X'
        # Convert prediction to number
        pred_num = int(prediction.split('__label__')[1])
        # Check if prediction is correct
        if pred_num == label_num:
            total_acc_test += 1

        # Append model outputs and labels to lists
        all_outputs.append(pred_num - 1)  # Adjust label to 0-based
        all_labels.append(label_num - 1)  # Adjust label to 0-based
        all_probs.append(probs)

    # Calculate accuracy
    test_accuracy = total_acc_test / len(val_data)

    # Calculate F1 score
    f1 = f1_score(all_labels, all_outputs, average='weighted')

    # Calculate ROC AUC score
    n_classes = 8
    all_scores = np.array(all_probs)

    roc_auc = roc_auc_score(all_labels, all_scores, multi_class='ovo')
    # Plot ROC curve
    fpr = dict()
    tpr = dict()
    roc_auc_per_class = dict()
    mean_fpr = np.linspace(0, 1, 10000)  # For example, using 10,000 points

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((np.array(all_labels) == i), all_scores[:, i])
        roc_auc_per_class[i] = auc(fpr[i], tpr[i])

    # Calculate mean ROC curve
    all_tpr = np.zeros_like(mean_fpr)
    for i in range(n_classes):
        tpr_interp = interp1d(fpr[i], tpr[i])
        all_tpr += tpr_interp(mean_fpr)

    mean_tpr = all_tpr / n_classes  # Mean TPR
    mean_roc_auc = auc(mean_fpr, mean_tpr)
    roc_data = {
        'version': version,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc_per_class': roc_auc_per_class,
        'mean_fpr': mean_fpr,
        'mean_tpr': mean_tpr,
        'mean_roc_auc': mean_roc_auc,
        'test_accuracy': test_accuracy,
        'f1': f1,
        'roc_auc': roc_auc
    }

    with open(f'{version}_EVRROC_data', 'wb') as file:  # Replace with your desired pickle file path if needed
        pickle.dump(roc_data, file)

    # Uncomment the following block if you need to plot and save ROC curves
    # plt.figure(figsize=(8, 6))
    # for i in range(n_classes):
    #     plt.plot(fpr[i], tpr[i], label=f'Class {i + 1} (AUC = {roc_auc_per_class[i]:.2f})')

    # plt.plot(mean_fpr, mean_tpr, color='b', linestyle='--', linewidth=2, label=f'Mean ROC (AUC = {mean_roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC)')
    # plt.legend(loc="lower right")

    # # Save the plot
    # if not os.path.exists('image'):
    #     os.makedirs('image')  # Ensure the directory exists
    # plt.savefig(
    #     f'image/{version}_roc_curve.png',  # Replace 'image/' and 'version' as needed
    #     dpi=300,  # Increase resolution
    #     format='png',  # Can change to 'pdf' or 'svg', etc.
    #     bbox_inches='tight',  # Automatically trim excess whitespace
    #     pad_inches=0.1  # Adjust the size of the trimmed whitespace
    # )
    # plt.close()  # Close the plot window

    with open('fasttextoutput.log', 'a') as log_file:  # Replace with your desired log file path if needed
        print(f'{version} Test Accuracy: {test_accuracy: .3f}', file=log_file)
        print(f'{version} Test F1 Score: {f1: .3f}', file=log_file)
        print(f'{version} Test ROC AUC: {roc_auc: .3f}', file=log_file)
        print(classification_report(all_labels, all_outputs), file=log_file)

    # Output results to console
    print(f'{version} Test Accuracy: {test_accuracy: .3f}')
    print(f'{version} Test F1 Score: {f1: .3f}')
    print(f'{version} Test ROC AUC: {roc_auc: .3f}')
    print(classification_report(all_labels, all_outputs))


# Cross-validation
for i in range(5):
    # Create training and validation sets
    version = f"{a}_{i}"
    train_data = []
    for j in range(k):
        if j != i:
            train_data.extend(folds[j])
    val_data = folds[i]

    # Save training and validation data
    train_file = f"fasttrain{a}_{i}.txt"  # Replace with your desired training file path if needed
    val_file = f"fasttest{a}_{i}.txt"    # Replace with your desired validation file path if needed
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_data)
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(val_data)

    # Train model
    model = fasttext.train_supervised(input=train_file, lr=lr, epoch=epoch, wordNgrams=wordNgrams)
    evaluate_fasttext(model, val_data, version)

    # Remove temporary files
    os.remove(train_file)
    os.remove(val_file)
