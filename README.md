# AI-Driven-Knowledge-Data-Graphs-for-Tracking-Global-Arbovirus-Research
## ArboBert: A BERT Model Specialized for Arbovirus Research
## Introduction
ArboBert is a specialized BERT model designed for arbovirus disease research. Leveraging the robust capabilities of BioBERT, ArboBert is fine-tuned to excel in processing domain-specific text data, particularly in recognizing and understanding specialized terminology within the arbovirus disease field.

## Dataset
We curated a dataset comprising approximately 70,000 abstracts related to arboviruses. These abstracts are packaged in CSV format and contain a wealth of specialized terminology and concepts pertinent to the field. This dataset serves as the foundation for further pre-training and fine-tuning of the ArboBert model.  
```
ArboBert/MLM_pub_data_7w.csv
```
## Pre-training Methods


### Masked Language Modeling (MLM) & Next Sentence Prediction (NSP)
**MLM**:Customized masking strategy focusing on domain-specific terms.Overall masking probability: 15%. Specialized terms: 100% masking probability to enhance semantic learning.
**NSP**:Enhances understanding of text coherence and structure. Balanced positive and negative sample ratios for effective learning.

## ArboBert Training Requirements
We use linux and miniconda3 environment with NVIDIA GeForce RTX 3080 Ti × 4 as model training platform  
1.Download and install [miniconda3](https://docs.anaconda.com/miniconda/install/#quick-command-line-install)  
2.Navigate to the directory containing environment.yml and execute the following command to create the environment from the environment.yml file:
```
conda env create -f environment.yml
```
3.After the environment is created, activate it using:
```
conda activate your_env_name
```
Replace **your_env_name** with the name specified in the environment.yml file.  
## Pretraining Process
ArboBert's pretraining process is divided into **four main steps**:  

Step | Training Scripts | Output 
---- |----------------| ----
1.MLM Hyperparameter Search | ```MLM_H_S.py```    | Best hyperparameter set identified and logged in ```hyperparameter_search_log.txt```
2.MLM Pretraining |```Train_MLM_DIYMASK.py```|Pretrained MLM model saved to ```ArboBert_LORA_MLM_model``` directory
3.NSP Hyperparameter Search|```NSP_H_S.py```|Best hyperparameter set identified and logged in ```hyperparameter_search_log2.txt```
4.NSP Pretraining|```Train_NSP.py```|Pretrained NSP model saved to ```ArboBert_model``` directory  

By meticulously following the outlined training procedures, you can effectively train the ***ArboBert*** model.  

# Classification Model Training  
We provide a comprehensive comparison of the four classification models used in this project: FastText, BERT, BioBert, and ArboBert.
## Dataset for Classification Model Training  
**Data Source**: All experimental data is sourced from ```all_data_1800.csv```(FastText used ```all_data_1800.txt```), which contains *1,800 records*. Each record represents a text sample along with its corresponding label.  
**Data Preprocessing**: Prior to training, the raw data was cleaned by removing irrelevant characters and normalizing the text to ensure consistency and quality for model training.  
**Data Splitting**: The dataset was randomly divided into training, validation, and test sets following an *8:1:1* ratio. A fixed random seed (42) was used to maintain reproducibility of the experimental results.

Model | Functionality | Training Script | Model Path 
---- | ---- |-----------------|-------------
FastText|Efficient text classification and word vector learning	|```fastext_model.py``` | Imported directly via```import fasttext``` 
[BERT](https://huggingface.co/google-bert)|Bidirectional contextual understanding, suitable for various NLP tasks|```bert_classification.py```| ```classification_model\BERT_classification\BERTmodel``` 
[BioBert](https://huggingface.co/dmis-lab)|Specialized for biomedical text, excellent at handling domain-specific terminology| ```biobert_classification.py``` | ```classification_model\BioBert_classification\biobert```
ArboBert|Specialized for arbovirus domain, enhances classification performance|```arbobert_classification.py```| *Depends on the trained model (see Training Scripts)*  

**Evaluation metrics:** Accuracy, Precision, Recall, F1 Score
## Using ArboBert for Classification
We selected the ArboBert model with the highest test accuracy to classify approximately 70,000 abstracts in the arbovirus domain.  
This is the script used to run ArboBert classification:
```
arbobert_all_predict.py
```
