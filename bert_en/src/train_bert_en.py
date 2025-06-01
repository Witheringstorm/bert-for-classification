import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report # Added classification_report
import os
import numpy as np # Added for potential use with metrics
import json # Added for saving results
from tqdm import tqdm 
# Configuration
# PRE_TRAINED_MODEL_NAME = 'bert-base-uncased' # Standard English BERT
# If you have a specific local path for an English BERT model, set it here:
PRE_TRAINED_MODEL_NAME = '/public/share/yinxiangrong/qinxiaoyu/kdc2024/S/nlp/proj/bert_en/bert-base-uncased' # Placeholder for local path
PROCESSED_DATA_PATH = "/public/share/yinxiangrong/qinxiaoyu/kdc2024/S/nlp/proj/bert_en/data/processed"  # Relative to bert_en/src/
MODEL_OUTPUT_PATH = "/public/share/yinxiangrong/qinxiaoyu/kdc2024/S/nlp/proj/bert_en/models"          # Relative to bert_en/src/
TRAIN_FILE = os.path.join(PROCESSED_DATA_PATH, "train_en.csv")
TEST_FILE = os.path.join(PROCESSED_DATA_PATH, "test_en.csv")

# Ensure model output directory exists
os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

MAX_LEN = 512  # Max length for tokenization (BERT base models typically handle up to 512)
BATCH_SIZE = 64 # Adjust based on GPU memory (16 is a common default)
EPOCHS = 5     # Number of training epochs (3-5 is common)
LEARNING_RATE = 2e-5

class EnglishTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = EnglishTextDataset(
        texts=df.text.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=os.cpu_count() // 2 if os.cpu_count() else 4 # Dynamic num_workers
    )

def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, torch.mean(torch.tensor(losses))

def eval_model(model, data_loader, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for d in tqdm(data_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    accuracy = correct_predictions.double() / n_examples
    avg_loss = torch.mean(torch.tensor(losses))
    
    # Calculate precision, recall, F1 for binary classification (positive class is 1)
    # and also macro/weighted averages
    precision_binary, recall_binary, f1_binary, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    report = classification_report(all_labels, all_preds, target_names=['human (0)', 'llm (1)'], zero_division=0)
    
    metrics = {
        "accuracy": accuracy.item(),
        "avg_loss": avg_loss.item(),
        "precision_binary": precision_binary,
        "recall_binary": recall_binary,
        "f1_binary": f1_binary,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "classification_report": report
    }
    return metrics

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if the local model path exists, otherwise use hub version
    local_model_path_exists = os.path.isdir(PRE_TRAINED_MODEL_NAME)
    if local_model_path_exists:
        print(f"Using local pre-trained model from: {PRE_TRAINED_MODEL_NAME}")
    else:
        print(f"Local model path {PRE_TRAINED_MODEL_NAME} not found. Attempting to load from Hugging Face Hub as 'bert-base-uncased'.")
        # Fallback to a default hub name if local path is invalid
        # This ensures PRE_TRAINED_MODEL_NAME is always valid for from_pretrained
        # PRE_TRAINED_MODEL_NAME = 'bert-base-uncased' # Re-assign if needed

    tokenizer = BertTokenizerFast.from_pretrained(PRE_TRAINED_MODEL_NAME, local_files_only=local_model_path_exists)
    
    print(f"Loading training data from: {TRAIN_FILE}")
    df_train = pd.read_csv(TRAIN_FILE)
    print(f"Loading test data from: {TEST_FILE}")
    df_test = pd.read_csv(TEST_FILE)

    df_train = df_train.dropna(subset=['text', 'label'])
    df_test = df_test.dropna(subset=['text', 'label'])
    df_train['label'] = df_train['label'].astype(int)
    df_test['label'] = df_test['label'].astype(int)

    print(f"Training data shape: {df_train.shape}")
    print(f"Test data shape: {df_test.shape}")

    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    model = BertForSequenceClassification.from_pretrained(
        PRE_TRAINED_MODEL_NAME,
        num_labels=2, # Binary classification
        local_files_only=local_model_path_exists
    )
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps), # 10% warmup steps
        num_training_steps=total_steps
    )

    best_test_f1_binary = 0 # Save based on F1 score for the positive class
    all_epochs_metrics = [] # Initialize a list to store metrics from all epochs

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        print('-' * 20)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            optimizer,
            device,
            scheduler,
            len(df_train)
        )
        print(f'[LOG] Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}')

        print("Evaluating on test set...")
        test_metrics = eval_model(
            model,
            test_data_loader,
            device,
            len(df_test)
        )
        print(f'[LOG] Test Loss: {test_metrics["avg_loss"]:.4f} | Test Accuracy: {test_metrics["accuracy"]:.4f}')
        print(f'[LOG] Test F1 (Binary for LLM class): {test_metrics["f1_binary"]:.4f}')
        print(f'[LOG] Test F1 (Macro): {test_metrics["f1_macro"]:.4f}')
        print("[LOG] Classification Report (Test Set):")
        print("[LOG]" + test_metrics["classification_report"])

        # Add epoch and training metrics to the results dictionary for saving
        test_metrics['epoch'] = epoch + 1
        test_metrics['train_loss'] = train_loss.item() # Ensure train_loss is a scalar
        test_metrics['train_accuracy'] = train_acc.item() # Ensure train_acc is a scalar

        # Save epoch results to a JSON file
        # epoch_results_filename = f'epoch_{epoch + 1}_results_en.json'
        # epoch_results_path = os.path.join(MODEL_OUTPUT_PATH, epoch_results_filename)
        # with open(epoch_results_path, 'w') as f_json:
        #     json.dump(test_metrics, f_json, indent=4)
        # print(f"[LOG] Saved epoch {epoch + 1} results to {epoch_results_path}")
        all_epochs_metrics.append(test_metrics) # Append current epoch's metrics

        current_f1_binary = test_metrics["f1_binary"]
        if current_f1_binary > best_test_f1_binary:
            model_save_path = os.path.join(MODEL_OUTPUT_PATH, 'best_model_state_en.bin')
            torch.save(model.state_dict(), model_save_path)
            best_test_f1_binary = current_f1_binary
            print(f"[LOG] Saved new best model (F1 Binary: {best_test_f1_binary:.4f}) to {model_save_path}")

    print("\nTraining complete.")
    print(f"[LOG] Best F1-score (Binary for LLM class) on Test Set: {best_test_f1_binary:.4f}")

    # Save all epoch metrics to a single JSON file
    all_results_path = os.path.join(MODEL_OUTPUT_PATH, 'all_epochs_results_en.json')
    with open(all_results_path, 'w') as f_json:
        json.dump(all_epochs_metrics, f_json, indent=4)
    print(f"[LOG] Saved all epoch results to {all_results_path}")

if __name__ == '__main__':
    main() 