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
PROCESSED_DATA_PATH = "/public/share/yinxiangrong/qinxiaoyu/kdc2024/S/nlp/proj/bert_en/data/processed"  # Base path for processed domain subdirectories
MODEL_OUTPUT_PATH = "/public/share/yinxiangrong/qinxiaoyu/kdc2024/S/nlp/proj/bert_en/models"          # Relative to bert_en/src/
# TRAIN_FILE = os.path.join(PROCESSED_DATA_PATH, "train_en.csv") # Will be set dynamically
# TEST_FILE = os.path.join(PROCESSED_DATA_PATH, "test_en.csv") # Will be set dynamically

DOMAINS = ["essay", "reuter", "wp"] # Domains for cross-validation

# Ensure model output directory exists
os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

MAX_LEN = 512  # Max length for tokenization (BERT base models typically handle up to 512)
BATCH_SIZE = 64 # Adjust based on GPU memory (16 is a common default)
EPOCHS = 5     # Number of training epochs
PATIENCE_EPOCHS = 1 # Number of epochs to wait for improvement before early stopping
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
        for d in tqdm(data_loader, desc="Evaluating"): # Added desc to tqdm
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
        # PRE_TRAINED_MODEL_NAME = 'bert-base-uncased' # Re-assign if needed

    tokenizer = BertTokenizerFast.from_pretrained(PRE_TRAINED_MODEL_NAME, local_files_only=local_model_path_exists)
    
    all_cross_domain_results = [] # To store results from all 3x3 evaluations

    for train_domain in DOMAINS:
        print(f"\n{'='*20} TRAINING ON DOMAIN: {train_domain.upper()} {'='*20}")

        train_file_path = os.path.join(PROCESSED_DATA_PATH, train_domain, "train_en.csv")
        val_file_path = os.path.join(PROCESSED_DATA_PATH, train_domain, "val_en.csv")
        
        print(f"Loading training data from: {train_file_path}")
        df_train = pd.read_csv(train_file_path)
        print(f"Loading validation data from: {val_file_path}")
        df_val = pd.read_csv(val_file_path)

        df_train = df_train.dropna(subset=['text', 'label'])
        df_val = df_val.dropna(subset=['text', 'label'])
        df_train['label'] = df_train['label'].astype(int)
        df_val['label'] = df_val['label'].astype(int)

        print(f"Training data shape ({train_domain}): {df_train.shape}")
        print(f"Validation data shape ({train_domain}): {df_val.shape}")

        if df_train.empty or df_val.empty:
            print(f"Skipping training on {train_domain} due to missing train or validation data.")
            continue

        train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
        val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)

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
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        best_val_f1_binary = 0
        epochs_no_improve = 0
        best_model_state = None
        best_epoch = 0
        
        # Path to save the best model for the current training domain
        domain_model_save_path = os.path.join(MODEL_OUTPUT_PATH, f'best_model_state_{train_domain}_en.bin')


        for epoch in range(EPOCHS):
            print(f'\nEpoch {epoch + 1}/{EPOCHS} (Training on {train_domain})')
            print('-' * 20)

            train_acc, train_loss = train_epoch(
                model,
                train_data_loader,
                optimizer,
                device,
                scheduler,
                len(df_train)
            )
            print(f'[LOG] Train Domain: {train_domain} | Epoch: {epoch + 1} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}')

            print(f"Evaluating on validation set for {train_domain}...")
            val_metrics = eval_model(
                model,
                val_data_loader, # Evaluate on the validation set of the current training domain
                device,
                len(df_val)
            )
            print(f'[LOG] Train Domain: {train_domain} | Epoch: {epoch + 1} | Val Loss: {val_metrics["avg_loss"]:.4f} | Val Accuracy: {val_metrics["accuracy"]:.4f}')
            print(f'[LOG] Train Domain: {train_domain} | Epoch: {epoch + 1} | Val F1 (Binary for LLM class): {val_metrics["f1_binary"]:.4f}')
            
            current_val_f1_binary = val_metrics["f1_binary"]
            if current_val_f1_binary > best_val_f1_binary:
                best_val_f1_binary = current_val_f1_binary
                best_model_state = model.state_dict().copy() # Save a copy of the state dict
                best_epoch = epoch + 1
                epochs_no_improve = 0
                torch.save(best_model_state, domain_model_save_path)
                print(f"[LOG] New best validation F1 ({best_val_f1_binary:.4f}) for {train_domain} at epoch {best_epoch}. Model saved to {domain_model_save_path}")
            else:
                epochs_no_improve += 1
                print(f"[LOG] Validation F1 did not improve for {epochs_no_improve} epoch(s) for {train_domain}.")

            if epochs_no_improve >= PATIENCE_EPOCHS:
                print(f"[LOG] Early stopping triggered for {train_domain} at epoch {epoch + 1}. Best F1 was {best_val_f1_binary:.4f} at epoch {best_epoch}.")
                break
        
        if best_model_state is None:
            print(f"[LOG] No model was saved for training domain {train_domain} (e.g., validation F1 never improved or training was skipped). Skipping evaluation.")
            continue

        print(f"\nLoading best model for {train_domain} (from epoch {best_epoch}, Val F1: {best_val_f1_binary:.4f}) for cross-domain testing.")
        # Re-initialize model and load best state to ensure clean state
        model_for_testing = BertForSequenceClassification.from_pretrained(
            PRE_TRAINED_MODEL_NAME,
            num_labels=2,
            local_files_only=local_model_path_exists
        )
        model_for_testing.load_state_dict(torch.load(domain_model_save_path)) # Load the specific domain's best model
        model_for_testing = model_for_testing.to(device)
        model_for_testing.eval() # Ensure model is in eval mode

        for test_domain in DOMAINS:
            print(f"\n--- Testing on DOMAIN: {test_domain.upper()} (Model trained on {train_domain.upper()}) ---")
            test_file_path = os.path.join(PROCESSED_DATA_PATH, test_domain, "test_en.csv")
            
            if not os.path.exists(test_file_path):
                print(f"[LOG] Test file not found: {test_file_path}. Skipping evaluation for this test domain.")
                test_results = {
                    "train_domain": train_domain,
                    "test_domain": test_domain,
                    "best_train_epoch_on_val": best_epoch if best_model_state else "N/A",
                    "status": "skipped_no_test_file",
                    "accuracy": "N/A",
                    "avg_loss": "N/A",
                    "precision_binary": "N/A",
                    "recall_binary": "N/A",
                    "f1_binary": "N/A",
                    "precision_macro": "N/A",
                    "recall_macro": "N/A",
                    "f1_macro": "N/A",
                    "precision_weighted": "N/A",
                    "recall_weighted": "N/A",
                    "f1_weighted": "N/A",
                    "classification_report": "N/A"
                }
                all_cross_domain_results.append(test_results)
                continue

            print(f"Loading test data from: {test_file_path}")
            df_test = pd.read_csv(test_file_path)
            df_test = df_test.dropna(subset=['text', 'label'])
            df_test['label'] = df_test['label'].astype(int)
            print(f"Test data shape ({test_domain}): {df_test.shape}")

            if df_test.empty:
                print(f"[LOG] Test data for {test_domain} is empty. Skipping evaluation.")
                test_results = {
                    "train_domain": train_domain,
                    "test_domain": test_domain,
                    "best_train_epoch_on_val": best_epoch if best_model_state else "N/A",
                    "status": "skipped_empty_test_data",
                    "accuracy": "N/A",
                    "avg_loss": "N/A",
                    # ... (all metric fields as N/A)
                    "classification_report": "N/A"
                }
                all_cross_domain_results.append(test_results)
                continue
                
            test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

            test_metrics = eval_model(
                model_for_testing, # Use the reloaded best model for this training domain
                test_data_loader,
                device,
                len(df_test)
            )
            
            print(f"[LOG] Results for Model (Trained on {train_domain}) on Test Set ({test_domain}):")
            print(f'[LOG] Test Loss: {test_metrics["avg_loss"]:.4f} | Test Accuracy: {test_metrics["accuracy"]:.4f}')
            print(f'[LOG] Test F1 (Binary for LLM class): {test_metrics["f1_binary"]:.4f}')
            print(f'[LOG] Test F1 (Macro): {test_metrics["f1_macro"]:.4f}')
            print("[LOG] Classification Report (Test Set):")
            # Indent classification report for better readability in logs and JSON
            indented_report = "\\n".join(["[LOG] " + line for line in test_metrics["classification_report"].splitlines()])
            print(indented_report)

            result_entry = {
                "train_domain": train_domain,
                "test_domain": test_domain,
                "best_train_epoch_on_val": best_epoch if best_model_state else "N/A", # Epoch when best model was saved for this train_domain
                "status": "evaluated",
                **test_metrics # Unpack all metrics from eval_model
            }
            all_cross_domain_results.append(result_entry)

    print("\n{'='*20} CROSS-DOMAIN EVALUATION COMPLETE {'='*20}")
    
    # Print all results clearly
    print("\nSummary of All Cross-Domain Evaluations:")
    for res in all_cross_domain_results:
        print(f"\nTrained on: {res['train_domain']}, Tested on: {res['test_domain']} (Best model from epoch {res['best_train_epoch_on_val']} of {res['train_domain']} training)")
        if res['status'] == 'evaluated':
            print(f"  Accuracy: {res['accuracy']:.4f}, F1 (Binary): {res['f1_binary']:.4f}, F1 (Macro): {res['f1_macro']:.4f}")
            print(f"  Classification Report:\n{res['classification_report']}") # Display full report
        else:
            print(f"  Status: {res['status']}")


    # Save all epoch metrics to a single JSON file
    all_results_path = os.path.join(MODEL_OUTPUT_PATH, 'cross_domain_results_en.json')
    with open(all_results_path, 'w') as f_json:
        # Convert tensor objects to Python scalars/lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, torch.Tensor):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic): # Handles numpy scalars like np.float32
                return obj.item()
            return obj

        serializable_results = []
        for res_dict in all_cross_domain_results:
            cleaned_dict = {}
            for k, v in res_dict.items():
                if k == 'classification_report': # Keep report as string
                    cleaned_dict[k] = v 
                else:
                    cleaned_dict[k] = convert_for_json(v)
            serializable_results.append(cleaned_dict)

        json.dump(serializable_results, f_json, indent=4)
    print(f"\n[LOG] Saved all cross-domain evaluation results to {all_results_path}")

if __name__ == '__main__':
    main() 