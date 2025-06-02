import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import os
from tqdm import tqdm
import json
import numpy as np

# Configuration
PRE_TRAINED_MODEL_NAME = '/public/share/yinxiangrong/qinxiaoyu/kdc2024/S/nlp/proj/bert_cn/bert-base-chinese'
PROCESSED_DATA_PATH = "/public/share/yinxiangrong/qinxiaoyu/kdc2024/S/nlp/proj/bert_cn/data/processed" # Relative to bert_cn/src/
MODEL_OUTPUT_PATH = "/public/share/yinxiangrong/qinxiaoyu/kdc2024/S/nlp/proj/bert_cn/models" # Relative to bert_cn/src/
TRAIN_FILE = os.path.join(PROCESSED_DATA_PATH, "train_cn.csv")
VAL_FILE = os.path.join(PROCESSED_DATA_PATH, "val_cn.csv")
TEST_FILE = os.path.join(PROCESSED_DATA_PATH, "test_cn.csv")

# Ensure model output directory exists
os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

MAX_LEN = 512 # Max length for tokenization
BATCH_SIZE = 128 # Adjust based on GPU memory
EPOCHS = 10 # Number of training epochs
PATIENCE_EPOCHS = 2 # Added for early stopping
LEARNING_RATE = 2e-5

class ChineseTextDataset(Dataset):
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
            return_token_type_ids=False, # Not needed for BERT sequence classification
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
    ds = ChineseTextDataset(
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

    for d in tqdm(data_loader, desc="Training Epoch"):
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
        for d in tqdm(data_loader, desc="Evaluating"):
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

    precision_binary, recall_binary, f1_binary, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    report = classification_report(all_labels, all_preds, target_names=['human (0)', 'llm (1)'], zero_division=0, output_dict=False)
    
    metrics = {
        "accuracy": accuracy.item() if isinstance(accuracy, torch.Tensor) else accuracy,
        "avg_loss": avg_loss.item() if isinstance(avg_loss, torch.Tensor) else avg_loss,
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
    print(f"[LOG] Using device: {device}")

    tokenizer = BertTokenizerFast.from_pretrained(PRE_TRAINED_MODEL_NAME,local_files_only=True)
    
    print(f"[LOG] Loading training data from: {TRAIN_FILE}")
    df_train = pd.read_csv(TRAIN_FILE)
    print(f"[LOG] Loading validation data from: {VAL_FILE}")
    df_val = pd.read_csv(VAL_FILE)
    print(f"[LOG] Loading test data from: {TEST_FILE}")
    df_test = pd.read_csv(TEST_FILE)

    df_train = df_train.dropna(subset=['text', 'label'])
    df_val = df_val.dropna(subset=['text', 'label'])
    df_test = df_test.dropna(subset=['text', 'label'])
    df_train['label'] = df_train['label'].astype(int)
    df_val['label'] = df_val['label'].astype(int)
    df_test['label'] = df_test['label'].astype(int)

    print(f"[LOG] Training data shape: {df_train.shape}")
    print(f"[LOG] Validation data shape: {df_val.shape}")
    print(f"[LOG] Test data shape: {df_test.shape}")

    if df_train.empty or df_val.empty or df_test.empty:
        print("[LOG] One of the data files (train, val, or test) is empty. Exiting.")
        return

    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    model = BertForSequenceClassification.from_pretrained(
        PRE_TRAINED_MODEL_NAME,
        num_labels=2,
        local_files_only=True
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
    best_model_state_path = os.path.join(MODEL_OUTPUT_PATH, 'best_model_state_cn.bin')
    best_epoch = 0

    for epoch in range(EPOCHS):
        print(f'\n[LOG] Epoch {epoch + 1}/{EPOCHS}')
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

        print("[LOG] Evaluating on validation set...")
        val_metrics = eval_model(
            model,
            val_data_loader,
            device,
            len(df_val)
        )
        print(f'[LOG] Val Loss: {val_metrics["avg_loss"]:.4f} | Val Accuracy: {val_metrics["accuracy"]:.4f}')
        print(f'[LOG] Val F1 (Binary for LLM class): {val_metrics["f1_binary"]:.4f}')
        print(f'[LOG] Val F1 (Macro): {val_metrics["f1_macro"]:.4f}')
        print("[LOG] Validation Classification Report:")
        indented_report_val = "\\n".join(["[LOG] " + line for line in val_metrics["classification_report"].splitlines()])
        print(indented_report_val)

        current_val_f1_binary = val_metrics["f1_binary"]
        if current_val_f1_binary > best_val_f1_binary:
            torch.save(model.state_dict(), best_model_state_path)
            best_val_f1_binary = current_val_f1_binary
            best_epoch = epoch + 1
            epochs_no_improve = 0
            print(f"[LOG] New best validation F1 ({best_val_f1_binary:.4f}) at epoch {best_epoch}. Model saved to {best_model_state_path}")
        else:
            epochs_no_improve += 1
            print(f"[LOG] Validation F1 did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= PATIENCE_EPOCHS:
            print(f"[LOG] Early stopping triggered at epoch {epoch + 1}. Best F1 was {best_val_f1_binary:.4f} at epoch {best_epoch}.")
            break
    
    print("\n[LOG] Training complete or early stopping triggered.")
    
    if os.path.exists(best_model_state_path):
        print(f"[LOG] Loading best model from epoch {best_epoch} (Val F1: {best_val_f1_binary:.4f}) for final testing...")
        model_for_testing = BertForSequenceClassification.from_pretrained(
            PRE_TRAINED_MODEL_NAME,
            num_labels=2,
            local_files_only=True
        )
        model_for_testing.load_state_dict(torch.load(best_model_state_path))
        model_for_testing = model_for_testing.to(device)
        model_for_testing.eval()

        print("[LOG] Evaluating on test set with the best model...")
        test_metrics = eval_model(
            model_for_testing,
            test_data_loader,
            device,
            len(df_test)
        )
        print(f'[LOG] Test Loss: {test_metrics["avg_loss"]:.4f} | Test Accuracy: {test_metrics["accuracy"]:.4f}')
        print(f'[LOG] Test F1 (Binary for LLM class): {test_metrics["f1_binary"]:.4f}')
        print(f'[LOG] Test F1 (Macro): {test_metrics["f1_macro"]:.4f}')
        print("[LOG] Test Set Classification Report:")
        indented_report_test = "\\n".join(["[LOG] " + line for line in test_metrics["classification_report"].splitlines()])
        print(indented_report_test)

        test_results_output_path = os.path.join(MODEL_OUTPUT_PATH, 'test_results_cn.json')
        
        def convert_for_json(obj):
            if isinstance(obj, torch.Tensor):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            return obj

        serializable_test_metrics = {}
        for k, v in test_metrics.items():
            if k == 'classification_report':
                serializable_test_metrics[k] = v 
            else:
                serializable_test_metrics[k] = convert_for_json(v)
        
        serializable_test_metrics['best_model_val_f1_binary'] = best_val_f1_binary
        serializable_test_metrics['best_model_epoch'] = best_epoch
        
        with open(test_results_output_path, 'w') as f_json:
            json.dump(serializable_test_metrics, f_json, indent=4)
        print(f"[LOG] Saved test results to {test_results_output_path}")
    else:
        print("[LOG] No best model was saved (e.g., validation F1 never improved or training was skipped). Skipping final test evaluation.")

if __name__ == '__main__':
    main() 