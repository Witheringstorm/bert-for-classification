import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
from tqdm import tqdm

# Configuration
PRE_TRAINED_MODEL_NAME = '/public/share/yinxiangrong/qinxiaoyu/kdc2024/S/nlp/proj/bert_cn/bert-base-chinese'
PROCESSED_DATA_PATH = "/public/share/yinxiangrong/qinxiaoyu/kdc2024/S/nlp/proj/bert_cn/data/processed" # Relative to bert_cn/src/
MODEL_OUTPUT_PATH = "/public/share/yinxiangrong/qinxiaoyu/kdc2024/S/nlp/proj/bert_cn/models" # Relative to bert_cn/src/
TRAIN_FILE = os.path.join(PROCESSED_DATA_PATH, "train_cn.csv")
TEST_FILE = os.path.join(PROCESSED_DATA_PATH, "test_cn.csv")

# Ensure model output directory exists
os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

MAX_LEN = 512 # Max length for tokenization
BATCH_SIZE = 128 # Adjust based on GPU memory
EPOCHS = 10 # Number of training epochs
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
        num_workers=4 # Adjust based on your system
    )

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    print("Training model...")
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
            labels=labels # Pass labels to get loss directly from the model
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, torch.mean(torch.tensor(losses))

def eval_model(model, data_loader, loss_fn, device, n_examples):
    print("Evaluating model...")
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
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    
    return accuracy, avg_loss, precision, recall, f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = BertTokenizerFast.from_pretrained(PRE_TRAINED_MODEL_NAME,local_files_only=True)
    
    print(f"Loading training data from: {TRAIN_FILE}")
    df_train = pd.read_csv(TRAIN_FILE)
    print(f"Loading test data from: {TEST_FILE}")
    df_test = pd.read_csv(TEST_FILE)

    # Handle potential NaN values in text or label columns if any
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
        num_labels=2  # Binary classification: human (0) vs generated (1)
    )
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0, # Default, can be adjusted
        num_training_steps=total_steps
    )

    loss_fn = torch.nn.CrossEntropyLoss().to(device) # BertForSequenceClassification includes loss calculation if labels are provided

    best_test_accuracy = 0

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn, # Loss is calculated by model, but can be passed for consistency if needed elsewhere
            optimizer,
            device,
            scheduler,
            len(df_train)
        )
        print(f'Train loss {train_loss} accuracy {train_acc}')

        test_acc, test_loss, test_precision, test_recall, test_f1 = eval_model(
            model,
            test_data_loader,
            loss_fn, # Loss is calculated by model
            device,
            len(df_test)
        )
        print(f'Test loss {test_loss} accuracy {test_acc}')
        print(f'Test Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1-score: {test_f1:.4f}')

        if test_acc > best_test_accuracy:
            torch.save(model.state_dict(), os.path.join(MODEL_OUTPUT_PATH, 'best_model_state_cn.bin'))
            best_test_accuracy = test_acc
            print(f"Saved new best model with accuracy: {best_test_accuracy:.4f} to {os.path.join(MODEL_OUTPUT_PATH, 'best_model_state_cn.bin')}")

    print("Training complete.")
    # To load the model later:
    # model = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels=2)
    # model.load_state_dict(torch.load(os.path.join(MODEL_OUTPUT_PATH, 'best_model_state_cn.bin')))
    # model = model.to(device)

if __name__ == '__main__':
    main() 