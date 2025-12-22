import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel

import os
import json
import pandas as pd
import numpy as np

import time

from typing import Sequence, Optional, Dict, Any, Tuple

class TokenizedDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'label': self.labels[idx]
        }
    

class HSClassifier(nn.Module):
    def __init__(self,
                 n_classes: int,
                 fine_tune: bool = False,
                 n_finetune_layers: int = 0):
        """
        Args:
          n_classes:      number of target classes
          fine_tune:      if True, you’ll unfreeze either all or the last layers
          n_finetune_layers:
                          • =0 (default) → if fine_tune=True, unfreeze *all* DistilBERT layers  
                          • >0             → unfreeze only that many of the *last* transformer blocks  
                          • ignored if fine_tune=False (encoder stays fully frozen)
        """
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # Freeze everything by default
        for param in self.distilbert.parameters():
            param.requires_grad = False

        # If fine_tune, decide what to unfreeze
        if fine_tune:
            if n_finetune_layers > 0:
                # Unfreeze only the last `n_finetune_layers` transformer blocks
                for block in self.distilbert.transformer.layer[-n_finetune_layers:]:
                    for param in block.parameters():
                        param.requires_grad = True
            else:
                # n_finetune_layers == 0 → unfreeze *all* DistilBERT params
                for param in self.distilbert.parameters():
                    param.requires_grad = True

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.distilbert.config.hidden_size, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, n_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0, :]  # Take <CLS> token representation
        logits = self.classifier(hidden_state)
        return logits
    

from tqdm.auto import tqdm


# Accuracy functions
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item()

def top5_accuracy(outputs, labels):
    top5 = torch.topk(outputs, 5, dim=1).indices
    return sum([labels[i] in top5[i] for i in range(labels.size(0))])

def train_epoch(model, data_loader, criterion, optimizer, device):
    print("Model is training on:", next(model.parameters()).device)
    model.train()

    losses = []
    correct = 0
    correct_top5 = 0

    # wrap your DataLoader in a tqdm iterator
    loop = tqdm(data_loader, desc="Training", leave=False)
    for batch in loop:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = criterion(outputs, labels)

        correct      += accuracy(outputs, labels)
        correct_top5 += top5_accuracy(outputs, labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # update the tqdm bar with current metrics
        loop.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct/len(data_loader.dataset):.4f}",
            top5=f"{correct_top5/len(data_loader.dataset):.4f}"
        )

    # make sure you end the line so console prompt isn't on the last bar
    print()

    return (
        correct      / len(data_loader.dataset),
        correct_top5 / len(data_loader.dataset),
        sum(losses)  / len(losses)
    )

def eval_model(model, data_loader, criterion, device):
    # print("Model is eval on:", next(model.parameters()).device)
    model = model.eval()
    losses = []
    correct = 0
    correct_top5 = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

            correct += accuracy(outputs, labels)
            correct_top5 += top5_accuracy(outputs, labels)
            
            losses.append(loss.item())
    
    # print("Input_ids device:", input_ids.device)
    # print("Labels device:", labels.device)
    # print("outputs device:", outputs.device)
    
    return (correct / len(data_loader.dataset), 
            correct_top5 / len(data_loader.dataset),
              np.mean(losses))


class HSDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.descriptions = dataframe['GOODS_DESCRIPTION'].tolist()
        self.labels = dataframe['HS04'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.descriptions[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': self.labels[idx],
            'description': self.descriptions[idx]
        }
        return item


def predict_and_evaluate(
    model, tokenizer, unseen_sample, id2label,
    max_length=128, device='cpu', batch_size=32
):
    """
    Predicts the top 5 classes and their probabilities for an unseen sample using a given model and tokenizer.
    Calculates the accuracy for top 1 to top 5 predictions.
    Uses a DataLoader for GPU memory efficiency.
    """
    # Dataset & DataLoader
    dataset = HSDataset(unseen_sample, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    model.eval()

    all_top5_predicted_labels = []
    all_top5_predicted_probs = []
    all_true_labels = []
    all_descriptions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = F.softmax(outputs, dim=1)
            top5_probs, top5_preds = torch.topk(probabilities, 5, dim=1)

            # Convert predictions and probabilities to lists
            for i in range(top5_preds.size(0)):
                pred_labels = [id2label[idx.item()] for idx in top5_preds[i]]
                pred_probs = [prob.item() for prob in top5_probs[i]]
                all_top5_predicted_labels.append(pred_labels)
                all_top5_predicted_probs.append(pred_probs)
            
            all_true_labels.extend(batch['label'])
            all_descriptions.extend(batch['description'])

    # Accuracy calculations
    accuracy_top1 = sum([
        all_true_labels[i] == all_top5_predicted_labels[i][0]
        for i in range(len(all_true_labels))
    ]) / len(all_true_labels)
    accuracy_top2 = sum([
        all_true_labels[i] in all_top5_predicted_labels[i][:2]
        for i in range(len(all_true_labels))
    ]) / len(all_true_labels)
    accuracy_top3 = sum([
        all_true_labels[i] in all_top5_predicted_labels[i][:3]
        for i in range(len(all_true_labels))
    ]) / len(all_true_labels)
    accuracy_top4 = sum([
        all_true_labels[i] in all_top5_predicted_labels[i][:4]
        for i in range(len(all_true_labels))
    ]) / len(all_true_labels)
    accuracy_top5 = sum([
        all_true_labels[i] in all_top5_predicted_labels[i][:5]
        for i in range(len(all_true_labels))
    ]) / len(all_true_labels)

    print(f"Top-1 Accuracy: {accuracy_top1:.4f} %")
    print(f"Top-2 Accuracy: {accuracy_top2:.4f} %")
    print(f"Top-3 Accuracy: {accuracy_top3:.4f} %")
    print(f"Top-4 Accuracy: {accuracy_top4:.4f} %")
    print(f"Top-5 Accuracy: {accuracy_top5:.4f} %")

    metrics = {
    "top_1_acc": float(accuracy_top1),
    "top_2_acc": float(accuracy_top2),
    "top_3_acc": float(accuracy_top3),
    "top_4_acc": float(accuracy_top4),
    "top_5_acc": float(accuracy_top5),
    }

    # Results DataFrame
    results = pd.DataFrame({
        'Description': all_descriptions,
        'True Label': all_true_labels,
        'Top1': [labels[0] for labels in all_top5_predicted_labels],
        'Proba Top1': [probs[0] for probs in all_top5_predicted_probs],
        'Top2': [labels[1] for labels in all_top5_predicted_labels],
        'Proba Top2': [probs[1] for probs in all_top5_predicted_probs],
        'Top3': [labels[2] for labels in all_top5_predicted_labels],
        'Proba Top3': [probs[2] for probs in all_top5_predicted_probs],
        'Top4': [labels[3] for labels in all_top5_predicted_labels],
        'Proba Top4': [probs[3] for probs in all_top5_predicted_probs],
        'Top5': [labels[4] for labels in all_top5_predicted_labels],
        'Proba Top5': [probs[4] for probs in all_top5_predicted_probs],
    })
    results.set_index(unseen_sample.index, inplace=True)

    return results, metrics


def bootstrap_sampling(df, test_fraction=0.1):
    # Determine the number of test samples
    n_test = int(len(df) * test_fraction)
    # Perform bootstrap sampling for the test set
    test_set = df.sample(n=n_test, replace=True)
    # Remove the test samples from the original dataframe to create the training set
    train_set = df.drop(test_set.index)
    
    return train_set, test_set


def iterative_training(
    train_type: str,
    text_col: str,
    target_col: str,
    iterations: int,
    num_epochs: int,
    max_length: int,
    loader_batch_size: int,
    shuffle: bool,
    lr: float,
    fraction: float,
    out_dir: str,
    *,
    df: pd.DataFrame,
    seeds: Sequence[int],
    tokenizer,
    label_dir: str,
    fine_tune: bool = False,
    val_shuffle: Optional[bool] = None,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Entrena iterations modelos (distintos seeds) y devuelve:
      - scored_dfs: dict {model_name: results_df}
      - metrics_df: dataframe con métricas por modelo
    """

    if val_shuffle is None:
        val_shuffle = False  # en general no querés shuffle en validación

    if iterations > len(seeds):
        raise ValueError(f"iterations ({iterations}) > len(seeds) ({len(seeds)}).")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    all_metrics = []
    scored_dfs = {}

    for iter_i in range(iterations):
        seed = seeds[iter_i]
        print(f"\n=== Iteration {iter_i+1}/{iterations} seed {seed} ===")

        model_name = f"DBERT_{train_type}_{text_col}_{target_col}_seed{seed}"
        print(f"Model name: {model_name}")

        train_df, val_df = bootstrap_sampling(df, test_fraction=fraction, seed=seed)

        # Label mappings (determinístico)
        unique_labels = sorted(df[target_col].astype(str).unique().tolist())
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        id2label = {idx: label for label, idx in label2id.items()}

        # Tokenizing training data
        train_encodings = tokenizer(
            list(train_df[text_col]),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        train_labels = torch.tensor([label2id[str(lbl)] for lbl in train_df[target_col].astype(str)])

        train_dataset = TokenizedDataset(train_encodings, train_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=loader_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        # Tokenizing val data
        val_encodings = tokenizer(
            list(val_df[text_col]),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        val_labels = torch.tensor([label2id[str(lbl)] for lbl in val_df[target_col].astype(str)])

        val_dataset = TokenizedDataset(val_encodings, val_labels)
        val_loader = DataLoader(
            val_dataset,
            batch_size=loader_batch_size,
            shuffle=val_shuffle,
            num_workers=num_workers,
        )

        # Save labels dictionary
        labels_dict = {"label2id": label2id, "id2label": id2label}
        labels_path = os.path.join(label_dir, f"labels_dict_{model_name}.json")
        with open(labels_path, "w") as f:
            json.dump(labels_dict, f, indent=4, ensure_ascii=False)

        # Model
        model = HSClassifier(n_classes=len(label2id), fine_tune=fine_tune)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Training on {device}")
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        history = {
            "train_loss": [],
            "train_acc": [],
            "train_top5_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_top5_acc": [],
        }

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}\n" + "-" * 10)
            start_time = time.time()

            train_acc, train_top5_acc, train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            print(f"Train loss {train_loss} accuracy {train_acc} top5_accuracy {train_top5_acc}")

            val_acc, val_top5_acc, val_loss = eval_model(
                model, val_loader, criterion, device
            )
            print(f"Validation loss {val_loss} accuracy {val_acc} top5_accuracy {val_top5_acc}")

            epoch_time = time.time() - start_time
            print(f"Epoch {epoch + 1} completed in {epoch_time/60:.2f} minutes.\n")

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["train_top5_acc"].append(train_top5_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_top5_acc"].append(val_top5_acc)

        # Evaluate (sobre val_df)
        results, metrics = predict_and_evaluate(
            model,
            tokenizer,
            val_df,
            id2label,
            max_length=max_length,
            device=device,
        )

        scored_dfs[model_name] = results
        all_metrics.append({"model": model_name, **metrics})

        del model
        torch.cuda.empty_cache()

    metrics_df = pd.DataFrame(all_metrics).set_index("model").sort_index()

    # Guardar métricas (nombre correcto)
    metrics_path = os.path.join(out_dir, f"metrics_{train_type}_{text_col}_{target_col}.csv")
    metrics_df.to_csv(metrics_path, index=True)
    print(f"Saved metrics to {metrics_path}")

    return scored_dfs, metrics_df
