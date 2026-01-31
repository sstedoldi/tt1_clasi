# Utils for DistilBERT iterative training and evaluation
# distilbert_utils.py
import logging
import pandas as pd
import numpy as np
import time
import gc
import os, random, copy
import json
from tqdm.auto import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import DistilBertTokenizerFast, DistilBertModel
from transformers import set_seed as hf_set_seed

from typing import Sequence, Optional, Dict, Tuple

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    

# Accuracy functions
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item()

def topN_accuracy(outputs, labels, n=3):
    topN = torch.topk(outputs, n, dim=1).indices
    return sum([labels[i] in topN[i] for i in range(labels.size(0))])

# def top5_accuracy(outputs, labels):
#     top5 = torch.topk(outputs, 5, dim=1).indices
#     return sum([labels[i] in top5[i] for i in range(labels.size(0))])

def train_epoch(model, data_loader, criterion, optimizer, device, verbose=False):
    if verbose:
        logger.info(f"Model is training on: {device}")
    
    model.train()
    total_loss = 0

    top_n_correct = {n: 0 for n in range(1, 6)}
    n_samples = len(data_loader.dataset)

    loop = tqdm(data_loader, desc="Training", leave=False)
    for batch in loop:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        # Forward
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics update
        total_loss += loss.item()
        for n in range(1, 6):
            top_n_correct[n] += topN_accuracy(outputs, labels, n=n)

        # Update tqdm progress bar
        loop.set_postfix(
            loss=f"{loss.item():.4f}",
            top1=f"{top_n_correct[1]/n_samples:.4f}",
            top5=f"{top_n_correct[5]/n_samples:.4f}"
        )

    avg_loss = total_loss / len(data_loader)
    accuracies = [top_n_correct[n] / n_samples for n in range(1, 6)]

    return (accuracies[0], *accuracies, avg_loss)
    # (Top1, Top1, Top2, Top3, Top4, Top5, AvgLoss) 


def eval_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0

    top_n_correct = {n: 0 for n in range(1, 6)}
    n_samples = len(data_loader.dataset)
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = criterion(outputs, labels)

            total_loss += loss.item()
            for n in range(1, 6):
                top_n_correct[n] += topN_accuracy(outputs, labels, n=n)
    
    avg_loss = total_loss / len(data_loader)
    accuracies = [top_n_correct[n] / n_samples for n in range(1, 6)]

    return (accuracies[0], *accuracies, avg_loss)
    # (Top1, Top1, Top2, Top3, Top4, Top5, AvgLoss) 


class EarlyStopping:
    """
    Stops training when a monitored metric has stopped improving.

    mode:
      - "min": lower is better (e.g., val_loss)
      - "max": higher is better (e.g., val_acc)
    """
    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = "min",
        warmup_epochs: int = 0,
        restore_best: bool = True,
    ):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.warmup_epochs = warmup_epochs
        self.restore_best = restore_best

        self.best_score = None
        self.best_state = None
        self.bad_epochs = 0
        self.epoch = 0

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < (self.best_score - self.min_delta)
        else:
            return score > (self.best_score + self.min_delta)

    def step(self, score: float, model) -> bool:
        """
        Returns True if training should stop.
        """
        self.epoch += 1

        # Warmup: never stop during first N epochs
        if self.epoch <= self.warmup_epochs:
            if self.best_score is None or self._is_improvement(score):
                self.best_score = score
                if self.restore_best:
                    self.best_state = copy.deepcopy(model.state_dict())
            return False

        if self._is_improvement(score):
            self.best_score = score
            self.bad_epochs = 0
            if self.restore_best:
                self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.bad_epochs += 1

        return self.bad_epochs >= self.patience

    def restore(self, model):
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)


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

    logger.info(f"Top-1 Accuracy: {accuracy_top1:.4f} %")
    logger.info(f"Top-2 Accuracy: {accuracy_top2:.4f} %")
    logger.info(f"Top-3 Accuracy: {accuracy_top3:.4f} %")
    logger.info(f"Top-4 Accuracy: {accuracy_top4:.4f} %")
    logger.info(f"Top-5 Accuracy: {accuracy_top5:.4f} %")

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


def bootstrap_sampling(df, test_fraction=0.1, seed=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Determine the number of test samples
    n_test = int(len(df) * test_fraction)
    # Perform bootstrap sampling for the test set
    test_set = df.sample(n=n_test, replace=True, random_state=seed)
    # Remove the test samples from the original dataframe to create the training set
    train_set = df.drop(test_set.index)
    
    return train_set, test_set


def seed_everything(seed: int, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    hf_set_seed(seed)  # cubre random/np/torch también, pero lo dejamos explícito

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        # Para mayor determinismo en CUDA (matmuls). En algunas GPUs/ops es clave:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def plot_training_results(all_histories, out_dir):
    """Genera una figura con Loss y Top-5 Acc para todas las iteraciones."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    for model_name, hist in all_histories.items():
        epochs = range(1, len(hist['train_loss']) + 1)
        # Plot Loss
        ax1.plot(epochs, hist['val_loss'], label=f'{model_name} (val)')
        # Plot Top-5 Accuracy
        ax2.plot(epochs, hist['val_top5_acc'], label=f'{model_name} (val)')

    ax1.set_title('Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.legend(fontsize='small', ncol=2)
    
    ax2.set_title('Validation Top-5 Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.legend(fontsize='small', ncol=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_summary.png'))
    plt.close()


def iterative_training(
    train_type: str,
    text_col: str,
    target_col: str,
    iterations: int,
    max_length: int,
    loader_batch_size: int,
    shuffle: bool,
    lr: float,
    fraction: float,
    out_dir: str,
    *,
    df: pd.DataFrame,
    max_epochs: int = 30,
    early_stopping: bool = True,
    monitor: str = "val_loss",   # "val_loss" or "val_acc" etc.
    patience: int = 3,
    min_delta: float = 0.0,
    warmup_epochs: int = 1,
    seeds: Sequence[int] = None,
    tokenizer,
    label_dir: str,
    fine_tune: bool = False,
    n_finetune_layers: int = 0,
    val_shuffle: Optional[bool] = None,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Entrena iterations modelos (distintos seeds) y devuelve:
      - scored_dfs: dict {model_name: results_df}
      - metrics_df: dataframe con métricas por modelo
    """

    if val_shuffle is None:
        val_shuffle = False  # en general no querés shuffle en validación

    if seeds is None:
        min_val = 0
        max_val = 999999999
        seeds = []
        for i in range(iterations):
            seed = random.randint(min_val, max_val)
            seeds.append(seed)

    if iterations > len(seeds):
        raise ValueError(f"iterations ({iterations}) > len(seeds) ({len(seeds)}).")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    all_metrics = []
    scored_dfs = {}
    master_history = []

    for iter_i in range(iterations):
        seed = seeds[iter_i]
        logger.info(f"Iniciando Iteración {iter_i+1}/{iterations} - Seed: {seed}")
        seed_everything(seed, deterministic=False)

        model_name = f"DBERT_{train_type}_{text_col}_{target_col}_seed{seed}"
        logger.info(f"Model name: {model_name}")

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
        model = HSClassifier(n_classes=len(label2id), 
                             fine_tune=fine_tune,
                             n_finetune_layers=n_finetune_layers)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Training on {device}")
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        history = {
            "train_loss": [],
            "train_acc": [],
            "train_top1_acc": [], # idem train_acc
            "train_top2_acc": [],
            "train_top3_acc": [],
            "train_top4_acc": [],
            "train_top5_acc": [],
            "val_loss": [],
            "val_acc": [], 
            "val_top1_acc": [], # idem val_acc
            "val_top2_acc": [],
            "val_top3_acc": [],
            "val_top4_acc": [],
            "val_top5_acc": [],
        }

        if monitor == "val_loss":
            es_mode = "min"
        elif monitor in ("val_acc", "val_top5_acc"):
            es_mode = "max"
        else:
            raise ValueError(f"Unknown monitor={monitor}")

        es = EarlyStopping(
            patience=patience,
            min_delta=min_delta,
            mode=es_mode,
            warmup_epochs=warmup_epochs,
            restore_best=True,
        ) if early_stopping else None

        for epoch in range(max_epochs):
            if verbose:
                logger.info(f"Epoch {epoch + 1}/{max_epochs}\n" + "-" * 10)
    
            start_time = time.time()

            train_acc, train_top1_acc, train_top2_acc, train_top3_acc, train_top4_acc, train_top5_acc, train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device,
                verbose=verbose
            )
            val_acc, val_top1_acc, val_top2_acc, val_top3_acc, val_top4_acc, val_top5_acc, val_loss = eval_model(
                model, val_loader, criterion, device
            )

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["train_top1_acc"].append(train_top1_acc)
            history["train_top2_acc"].append(train_top2_acc)
            history["train_top3_acc"].append(train_top3_acc)
            history["train_top4_acc"].append(train_top4_acc)
            history["train_top5_acc"].append(train_top5_acc)

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_top1_acc"].append(val_top1_acc)
            history["val_top2_acc"].append(val_top2_acc)
            history["val_top3_acc"].append(val_top3_acc)
            history["val_top4_acc"].append(val_top4_acc)
            history["val_top5_acc"].append(val_top5_acc)

            if verbose:
                logger.info(f"Train loss {train_loss:.4f} acc {train_acc:.4f} top1-5  {train_top1_acc:.4f} {train_top2_acc:.4f} {train_top3_acc:.4f} {train_top4_acc:.4f} {train_top5_acc:.4f}")
                logger.info(f"Val   loss {val_loss:.4f}   acc {val_acc:.4f}   top1-5  {val_top1_acc:.4f}   {val_top2_acc:.4f}   {val_top3_acc:.4f}   {val_top4_acc:.4f}   {val_top5_acc:.4f}")

            epoch_time = time.time() - start_time
            if verbose:
                logger.info(f"Epoch completed in {epoch_time/60:.2f} minutes.\n")

            # ---- EARLY STOPPING DECISION ----
            if es is not None:
                current = {
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_top5_acc": val_top5_acc,
                }[monitor]

                if es.step(current, model):
                    logger.info(
                        f"[EarlyStopping] Stop at epoch {epoch+1}. "
                        f"Best {monitor}={es.best_score:.6f}"
                    )
                    break

        for epoch_idx in range(len(history['train_loss'])):
            master_history.append({
                'iteration': iter_i,
                'seed': seed,
                'epoch': epoch_idx + 1,
                'train_loss': history['train_loss'][epoch_idx],
                'train_acc': history['train_acc'][epoch_idx],
                'train_top5_acc': history['train_top5_acc'][epoch_idx],                
                'val_loss': history['val_loss'][epoch_idx],
                'val_acc': history['val_acc'][epoch_idx],
                'val_top5_acc': history['val_top5_acc'][epoch_idx]
            })
        

        if es is not None:
            es.restore(model)

        # Evaluate (sobre val_df)
        results, metrics = predict_and_evaluate(
            model,
            tokenizer,
            val_df,
            id2label,
            max_length=max_length,
            device=device,
        )

        scored_dfs[model_name] = {"results": results, "history": history}
        all_metrics.append({"model": model_name, **metrics})

        del model
        del optimizer
        del train_encodings
        del val_encodings
        del train_dataset
        del val_dataset
        del train_loader
        del val_loader
        
        torch.cuda.empty_cache()
        gc.collect() 
        
        logger.info(f"Memoria liberada tras iteración {iter_i+1}")

    metrics_df = pd.DataFrame(all_metrics).set_index("model").sort_index()

    # Guardar métricas (nombre correcto)
    metrics_path = os.path.join(out_dir, f"metrics_{train_type}_{text_col}_{target_col}.csv")
    metrics_df.to_csv(metrics_path, index=True)
    logger.info(f"Saved metrics to {metrics_path}")

    # Guardar CSV de historial completo
    history_df = pd.DataFrame(master_history)
    history_path = os.path.join(out_dir, f"history_all_iters_{train_type}.csv")
    history_df.to_csv(history_path, index=False)
    logger.info(f"Historial guardado en {history_path}")

    # Generar Plots
    plot_training_results({k: v['history'] for k, v in scored_dfs.items()}, out_dir)
    logger.info("Gráficos de entrenamiento generados.")

    # Ajustamos el retorno para mantener compatibilidad si es necesario
    final_scored_dfs = {k: v['results'] for k, v in scored_dfs.items()}

    return final_scored_dfs, metrics_df
