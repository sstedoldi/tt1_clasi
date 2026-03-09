"""Inference utilities for the HS04 DistilBERT model trained in this project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Union

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizerFast


DEFAULT_CONFIG_PATH = Path("results/distilbert/fft_final/final_model/training_config.json")
DEFAULT_LABELS_PATH = Path(
    "results/distilbert/fft_final/labels/labels_dict_FINAL_DBERT_fft_GOODS_DESCRIPTION_HS04_seed32.json"
)
DEFAULT_MODEL_PATH = Path("results/distilbert/fft_final/final_model/pytorch_model.bin")
DEFAULT_TOKENIZER_NAME = "distilbert-base-uncased"


class HSClassifier(nn.Module):
    def __init__(
        self,
        n_classes: int,
        fine_tune: bool = False,
        n_finetune_layers: int = 0,
    ) -> None:
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(DEFAULT_TOKENIZER_NAME)

        for param in self.distilbert.parameters():
            param.requires_grad = False

        if fine_tune:
            if n_finetune_layers > 0:
                for block in self.distilbert.transformer.layer[-n_finetune_layers:]:
                    for param in block.parameters():
                        param.requires_grad = True
            else:
                for param in self.distilbert.parameters():
                    param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(self.distilbert.config.hidden_size, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, n_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0, :]
        return self.classifier(hidden_state)


class HS04DistilBERTPredictor:
    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
        labels_path: Union[str, Path] = DEFAULT_LABELS_PATH,
        tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
        device: Union[str, torch.device, None] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.labels_path = Path(labels_path)
        self.tokenizer_name = tokenizer_name
        self.device = self._resolve_device(device)

        self.config: Dict[str, object] = {}
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        self.max_length = 300
        self.tokenizer: DistilBertTokenizerFast
        self.model: HSClassifier

        self._load_artifacts()

    def _resolve_device(self, device: Union[str, torch.device, None]) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_json(self, path: Path) -> Dict[str, object]:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _load_artifacts(self) -> None:
        self.config = self._load_json(self.config_path)
        labels_dict = self._load_json(self.labels_path)

        self.label2id = {str(k): int(v) for k, v in labels_dict["label2id"].items()}
        self.id2label = {int(k): str(v) for k, v in labels_dict["id2label"].items()}
        self.max_length = int(self.config.get("max_length", 300))

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.tokenizer_name)

        self.model = HSClassifier(
            n_classes=len(self.label2id),
            fine_tune=bool(self.config.get("fine_tune", False)),
            n_finetune_layers=int(self.config.get("n_finetune_layers", 0)),
        )

        state_dict = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _predict_proba(self, texts: Sequence[str], batch_size: int) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        all_probs: List[torch.Tensor] = []
        self.model.eval()

        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                encodings = self.tokenizer(
                    list(batch_texts),
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                input_ids = encodings["input_ids"].to(self.device)
                attention_mask = encodings["attention_mask"].to(self.device)
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = F.softmax(logits, dim=1).cpu()
                all_probs.append(probs)

        return torch.cat(all_probs, dim=0)

    def _top_k_predictions(self, probs: torch.Tensor, top_k: int) -> List[Dict[str, float]]:
        k = min(max(top_k, 1), probs.shape[0])
        top_probs, top_indices = torch.topk(probs, k=k, dim=-1)
        predictions: List[Dict[str, float]] = []

        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            predictions.append(
                {
                    "label": self.id2label[int(idx)],
                    "probability": float(prob),
                }
            )
        return predictions

    def predict(self, text: str, top_k: int = 5) -> Dict[str, object]:
        probs = self._predict_proba([str(text)], batch_size=1)[0]
        predictions = self._top_k_predictions(probs, top_k=top_k)
        top1 = predictions[0]
        return {
            "Description": str(text),
            "Top1": top1["label"],
            "Proba Top1": top1["probability"],
            "predictions": predictions,
        }

    def predict_batch(
        self,
        texts: Sequence[str],
        top_k: int = 5,
        batch_size: int = 32,
    ) -> pd.DataFrame:
        text_list = [str(text) for text in texts]
        if not text_list:
            return pd.DataFrame(columns=["Description"])

        probs = self._predict_proba(text_list, batch_size=batch_size)
        rows: List[Dict[str, object]] = []

        for text, prob_row in zip(text_list, probs):
            predictions = self._top_k_predictions(prob_row, top_k=top_k)
            row: Dict[str, object] = {"Description": text}
            for rank, pred in enumerate(predictions, start=1):
                row[f"Top{rank}"] = pred["label"]
                row[f"Proba Top{rank}"] = pred["probability"]
            rows.append(row)

        return pd.DataFrame(rows)
