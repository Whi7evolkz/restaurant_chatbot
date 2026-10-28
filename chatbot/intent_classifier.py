# ============ chatbot/intent_classifier.py ============
"""
Μικρός ταξινομητής προθέσεων (intent classifier) σε PyTorch.

Αρχιτεκτονική (fastText-style):
    κείμενο → hashed n-grams χαρακτήρων + λέξεων → nn.EmbeddingBag (mean)
            → Linear → ReLU → Dropout → Linear → logits (12 intents)

Ρόλος στο σύστημα:
  1. Γεμίζει τη στήλη «Detected Intent» στο Excel.
  2. Fallback: αν το fine-tuned LLM δεν δώσει έγκυρο JSON, η απόφαση
     βασίζεται στον classifier + κανόνες (rule_parser.py).
  3. Μέτρο σύγκρισης στην αξιολόγηση (agreement με το LLM).

Χρησιμοποιούμε zlib.crc32 για hashing, ώστε τα ίδια n-grams να πηγαίνουν
πάντα στο ίδιο bucket (το hash() της Python αλλάζει σε κάθε εκτέλεση).
"""
from __future__ import annotations

import zlib
from pathlib import Path

import torch
import torch.nn as nn

import config
from chatbot.menu_data import normalize

NUM_BUCKETS = 2 ** 15
EMBED_DIM = 64
HIDDEN_DIM = 64


def featurize(message: str, pending: str | None = None) -> list[int]:
    """Μετατρέπει ένα μήνυμα σε λίστα από ids (hashed features)."""
    text = normalize(message)
    feats: list[str] = []
    words = text.split()
    feats += [f"w:{w}" for w in words]
    feats += [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
    padded = f" {text} "
    feats += [f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2)]
    feats += [f"c4:{padded[i:i + 4]}" for i in range(len(padded) - 3)]
    feats.append(f"p:{pending or 'none'}")          # η κατάσταση βοηθά π.χ. στο «ναι»
    if not words:
        feats.append("empty")
    return [zlib.crc32(f.encode("utf-8")) % NUM_BUCKETS for f in feats]


class IntentClassifier(nn.Module):
    def __init__(self, num_classes: int, num_buckets: int = NUM_BUCKETS,
                 embed_dim: int = EMBED_DIM, hidden_dim: int = HIDDEN_DIM, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.EmbeddingBag(num_buckets, embed_dim, mode="mean")
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embedding(ids, offsets))


def make_batch(feature_lists: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Λίστα από λίστες ids → (ενωμένα ids, offsets) για το EmbeddingBag."""
    offsets, flat, pos = [], [], 0
    for f in feature_lists:
        offsets.append(pos)
        flat.extend(f)
        pos += len(f)
    return torch.tensor(flat, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)


class IntentPredictor:
    """Φόρτωση εκπαιδευμένου μοντέλου και πρόβλεψη."""

    def __init__(self, path: Path = config.INTENT_MODEL_PATH):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.labels: list[str] = ckpt["labels"]
        self.model = IntentClassifier(len(self.labels), ckpt["num_buckets"], ckpt["embed_dim"], ckpt["hidden_dim"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, message: str, pending: str | None = None) -> tuple[str, float]:
        ids, offsets = make_batch([featurize(message, pending)])
        probs = torch.softmax(self.model(ids, offsets), dim=-1)[0]
        conf, idx = probs.max(dim=0)
        return self.labels[int(idx)], float(conf)

    @staticmethod
    def available(path: Path = config.INTENT_MODEL_PATH) -> bool:
        return Path(path).exists()
