# ============ training/train_intent.py ============
"""
Εκπαίδευση του μικρού PyTorch intent classifier (λίγα δευτερόλεπτα σε CPU).

    python -m training.train_intent
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import config  # noqa: E402
from chatbot.intent_classifier import (EMBED_DIM, HIDDEN_DIM, NUM_BUCKETS,  # noqa: E402
                                       IntentClassifier, featurize, make_batch)
from training.dataset_builder import load_jsonl  # noqa: E402

EPOCHS = 12
BATCH = 32
LR = 3e-3


def to_xy(rows, label_to_id):
    feats = [featurize(r["meta"]["message"], r["meta"]["pending"]) for r in rows]
    ys = [label_to_id[r["intent"]] for r in rows]
    return feats, ys


@torch.no_grad()
def accuracy(model, feats, ys) -> float:
    model.eval()
    ids, offsets = make_batch(feats)
    preds = model(ids, offsets).argmax(dim=-1)
    return float((preds == torch.tensor(ys)).float().mean())


def main():
    if not config.TRAIN_PATH.exists():
        print("✖ Δεν βρέθηκε το data/train.jsonl. Τρέξτε πρώτα: python -m training.dataset_builder")
        sys.exit(1)
    torch.manual_seed(config.SEED)
    random.seed(config.SEED)

    labels = list(config.INTENTS)
    label_to_id = {l: i for i, l in enumerate(labels)}
    train_x, train_y = to_xy(load_jsonl(config.TRAIN_PATH), label_to_id)
    val_x, val_y = to_xy(load_jsonl(config.VAL_PATH), label_to_id)

    model = IntentClassifier(len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    print(f"Intent classifier: {sum(p.numel() for p in model.parameters()):,} παράμετροι, "
          f"{len(train_x)} train / {len(val_x)} val")

    best_acc, best_state = -1.0, None
    t0 = time.time()
    order = list(range(len(train_x)))
    for epoch in range(1, EPOCHS + 1):
        model.train()
        random.shuffle(order)
        total = 0.0
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            ids, offsets = make_batch([train_x[j] for j in idx])
            y = torch.tensor([train_y[j] for j in idx])
            optimizer.zero_grad()
            loss = loss_fn(model(ids, offsets), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        val_acc = accuracy(model, val_x, val_y)
        print(f"  epoch {epoch:2d}  loss={total / len(order):.4f}  val_acc={val_acc:.3f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    torch.save({"state_dict": best_state, "labels": labels, "num_buckets": NUM_BUCKETS,
                "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM}, config.INTENT_MODEL_PATH)
    print(f"✔ Αποθηκεύτηκε: {config.INTENT_MODEL_PATH}  (best val_acc={best_acc:.3f}, {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
