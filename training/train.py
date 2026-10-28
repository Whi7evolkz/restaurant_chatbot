# ============ training/train.py ============
"""
LoRA fine-tuning του Qwen2.5-Instruct με ΚΑΘΑΡΟ PyTorch training loop.

Δεν χρησιμοποιούμε τον έτοιμο HF Trainer, ώστε να φαίνεται κάθε βήμα:
Dataset → DataLoader → forward → loss → backward → clip → optimizer.step → scheduler.

Το loss υπολογίζεται ΜΟΝΟ πάνω στα tokens της απάντησης (assistant JSON).
Τα tokens του prompt έχουν label -100 και αγνοούνται από το CrossEntropy.

Εκτέλεση:
    python -m training.train                       # πλήρης εκπαίδευση (config)
    python -m training.train --epochs 1 --limit 500   # γρήγορη εκδοχή για CPU
    python -m training.train --max-steps 5         # μόνο έλεγχος ότι δουλεύει
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

import config  # noqa: E402
from training.dataset_builder import load_jsonl  # noqa: E402


# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """CUDA (NVIDIA) → MPS (Apple Silicon) → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
class ChatSFTDataset(Dataset):
    """Κάθε δείγμα: input_ids = prompt + απάντηση, labels = -100 για το prompt."""

    def __init__(self, rows: list[dict], tokenizer, max_len: int):
        self.samples = []
        end = tokenizer.eos_token or ""
        truncated = 0
        for r in rows:
            prompt = tokenizer.apply_chat_template(r["messages"][:2], tokenize=False, add_generation_prompt=True)
            answer = r["messages"][2]["content"] + end
            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
            if len(p_ids) + len(a_ids) > max_len:
                truncated += 1
                p_ids = p_ids[-max(1, max_len - len(a_ids)):]
            self.samples.append({
                "input_ids": p_ids + a_ids,
                "labels": [-100] * len(p_ids) + a_ids,
            })
        if truncated:
            print(f"  (σημείωση: {truncated} δείγματα κόπηκαν στα {max_len} tokens)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_collate(pad_id: int):
    def collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            pad = max_len - n
            ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            mask.append([1] * n + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}
    return collate


@torch.no_grad()
def evaluate_loss(model, loader, device, autocast_ctx) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast_ctx():
            out = model(**batch)
        n_tokens = int((batch["labels"] != -100).sum())
        total += float(out.loss) * n_tokens
        count += n_tokens
    model.train()
    return total / max(1, count)


def plot_losses(step_log: list[dict], epoch_log: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    steps = [r["step"] for r in step_log]
    losses = [r["train_loss"] for r in step_log]
    ax.plot(steps, losses, label="train loss", color="#2E86AB", alpha=0.85)
    if epoch_log:
        ax.plot([r["step"] for r in epoch_log], [r["val_loss"] for r in epoch_log], "o-",
                label="validation loss", color="#E4572E")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("LoRA fine-tuning – training vs validation loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.TRAINING_LOSS_PNG, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="LoRA fine-tuning (plain PyTorch loop)")
    ap.add_argument("--base-model", default=config.BASE_MODEL_NAME)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--grad-accum", type=int, default=config.GRAD_ACCUM_STEPS)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--max-len", type=int, default=config.MAX_SEQ_LEN)
    ap.add_argument("--limit", type=int, default=0, help="χρήση μόνο των πρώτων N train δειγμάτων")
    ap.add_argument("--max-steps", type=int, default=0, help="σταμάτα μετά από τόσα optimizer steps")
    ap.add_argument("--output", default=str(config.ADAPTER_DIR))
    args = ap.parse_args()

    if not config.TRAIN_PATH.exists():
        print("✖ Δεν βρέθηκε data/train.jsonl. Τρέξτε πρώτα: python -m training.dataset_builder")
        sys.exit(1)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    set_seed(config.SEED)
    device = get_device()
    print(f"▶ Συσκευή: {device}   Μοντέλο: {args.base_model}")
    if device.type == "cpu":
        print("  ⚠ Εκπαίδευση σε CPU: θα πάρει αρκετό χρόνο. Για γρηγορότερα δείτε το README (Google Colab).")

    # ---------------- tokenizer & δεδομένα ----------------
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows = load_jsonl(config.TRAIN_PATH)
    val_rows = load_jsonl(config.VAL_PATH)
    if args.limit:
        train_rows = train_rows[: args.limit]
        val_rows = val_rows[: max(8, args.limit // 10)]
    train_ds = ChatSFTDataset(train_rows, tokenizer, args.max_len)
    val_ds = ChatSFTDataset(val_rows, tokenizer, args.max_len)
    collate = make_collate(tokenizer.pad_token_id)
    g = torch.Generator().manual_seed(config.SEED)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    avg_len = sum(len(s["input_ids"]) for s in train_ds.samples) / len(train_ds)
    print(f"  train={len(train_ds)}  val={len(val_ds)}  μέσο μήκος={avg_len:.0f} tokens")

    # ---------------- μοντέλο + LoRA ----------------
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.float32)
    model.config.use_cache = False
    lora_cfg = LoraConfig(r=config.LORA_R, lora_alpha=config.LORA_ALPHA, lora_dropout=config.LORA_DROPOUT,
                          target_modules=config.LORA_TARGET_MODULES, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Εκπαιδεύσιμες παράμετροι (LoRA): {trainable:,} / {total_params:,} "
          f"({100 * trainable / total_params:.2f}%)")

    # ---------------- mixed precision (μόνο σε NVIDIA GPU) ----------------
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    def autocast_ctx():
        if use_amp:
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
        return torch.autocast(device_type="cpu", enabled=False)

    # ---------------- optimizer & scheduler ----------------
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(config.WARMUP_RATIO * total_steps), total_steps)
    print(f"  epochs={args.epochs}  optimizer steps={total_steps}  "
          f"(batch {args.batch_size} × accumulation {args.grad_accum})\n")

    # ---------------- training loop ----------------
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    step_log, epoch_log = [], []
    best_val = float("inf")
    global_step = 0
    t0 = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        running, micro = 0.0, 0
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=True)
        for i, batch in enumerate(bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast_ctx():
                out = model(**batch)
                loss = out.loss / args.grad_accum
            scaler.scale(loss).backward()
            running += loss.item()
            micro += 1

            last_batch = (i + 1) == len(train_loader)
            if (i + 1) % args.grad_accum == 0 or last_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                step_loss = running * args.grad_accum / micro   # μέσο loss των micro-batches του βήματος
                step_log.append({"step": global_step, "epoch": epoch, "train_loss": step_loss,
                                 "lr": scheduler.get_last_lr()[0]})
                bar.set_postfix(loss=f"{step_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
                running, micro = 0.0, 0
                if args.max_steps and global_step >= args.max_steps:
                    break

        val_loss = evaluate_loss(model, val_loader, device, autocast_ctx)
        epoch_log.append({"step": global_step, "epoch": epoch, "val_loss": val_loss})
        print(f"  ► epoch {epoch}: validation loss = {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            print(f"    ✔ νέο καλύτερο μοντέλο → {out_dir}")
        if args.max_steps and global_step >= args.max_steps:
            break

    elapsed = time.time() - t0
    # ---------------- logs & γράφημα ----------------
    with open(config.TRAINING_LOG_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "epoch", "train_loss", "lr", "val_loss"])
        val_by_step = {r["step"]: r["val_loss"] for r in epoch_log}
        for r in step_log:
            w.writerow([r["step"], r["epoch"], f"{r['train_loss']:.5f}", f"{r['lr']:.2e}",
                        f"{val_by_step[r['step']]:.5f}" if r["step"] in val_by_step else ""])
    plot_losses(step_log, epoch_log)
    info = {"base_model": args.base_model, "trained_at": datetime.now().isoformat(timespec="seconds"),
            "device": str(device), "epochs": args.epochs, "optimizer_steps": global_step,
            "train_examples": len(train_ds), "best_val_loss": best_val, "minutes": round(elapsed / 60, 1),
            "lora": {"r": config.LORA_R, "alpha": config.LORA_ALPHA, "targets": config.LORA_TARGET_MODULES},
            "trainable_params": trainable}
    (out_dir / "training_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\n✔ Τέλος εκπαίδευσης σε {elapsed / 60:.1f} λεπτά. Best val loss = {best_val:.4f}")
    print(f"  Adapter: {out_dir}\n  Γράφημα: {config.TRAINING_LOSS_PNG}\n  Log: {config.TRAINING_LOG_CSV}")


if __name__ == "__main__":
    main()
