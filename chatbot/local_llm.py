# ============ chatbot/local_llm.py ============
"""
Φόρτωση του fine-tuned μοντέλου (base Qwen + LoRA adapter) και παραγωγή απόφασης.

Δεν υπάρχει κανένα API: το μοντέλο τρέχει τοπικά με PyTorch.
Το πρώτο τρέξιμο κατεβάζει το base μοντέλο (~1 GB) από το Hugging Face
και το κρατά στην cache, μετά δουλεύει και offline.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

import config
from chatbot.prompt_format import build_messages, build_user_content, parse_model_output

log = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def adapter_available(adapter_dir: Path = config.ADAPTER_DIR) -> bool:
    return (Path(adapter_dir) / "adapter_config.json").exists()


class LocalLLM:
    def __init__(self, adapter_dir: Path = config.ADAPTER_DIR, base_model: str | None = None,
                 use_adapter: bool = True):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        adapter_dir = Path(adapter_dir)
        if base_model is None:
            base_model = config.BASE_MODEL_NAME
            cfg_file = adapter_dir / "adapter_config.json"
            if cfg_file.exists():
                # το adapter θυμάται σε ποιο base μοντέλο εκπαιδεύτηκε (χρήσιμο και για τη σύγκριση base vs fine-tuned)
                base_model = json.loads(cfg_file.read_text(encoding="utf-8")).get("base_model_name_or_path") or base_model

        self.device = get_device()
        self.base_model = base_model
        self.use_adapter = use_adapter
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
        if use_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter_dir))
            model = model.merge_and_unload()     # ενσωμάτωση LoRA στα βάρη → ταχύτερη παραγωγή
        self.model = model.to(self.device).eval()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_raw(self, messages: list[dict], greedy: bool = True) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        gen_kwargs = dict(max_new_tokens=config.MAX_NEW_TOKENS, eos_token_id=self.tokenizer.eos_token_id,
                          pad_token_id=self.tokenizer.pad_token_id)
        if greedy:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        else:
            gen_kwargs.update(do_sample=True, temperature=0.2, top_p=0.9)
        out = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def predict(self, history, cart_items, pending, message) -> tuple[dict | None, str]:
        """Επιστρέφει (απόφαση ως dict ή None, ακατέργαστο κείμενο του μοντέλου)."""
        messages = build_messages(build_user_content(history, cart_items, pending, message))
        raw = self.generate_raw(messages, greedy=True)
        parsed = parse_model_output(raw)
        if parsed is None:
            # 2η προσπάθεια με χαμηλή θερμοκρασία δειγματοληψίας
            log.warning("Μη έγκυρο JSON από το μοντέλο: %r — νέα προσπάθεια", raw[:200])
            raw = self.generate_raw(messages, greedy=False)
            parsed = parse_model_output(raw)
        return parsed, raw
