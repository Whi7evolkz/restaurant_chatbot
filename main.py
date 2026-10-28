# ============ main.py ============
"""
CLI για συνομιλία με τον fine-tuned βοηθό του εστιατορίου.

    python main.py                 # με το fine-tuned μοντέλο (models/lora_adapter)
    python main.py --demo          # τρέχει αυτόματα τους διαλόγους-παραδείγματα
    python main.py --rules-only    # χωρίς LLM (μόνο PyTorch classifier + κανόνες)

Εντολές μέσα στη συνομιλία: /cart, /reset, /help, exit
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import config


def setup_logging() -> None:
    handler = logging.FileHandler(config.ERROR_LOG, encoding="utf-8")
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    root.addHandler(handler)
    for noisy in ("transformers", "httpx", "huggingface_hub", "urllib3", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


def fix_console_encoding() -> None:
    """Ελληνικά στο terminal των Windows."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass


INSTRUCTIONS_NO_ADAPTER = f"""
✖ Δεν βρέθηκε εκπαιδευμένο μοντέλο στο: {config.ADAPTER_DIR}

Τρέξτε πρώτα με τη σειρά:
    python -m training.dataset_builder
    python -m training.train_intent
    python -m training.train            (σε CPU: --epochs 1 --limit 500, ή στο Google Colab)

Αν εκπαιδεύσατε στο Colab, αντιγράψτε τον φάκελο lora_adapter μέσα στο models/.
Για δοκιμή χωρίς LLM:  python main.py --rules-only
"""

INSTRUCTIONS_NO_CLASSIFIER = """
✖ Δεν βρέθηκε ο intent classifier (models/intent_classifier.pt).
Τρέξτε:
    python -m training.dataset_builder
    python -m training.train_intent
"""

DEMO_DIALOGUES = [
    ["Does the carbonara contain gluten?"],
    ["I want 2 pizzas margherita and one coke.", "Yes."],
    ["I'd like to reserve a table for 4 on Friday at 8pm."],
    ["Γεια σας! Τι γλυκά έχετε;", "Έχει ξηρούς καρπούς ο μπακλαβάς;", "Πόσο κάνει το τιραμισού;",
     "Θέλω ένα τιραμισού και δύο φραπέ", "Ναι"],
    ["Θέλω να κλείσω τραπέζι", "Για 2 άτομα", "Αύριο στις 9 το βράδυ"],
]


def print_bot(text: str) -> None:
    print("🤖 Bot:", text.replace("\n", "\n        "), "\n")


def run_demo(bot) -> None:
    for i, dialogue in enumerate(DEMO_DIALOGUES, start=1):
        session = bot.new_session()
        print(f"────────── Διάλογος {i} ({session.session_id}) ──────────")
        for msg in dialogue:
            print("🙂 Εσύ:", msg)
            t0 = time.time()
            out = bot.handle(session, msg)
            print_bot(out["response"])
            print(f"   [intent={out['intent']} via {out['source']}, action={out['action']}, "
                  f"highlight={out['highlight'] or '-'}, {time.time() - t0:.1f}s]\n")
    print(f"✔ Δείτε τα αρχεία στον φάκελο: {config.OUTPUTS_DIR}")


def run_chat(bot) -> None:
    session = bot.new_session()
    print(f"\n🍽  {config.RESTAURANT_NAME} – ψηφιακός βοηθός (session {session.session_id})")
    print("   Γράψτε στα ελληνικά ή στα αγγλικά. Εντολές: /cart, /reset, /help, exit\n")
    while True:
        try:
            msg = input("🙂 Εσύ: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nΑντίο!")
            break
        if not msg:
            continue
        low = msg.lower()
        if low in ("exit", "quit", "έξοδος", "/exit"):
            print("Αντίο! 👋")
            break
        if low == "/reset":
            session = bot.new_session()
            print(f"   (νέα συνεδρία {session.session_id})\n")
            continue
        if low == "/cart":
            items = session.cart.items
            print("   Καλάθι:", ", ".join(f"{i['qty']}× {i['name_en']}" for i in items) or "άδειο",
                  f"| Σύνολο: €{session.cart.total():.2f}\n")
            continue
        if low == "/help":
            print("   Παραδείγματα: «Τι έχει το μενού;», «Έχει γλουτένη η καρμπονάρα;», "
                  "«Θέλω 2 μαργαρίτες», «Τραπέζι για 4 την Παρασκευή στις 8»\n")
            continue
        out = bot.handle(session, msg)
        print_bot(out["response"])


def main() -> None:
    fix_console_encoding()
    setup_logging()
    ap = argparse.ArgumentParser(description="AI Restaurant Ordering Assistant (fine-tuned, local)")
    ap.add_argument("--rules-only", action="store_true", help="χωρίς LLM: μόνο PyTorch classifier + κανόνες")
    ap.add_argument("--demo", action="store_true", help="εκτέλεση των διαλόγων-παραδειγμάτων")
    args = ap.parse_args()

    from chatbot.intent_classifier import IntentPredictor
    if not IntentPredictor.available():
        print(INSTRUCTIONS_NO_CLASSIFIER)
        sys.exit(1)

    use_llm = not args.rules_only
    if use_llm:
        from chatbot.local_llm import adapter_available
        if not adapter_available():
            print(INSTRUCTIONS_NO_ADAPTER)
            sys.exit(1)
        print("⏳ Φόρτωση του fine-tuned μοντέλου (την 1η φορά κατεβαίνει το base μοντέλο ~1 GB)...")

    from chatbot.conversation import RestaurantChatbot
    try:
        bot = RestaurantChatbot(use_llm=use_llm)
    except OSError as exc:
        logging.getLogger(__name__).exception("Αποτυχία φόρτωσης μοντέλου")
        print(f"✖ Δεν φόρτωσε το μοντέλο ({exc}).\n"
              "  Ελέγξτε τη σύνδεση στο internet για το πρώτο κατέβασμα ή τρέξτε: python main.py --rules-only")
        sys.exit(1)
    if use_llm:
        print(f"✔ Μοντέλο: {bot.llm.base_model} + LoRA  ({bot.llm.device})")
    else:
        print("ℹ Λειτουργία --rules-only: PyTorch intent classifier + κανόνες (χωρίς LLM)")

    if args.demo:
        run_demo(bot)
    else:
        run_chat(bot)


if __name__ == "__main__":
    main()
