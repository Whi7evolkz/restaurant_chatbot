# ============ chatbot/prompt_format.py ============
"""
Η ΜΟΡΦΗ της εισόδου που βλέπει το μοντέλο.

Πολύ σημαντικό: η ίδια ακριβώς συνάρτηση χρησιμοποιείται για να φτιαχτεί το
dataset εκπαίδευσης ΚΑΙ κατά τη συνομιλία. Αν διαφέρουν, το fine-tuned μοντέλο
θα βλέπει κάτι διαφορετικό από αυτό που έμαθε και θα αποδίδει χειρότερα.
"""
from __future__ import annotations

import json

import config


def format_cart(cart_items: list[dict]) -> str:
    """[{'name': 'Pizza Margherita', 'qty': 2}, ...] → '2x Pizza Margherita, 1x Coca-Cola'"""
    if not cart_items:
        return "empty"
    return ", ".join(f"{it['qty']}x {it['name']}" for it in cart_items)


def build_user_content(history: list[tuple[str, str]], cart_items: list[dict],
                       pending: str | None, message: str) -> str:
    """Συνθέτει το μήνυμα 'user' για το μοντέλο.

    history: λίστα (role, text) με role 'user' ή 'assistant' (τελευταία N μηνύματα)
    pending: None | 'awaiting_order_confirmation' | 'awaiting_reservation_details'
    """
    lines = ["[HISTORY]"]
    recent = history[-config.MEMORY_WINDOW:]
    if not recent:
        lines.append("(none)")
    for role, text in recent:
        who = "User" if role == "user" else "Assistant"
        text = " ".join(text.split())
        if len(text) > config.HISTORY_CHARS:
            text = text[: config.HISTORY_CHARS] + "…"
        lines.append(f"{who}: {text}")
    lines.append(f"[CART] {format_cart(cart_items)}")
    lines.append(f"[PENDING] {pending or 'none'}")
    lines.append(f"[MESSAGE] {message.strip()}")
    return "\n".join(lines)


def build_messages(user_content: str, target: dict | None = None) -> list[dict]:
    """Μηνύματα σε μορφή chat template (system / user / assistant)."""
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target_to_text(target)})
    return messages


def target_to_text(target: dict) -> str:
    """Το JSON στόχου ως συμπαγές κείμενο (λιγότερα tokens)."""
    ordered = {
        "intent": target["intent"],
        "language": target["language"],
        "action": target["action"],
        "arguments": target.get("arguments", {}),
        "reply": target.get("reply"),
    }
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Ανάγνωση της εξόδου του μοντέλου
# ---------------------------------------------------------------------------
def _first_json_object(text: str) -> str | None:
    """Βρίσκει το πρώτο ισορροπημένο {...} μέσα στο κείμενο (λαμβάνει υπόψη strings)."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_model_output(raw: str) -> dict | None:
    """Επιστρέφει έγκυρο dict ή None αν το μοντέλο δεν έβγαλε σωστό JSON."""
    block = _first_json_object(raw or "")
    if block is None:
        return None
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    intent, action = data.get("intent"), data.get("action")
    if intent not in config.INTENTS or action not in config.ACTIONS:
        return None
    args = data.get("arguments") or {}
    if not isinstance(args, dict):
        return None
    lang = data.get("language") if data.get("language") in ("el", "en") else "en"
    reply = data.get("reply")
    return {"intent": intent, "language": lang, "action": action, "arguments": args,
            "reply": reply if isinstance(reply, str) and reply.strip() else None}
