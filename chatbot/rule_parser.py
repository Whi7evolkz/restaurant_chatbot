# ============ chatbot/rule_parser.py ============
"""
Fallback χωρίς LLM: intent από τον PyTorch classifier + εξαγωγή στοιχείων με κανόνες.

Χρησιμοποιείται όταν:
  • το fine-tuned μοντέλο δεν έδωσε έγκυρο JSON (ούτε στη 2η προσπάθεια), ή
  • τρέχουμε με --rules-only (π.χ. πριν τελειώσει η εκπαίδευση).
Έτσι το demo δεν «σπάει» ποτέ.
"""
from __future__ import annotations

import re
from datetime import datetime

from chatbot.menu_data import NUMBER_WORDS, Menu, has_greek, normalize
from chatbot.response_builder import SMALL_TALK_REPLIES
from data_handlers.reservation_manager import TZ, parse_date

GREEKLISH_MARKERS = {"thelo", "exei", "exete", "poso", "kanei", "gia", "stis", "atoma", "nai", "oxi", "geia",
                     "parakalo", "ti", "mporo", "kratisi", "trapezi", "efxaristo", "eyxaristo", "kalispera",
                     "vale", "mou", "kai", "ena", "mia", "dyo", "tha", "ithela", "periexei", "allergiogona", "vgale",
                     "tin", "ton", "ta", "afairese", "akyrosi", "akyrose", "kalathi", "exo", "den", "pia", "simera",
                     "avrio", "oloklirose", "epivevaiose", "paraggelia", "deikse", "fagita", "ylika", "timi",
                     "kostizei", "glyka", "pota", "xoris", "eimai", "allergikos", "ok"}


def detect_language(message: str, history=None) -> str:
    if has_greek(message):
        return "el"
    tokens = set(normalize(message).split())
    if tokens & GREEKLISH_MARKERS:
        return "el"
    if len(tokens) <= 2 and history:      # «4», «ok» → γλώσσα του προηγούμενου μηνύματος
        for role, text in reversed(list(history)):
            if role == "assistant":
                return "el" if has_greek(text) else "en"
    return "en"


TIME_RE = re.compile(
    r"(?<![\d/])(\d{1,2}\s*[:.]\s*\d{2})(?![\d/])"
    r"|(?<![\d/])(\d{1,2})\s*(pm|am|μμ|πμ)(?!\w)"
    r"|(?:at|στισ|stis)\s+(\d{1,2}(?:\s*[:.]\s*\d{2})?)(?![\d/])(\s*(?:το βραδυ|το μεσημερι|to vrady|in the evening|pm|μμ|και μιση|kai misi))?"
    r"|(?<![\d/])(\d{1,2})\s*(?:in the evening|το βραδυ|to vrady|o clock|το μεσημερι)"
)
PARTY_RE = re.compile(
    r"(?:for|για|gia|we are|ειμαστε|eimaste)\s+(\d{1,2}|[a-zα-ω]+)(?!\s*[:./]\s*\d)"
    r"|(\d{1,2}|[a-zα-ω]+)\s*(?:people|persons|person|guests|ατομα|ατομο|atoma|pax)"
)


def extract_reservation(message: str, pending: str | None) -> dict:
    norm = normalize(message)
    args = {"party_size": None, "date_text": None, "time_text": None, "name": None}

    m = TIME_RE.search(norm)
    rest = norm
    if m:
        args["time_text"] = m.group(0).replace("στισ ", "").replace("stis ", "").replace("at ", "").strip()
        rest = norm[:m.start()] + " " + norm[m.end():]

    for pm in PARTY_RE.finditer(rest):
        tok = pm.group(1) or pm.group(2)
        n = int(tok) if tok.isdigit() else NUMBER_WORDS.get(tok)
        if n:
            args["party_size"] = n
            rest = rest[:pm.start()] + " " + rest[pm.end():]
            break
    if args["party_size"] is None and pending == "awaiting_reservation_details":
        only = norm.split()
        if len(only) == 1 and (only[0].isdigit() or only[0] in NUMBER_WORDS):
            args["party_size"] = int(only[0]) if only[0].isdigit() else NUMBER_WORDS[only[0]]

    if parse_date(rest, datetime.now(TZ), use_dateparser=False):
        args["date_text"] = rest.strip()

    nm = re.search(r"(?:under the name|the name is|name is|στο ονομα|ονομα|sto onoma)\s*:?\s*(\S+)", message,
                   flags=re.IGNORECASE)
    if nm is None:
        nm = re.search(r"(?:under the name|the name is|name is|στο όνομα|όνομα|sto onoma)\s*:?\s*(\S+)", message,
                       flags=re.IGNORECASE)
    if nm:
        args["name"] = nm.group(1).strip(".,;!?")
    return args


def last_dish_in_history(menu: Menu, history) -> dict | None:
    for _, text in reversed(list(history or [])):
        found = menu.scan_dishes(text)
        if found:
            return found[0][0]
    return None


def small_talk_kind(message: str, pending: str | None) -> str:
    n = normalize(message)
    if pending == "awaiting_order_confirmation" and re.match(r"^(no|not yet|οχι|oxi|wait|περιμενε|hold on)\b", n):
        return "keep_cart"
    rules = [
        ("thanks", ["thank", "thanks", "ευχαριστ", "efxarist", "eyxarist"]),
        ("goodbye", ["bye", "goodbye", "see you", "αντιο", "τα λεμε", "καληνυχτα", "γεια χαρα", "ta leme", "antio"]),
        ("hours", ["open", "close", "hours", "ανοιγ", "κλειν", "ωραρι", "ανοιχτ", "anoig", "klein"]),
        ("capabilities", ["who are you", "what can you", "robot", "help me", "ποιοσ εισαι", "τι μπορεισ", "ρομποτ",
                          "βοηθησεισ", "ti mporeis", "poios eisai"]),
    ]
    for kind, keys in rules:
        if any(k in n for k in keys):
            return kind
    greetings = ["hi", "hello", "hey", "good", "γεια", "καλησπερα", "καλημερα", "χαιρετε", "geia", "kalispera",
                 "kalimera", "τι κανεισ", "τι κανετε", "how are you"]
    if any(re.search(rf"(?<!\w){g}", n) for g in greetings) or len(n.split()) <= 2:
        return "greeting"
    return "out_of_scope"


def match_in_cart(menu: Menu, message: str, cart_items: list[dict]) -> list[dict]:
    """«δεν θέλω τη σαλάτα» → η σαλάτα που ΥΠΑΡΧΕΙ στο καλάθι (Greek Salad)."""
    words = {w for w in normalize(message).split() if len(w) >= 4}
    hits = []
    for it in cart_items or []:
        d = menu.find_dish(it.get("id") or it.get("name"))
        if not d:
            continue
        names = [d["name_en"], d["name_el"], *d.get("aliases", [])]
        tokens = {t for n in names for t in normalize(n).split() if len(t) >= 4}
        if words & tokens:
            hits.append(d)
    return hits


def rule_decide(message: str, intent: str, menu: Menu, history=None, pending: str | None = None,
                cart_has_items: bool = False, cart_items: list[dict] | None = None) -> dict:
    """Απόφαση {intent, language, action, arguments, reply} χωρίς LLM."""
    lang = detect_language(message, history)
    decision = {"intent": intent, "language": lang, "action": "none", "arguments": {}, "reply": None}
    found = menu.scan_dishes(message)
    dish = found[0][0] if found else None

    if intent == "ask_menu":
        decision.update(action="get_menu", arguments={"category": menu.find_category(message)})
    elif intent in ("ask_dish_info", "ask_price") and not dish and menu.find_category(message):
        # «what pizzas have you got?» → λίστα κατηγορίας, όχι ένα πιάτο
        decision.update(intent="ask_menu", action="get_menu", arguments={"category": menu.find_category(message)})
    elif intent in ("ask_dish_info", "ask_price"):
        dish = dish or last_dish_in_history(menu, history)
        decision.update(action="get_dish_info" if intent == "ask_dish_info" else "get_price",
                        arguments={"dish": dish["name_en"] if dish else None})
    elif intent == "ask_allergen":
        allergen = menu.find_allergen(menu.strip_dishes(message))
        decision.update(action="get_allergens",
                        arguments={"dish": dish["name_en"] if dish else None, "allergen": allergen})
    elif intent == "order_food":
        decision.update(action="add_item", arguments={"items": [{"name": d["name_en"], "qty": q or 1} for d, q in found]})
    elif intent == "remove_item":
        n = normalize(message).split()
        qty = 1 if any(w in n for w in ("one", "ενα", "μια", "ena", "mia")) else None
        targets = [d for d, _ in found] or match_in_cart(menu, message, cart_items)
        decision.update(action="remove_item", arguments={"items": [{"name": d["name_en"], "qty": qty} for d in targets]})
    elif intent in ("view_cart", "confirm_order", "cancel_order"):
        decision.update(action=intent)
    elif intent == "make_reservation":
        decision.update(action="make_reservation", arguments=extract_reservation(message, pending))
    elif intent == "small_talk":
        decision.update(reply=SMALL_TALK_REPLIES[small_talk_kind(message, pending)][lang])
    else:
        decision.update(intent="out_of_scope", reply=SMALL_TALK_REPLIES["out_of_scope"][lang])
    return decision
