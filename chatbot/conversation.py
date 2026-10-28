# ============ chatbot/conversation.py ============
"""
Ενορχήστρωση μιας συνομιλίας (orchestration):

  μήνυμα πελάτη
     │
     ├─► PyTorch intent classifier ─────────────► «Detected Intent» (Excel)
     │
     ├─► fine-tuned LLM (Qwen + LoRA) ──► JSON {intent, action, arguments}
     │        └─ αν αποτύχει → rule_parser (classifier + κανόνες)
     │
     ├─► guard rails (π.χ. αλλεργιογόνα ΠΑΝΤΑ μέσω get_allergens)
     ├─► ToolExecutor (menu.json, καλάθι, Excel, .ics, ημερολόγιο)
     ├─► ResponseBuilder (τελική απάντηση από τα πραγματικά δεδομένα)
     └─► ChatLogger (chat_log.xlsx με χρωματική επισήμανση)
"""
from __future__ import annotations

import logging
import re
import uuid
from collections import deque

import config
from chatbot.intent_classifier import IntentPredictor
from chatbot.menu_data import Menu, normalize
from chatbot.response_builder import FALLBACK_ERROR, SMALL_TALK_REPLIES, ResponseBuilder
from chatbot.rule_parser import detect_language, last_dish_in_history, match_in_cart, rule_decide
from chatbot.tools import ToolExecutor
from data_handlers.excel_logger import ChatLogger
from data_handlers.order_manager import Cart, OrderManager
from data_handlers.reservation_manager import ReservationManager

log = logging.getLogger(__name__)


class ChatSession:
    """Κατάσταση μιας συνομιλίας: μνήμη N μηνυμάτων, καλάθι, τι περιμένουμε."""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or f"S-{uuid.uuid4().hex[:6].upper()}"
        self.history: deque[tuple[str, str]] = deque(maxlen=config.MEMORY_WINDOW)
        self.cart = Cart()
        self.pending: str | None = None          # awaiting_order_confirmation | awaiting_reservation_details
        self.partial_reservation: dict = {}


class RestaurantChatbot:
    def __init__(self, use_llm: bool = True, llm=None):
        self.menu = Menu()
        self.responses = ResponseBuilder(self.menu)
        self.orders = OrderManager()
        self.reservations = ReservationManager()
        self.tools = ToolExecutor(self.menu, self.orders, self.reservations)
        self.chat_log = ChatLogger()
        self.classifier = IntentPredictor() if IntentPredictor.available() else None
        self.llm = llm
        if use_llm and self.llm is None:
            from chatbot.local_llm import LocalLLM
            self.llm = LocalLLM()

    def new_session(self) -> ChatSession:
        return ChatSession()

    # ------------------------------------------------------------------
    def handle(self, session: ChatSession, message: str) -> dict:
        message = (message or "").strip()
        lang = detect_language(message, session.history)
        cls_intent, cls_conf = ("n/a", 0.0)
        decision = {"intent": "out_of_scope", "action": "none", "language": lang}
        source = "rules"
        highlight = None
        try:
            if self.classifier:
                cls_intent, cls_conf = self.classifier.predict(message, session.pending)

            parsed = None
            if self.llm is not None:
                try:
                    parsed, _raw = self.llm.predict(list(session.history), session.cart.for_prompt(),
                                                    session.pending, message)
                except Exception:  # noqa: BLE001
                    log.exception("Σφάλμα κατά την παραγωγή από το LLM")
            if parsed is not None:
                decision, source = parsed, "llm"
            else:
                fallback_intent = cls_intent if cls_intent in config.INTENTS else "out_of_scope"
                decision = rule_decide(message, fallback_intent, self.menu, session.history, session.pending,
                                       bool(session.cart.items), session.cart.items)

            decision = self._guard(decision, message, cls_intent, cls_conf, session)
            lang = decision["language"]
            result = self.tools.execute(decision["action"], decision["arguments"], session)
            response = self.responses.build(decision["action"], result, lang, decision.get("reply"))
            self._update_state(session, decision, result)
            highlight = self._highlight(decision, result)
        except Exception:  # noqa: BLE001
            log.exception("Σφάλμα στην επεξεργασία μηνύματος: %r", message)
            response = FALLBACK_ERROR.get(lang, FALLBACK_ERROR["en"])

        session.history.append(("user", message))
        session.history.append(("assistant", response))
        try:
            self.chat_log.log(session.session_id, message, f"{cls_intent} ({cls_conf:.2f})",
                              f"{decision.get('intent')} [{source}]", decision.get("action", ""), response, highlight)
        except Exception:  # noqa: BLE001
            log.exception("Σφάλμα καταγραφής στο Excel")
        return {"response": response, "intent": decision.get("intent"), "action": decision.get("action"),
                "classifier_intent": cls_intent, "source": source, "highlight": highlight}

    # ------------------------------------------------------------------
    def _guard(self, d: dict, message: str, cls_intent: str, cls_conf: float, session: ChatSession) -> dict:
        """Κανόνες ασφαλείας πάνω στην απόφαση του μοντέλου."""
        d = dict(d)
        d.setdefault("arguments", {})
        d.setdefault("reply", None)

        # 1. Αλλεργιογόνα: ΠΑΝΤΑ μέσω του εργαλείου get_allergens.
        mentions_allergen = self.menu.find_allergen(self.menu.strip_dishes(message)) is not None
        if d["intent"] != "ask_allergen" and cls_intent == "ask_allergen" and cls_conf >= 0.9 and mentions_allergen:
            d = rule_decide(message, "ask_allergen", self.menu, session.history, session.pending,
                            bool(session.cart.items), session.cart.items)
        if d["intent"] == "ask_allergen" and d["action"] != "get_allergens":
            d["action"] = "get_allergens"

        # 1β. Ensemble: όταν ο classifier είναι σχεδόν βέβαιος (≥ 0.97) και διαφωνεί με το LLM,
        #     ακολουθούμε τον classifier (στα test dialogues οι προβλέψεις του με τόση
        #     βεβαιότητα ήταν 100% σωστές).
        if (cls_intent in config.INTENTS and cls_conf >= 0.97 and d["intent"] != cls_intent
                and {d["intent"], cls_intent} != {"small_talk", "out_of_scope"}):
            d = rule_decide(message, cls_intent, self.menu, session.history, session.pending,
                            bool(session.cart.items), session.cart.items)

        # 2. Ενέργειες που αλλάζουν την παραγγελία θέλουν συμφωνία με τον classifier
        #    (επιβεβαίωση/ακύρωση/αφαίρεση γίνεται μόνο αν συμφωνούν και τα δύο μοντέλα)
        risky = {"remove_item", "cancel_order", "confirm_order"}
        if cls_intent in config.INTENTS and d["intent"] != cls_intent and (
                d["intent"] in risky or (cls_intent in risky and cls_conf >= 0.9)):
            d = rule_decide(message, cls_intent, self.menu, session.history, session.pending,
                            bool(session.cart.items), session.cart.items)

        # 3. Συνέπεια intent ↔ action
        expected = config.INTENT_TO_ACTION.get(d["intent"])
        if expected and d["action"] != expected:
            d["action"] = expected

        # 4. Επαλήθευση οντοτήτων: τα ορίσματα του μοντέλου ελέγχονται απέναντι στο ίδιο το μήνυμα
        d["arguments"] = self._verify_entities(d["action"], dict(d["arguments"] or {}), message, session)

        # 4β. Ενέργεια χωρίς καμία οντότητα + ο classifier διαφωνεί → πιθανότατα λάθος intent
        a = d["arguments"]
        empty = ((d["action"] == "get_allergens" and not a.get("dish") and not a.get("allergen"))
                 or (d["action"] in ("get_price", "get_dish_info") and not a.get("dish"))
                 or (d["action"] in ("add_item", "remove_item") and not a.get("items")))
        # Επιπλέον, απάντηση αλλεργιογόνων μόνο αν το μήνυμα όντως μιλά για αλλεργίες
        if (d["action"] == "get_allergens" and not mentions_allergen
                and not re.search(r"allerg|αλλεργ|safe|free|χωρισ|xoris", normalize(message))):
            empty = True
        if empty and cls_intent in config.INTENTS:
            d = rule_decide(message, cls_intent, self.menu, session.history, session.pending,
                            bool(session.cart.items), session.cart.items)

        # 5. small talk: η απάντηση είναι ΠΑΝΤΑ ένα από τα εγκεκριμένα κείμενα
        #    (το πιο κοντινό σε αυτό που έγραψε το μοντέλο, ή με κανόνες)
        if d["action"] == "none":
            if d["intent"] == "out_of_scope":
                d["reply"] = SMALL_TALK_REPLIES["out_of_scope"][d["language"]]
            else:
                d["reply"] = self._canonical_reply(d.get("reply"), message, session, d["language"])
        return d

    @staticmethod
    def _canonical_reply(model_reply: str | None, message: str, session: ChatSession, lang: str) -> str:
        from difflib import SequenceMatcher
        from chatbot.rule_parser import small_talk_kind
        kind = small_talk_kind(message, session.pending)
        if kind not in ("greeting", "out_of_scope"):
            return SMALL_TALK_REPLIES[kind][lang]       # ρητές λέξεις-κλειδιά (ώρες, ευχαριστώ, ...)
        if model_reply:
            best_key, best = None, 0.0
            for key, texts in SMALL_TALK_REPLIES.items():
                score = SequenceMatcher(None, model_reply, texts[lang]).ratio()
                if score > best:
                    best_key, best = key, score
            if best >= 0.75:
                return SMALL_TALK_REPLIES[best_key][lang]
        return SMALL_TALK_REPLIES[kind][lang]

    def _verify_entities(self, action: str, args: dict, message: str, session: ChatSession) -> dict:
        """Το LLM «προτείνει», οι κανόνες «επαληθεύουν» (neuro-symbolic έλεγχος).

        Αν το μήνυμα αναφέρει ρητά πιάτο / αλλεργιογόνο / άτομα / ώρα και το μοντέλο
        έβγαλε κάτι διαφορετικό, κρατάμε ό,τι γράφει το μήνυμα. Έτσι ένα λάθος του
        μοντέλου δεν μπορεί να δώσει απάντηση αλλεργιογόνων για λάθος πιάτο."""
        found = self.menu.scan_dishes(message)
        found_ids = [d["id"] for d, _ in found]

        if action == "get_menu":
            # κατηγορία μόνο αν ο πελάτης την ανέφερε, αλλιώς όλο το μενού
            args["category"] = self.menu.find_category(message)

        elif action in ("get_dish_info", "get_price", "get_allergens"):
            llm_dish = self.menu.find_dish(args.get("dish"))
            history_dish = last_dish_in_history(self.menu, session.history)
            if found:
                if llm_dish is None or llm_dish["id"] not in found_ids:
                    args["dish"] = found[0][0]["name_en"]
            elif llm_dish is not None and (history_dish is None or llm_dish["id"] != history_dish["id"]):
                # το πιάτο δεν υπάρχει ούτε στο μήνυμα ούτε στη συζήτηση → πιθανό hallucination
                args["dish"] = history_dish["name_en"] if (history_dish and action != "get_allergens") else None
            elif llm_dish is None and history_dish is not None and action != "get_allergens":
                args["dish"] = history_dish["name_en"]   # «Πόσο κάνει;» → το τελευταίο πιάτο
            if action == "get_allergens":
                said = self.menu.find_allergen(self.menu.strip_dishes(message))
                # αν ο πελάτης δεν ανέφερε αλλεργιογόνο, δείχνουμε όλα (ασφαλέστερο)
                args["allergen"] = said

        elif action == "remove_item" and not found:
            in_cart = match_in_cart(self.menu, message, session.cart.items)
            if in_cart:
                args["items"] = [{"name": d["name_en"], "qty": None} for d in in_cart]

        elif action in ("add_item", "remove_item") and found:
            llm_items = [i for i in (args.get("items") or []) if isinstance(i, dict)]
            llm_by_id = {(self.menu.find_dish(i.get("name")) or {}).get("id"): i for i in llm_items}
            if set(llm_by_id) != set(found_ids):
                # το μοντέλο έβγαλε άλλα πιάτα από όσα γράφει το μήνυμα → κρατάμε το μήνυμα
                args["items"] = [{"name": d["name_en"], "qty": (q or 1) if action == "add_item" else q}
                                 for d, q in found]
            else:
                # ίδια πιάτα· αν το μήνυμα γράφει ρητά ποσότητα, αυτή υπερισχύει
                for d, q in found:
                    if q is not None:
                        llm_by_id[d["id"]]["qty"] = q

        elif action == "make_reservation":
            from chatbot.rule_parser import extract_reservation
            rules = extract_reservation(message, session.pending)
            for key, value in rules.items():
                if value not in (None, ""):
                    args[key] = value
        return args

    @staticmethod
    def _update_state(session: ChatSession, d: dict, result: dict) -> None:
        action = d["action"]
        if action == "add_item" and result.get("added"):
            session.pending = "awaiting_order_confirmation"
        elif action in ("confirm_order", "cancel_order"):
            session.pending = None
        elif action == "make_reservation":
            if result.get("status") == "ok":
                session.pending, session.partial_reservation = None, {}
            else:
                session.pending = "awaiting_reservation_details"
                session.partial_reservation = result.get("partial", {})
        elif session.pending == "awaiting_reservation_details" and action != "none":
            session.pending, session.partial_reservation = None, {}

    @staticmethod
    def _highlight(d: dict, result: dict) -> str | None:
        action = d["action"]
        if action == "get_allergens":
            return "Allergen question"
        if action == "get_price":
            return "Price question"
        if action == "make_reservation":
            return "Reservation request"
        if action == "confirm_order" and result.get("status") == "ok":
            return "Order confirmation"
        return None
