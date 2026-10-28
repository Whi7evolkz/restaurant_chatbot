# ============ chatbot/response_builder.py ============
"""
Δημιουργία της τελικής απάντησης προς τον πελάτη (ελληνικά ή αγγλικά).

ΓΙΑΤΙ ΕΔΩ ΚΑΙ ΟΧΙ ΑΠΟ ΤΟ ΜΟΝΤΕΛΟ:
Ένα μικρό LLM μπορεί να «επινοήσει» τιμές ή αλλεργιογόνα (hallucination).
Γι' αυτό το μοντέλο αποφασίζει μόνο ΤΙ θα γίνει (action) και όλα τα
πραγματικά στοιχεία (τιμές, αλλεργιογόνα, σύνολα, κωδικοί) γράφονται εδώ
από τα αποτελέσματα των tools, δηλαδή από το menu.json (grounding).
"""
from __future__ import annotations

import config

GREEK_DAYS = ["Δευτέρα", "Τρίτη", "Τετάρτη", "Πέμπτη", "Παρασκευή", "Σάββατο", "Κυριακή"]
EN_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ---------------------------------------------------------------------------
# Σταθερές απαντήσεις small talk / out of scope (τις μαθαίνει και το μοντέλο)
# ---------------------------------------------------------------------------
SMALL_TALK_REPLIES = {
    "greeting": {
        "en": f"Hello and welcome to {config.RESTAURANT_NAME_EN}! I can show you the menu, answer questions about prices and allergens, take your order or book a table.",
        "el": f"Γεια σας και καλώς ήρθατε στην {config.RESTAURANT_NAME}! Μπορώ να σας δείξω το μενού, να απαντήσω για τιμές και αλλεργιογόνα, να πάρω την παραγγελία σας ή να κλείσω τραπέζι.",
    },
    "thanks": {
        "en": "You're welcome! Is there anything else I can help you with?",
        "el": "Παρακαλώ! Μπορώ να σας βοηθήσω με κάτι άλλο;",
    },
    "goodbye": {
        "en": "Thank you for visiting! Have a lovely day.",
        "el": "Ευχαριστούμε πολύ! Να έχετε μια όμορφη μέρα.",
    },
    "capabilities": {
        "en": "I'm the restaurant's AI assistant. I can show the menu, tell you prices, ingredients and allergens, take your order and make a table reservation.",
        "el": "Είμαι ο ψηφιακός βοηθός του εστιατορίου. Μπορώ να σας δείξω το μενού, τιμές, υλικά και αλλεργιογόνα, να πάρω παραγγελία και να κλείσω τραπέζι.",
    },
    "hours": {
        "en": f"We are open every day from {config.OPENING_HOUR}:00 until midnight. Reservations are possible from {config.OPENING_HOUR}:00 to {config.LAST_SEATING_HOUR}:00.",
        "el": f"Είμαστε ανοιχτά καθημερινά από τις {config.OPENING_HOUR}:00 έως τα μεσάνυχτα. Κρατήσεις δεχόμαστε από τις {config.OPENING_HOUR}:00 έως τις {config.LAST_SEATING_HOUR}:00.",
    },
    "keep_cart": {
        "en": "No problem, your cart is kept. Tell me what else you would like, or say \"cancel\" to clear it.",
        "el": "Κανένα πρόβλημα, το καλάθι σας παραμένει. Πείτε μου τι άλλο θέλετε ή γράψτε «ακύρωση» για να το αδειάσω.",
    },
    "out_of_scope": {
        "en": "Sorry, I can only help with our menu, prices, allergens, orders and table reservations.",
        "el": "Λυπάμαι, μπορώ να βοηθήσω μόνο με το μενού, τις τιμές, τα αλλεργιογόνα, τις παραγγελίες και τις κρατήσεις.",
    },
}

FALLBACK_ERROR = {
    "en": "Sorry, something went wrong on my side. Please try again in a moment.",
    "el": "Συγγνώμη, κάτι πήγε στραβά. Παρακαλώ δοκιμάστε ξανά σε λίγο.",
}


def money(value: float, lang: str) -> str:
    if lang == "el":
        return f"{value:.2f}".replace(".", ",") + f" {config.CURRENCY_SYMBOL}"
    return f"{config.CURRENCY_SYMBOL}{value:.2f}"


def _join(items: list[str], lang: str) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    conj = " και " if lang == "el" else " and "
    return ", ".join(items[:-1]) + conj + items[-1]


class ResponseBuilder:
    def __init__(self, menu):
        self.menu = menu

    # βοηθητικά
    def _dn(self, dish: dict, lang: str) -> str:
        return self.menu.dish_name(dish, lang)

    def _the(self, dish: dict, capital: bool = False) -> str:
        """Ελληνικό όνομα με άρθρο: 'ο Μουσακάς', 'η Πίτσα Μαργαρίτα'."""
        art = dish.get("article_el", "")
        if capital:
            art = art.capitalize()
        return f"{art} {dish['name_el']}".strip()

    @staticmethod
    def _its(dish: dict) -> str:
        """Κτητική αντωνυμία: του / της / τους ανάλογα με το γένος."""
        return {"η": "της", "τα": "τους", "οι": "τους"}.get(dish.get("article_el"), "του")

    def _an(self, key: str, lang: str) -> str:
        return self.menu.allergen_name(key, lang)

    def _not_found(self, name, lang: str) -> str:
        name = name or "?"
        if lang == "el":
            return f"Δεν βρήκα το «{name}» στο μενού μας. Θέλετε να σας δείξω το μενού;"
        return f"I couldn't find \"{name}\" on our menu. Would you like to see the menu?"

    # ------------------------------------------------------------------
    def build(self, action: str, result: dict, lang: str, reply: str | None = None) -> str:
        handler = getattr(self, f"_r_{action}", None)
        if handler is None:
            return reply or SMALL_TALK_REPLIES["out_of_scope"][lang]
        return handler(result, lang, reply)

    # ---------------- menu ----------------
    def _r_get_menu(self, r, lang, _reply):
        parts = []
        for cat, dishes in r["categories"].items():
            title = self.menu.category_name(cat, lang)
            rows = [f"  • {self._dn(d, lang)} – {money(d['price'], lang)}" for d in dishes]
            parts.append(f"{title}:\n" + "\n".join(rows))
        head = "Το μενού μας:" if lang == "el" else "Here is our menu:"
        tail = ("\nΘέλετε να παραγγείλετε κάτι ή να μάθετε για υλικά/αλλεργιογόνα;" if lang == "el"
                else "\nWould you like to order something or ask about ingredients/allergens?")
        return head + "\n" + "\n".join(parts) + tail

    # ---------------- dish info ----------------
    def _r_get_dish_info(self, r, lang, _reply):
        if r["status"] != "ok":
            return self._not_found(r.get("query"), lang)
        d = r["dish"]
        ingredients = ", ".join(d["ingredients_el"] if lang == "el" else d["ingredients_en"])
        allergens = _join([self._an(a, lang) for a in d["allergens"]], lang)
        if lang == "el":
            al = f"Αλλεργιογόνα: {allergens}." if allergens else "Δεν περιέχει κανένα από τα 14 βασικά αλλεργιογόνα."
            return f"{self._dn(d, lang)} ({money(d['price'], lang)}): {ingredients}.\n{al}"
        al = f"Allergens: {allergens}." if allergens else "It contains none of the 14 major allergens."
        return f"{self._dn(d, lang)} ({money(d['price'], lang)}): {ingredients}.\n{al}"

    # ---------------- price ----------------
    def _r_get_price(self, r, lang, _reply):
        if r["status"] != "ok":
            return self._not_found(r.get("query"), lang)
        d = r["dish"]
        if lang == "el":
            return f"{self._dn(d, lang)}: {money(d['price'], lang)}. Θέλετε να το προσθέσω στην παραγγελία;"
        return f"The {self._dn(d, lang)} costs {money(d['price'], lang)}. Would you like to add it to your order?"

    # ---------------- allergens ----------------
    def _r_get_allergens(self, r, lang, _reply):
        note = ("\n(Για σοβαρή αλλεργία ενημερώστε και το προσωπικό, λόγω πιθανής επιμόλυνσης στην κουζίνα.)"
                if lang == "el" else
                "\n(For severe allergies please also inform our staff, as cross-contamination in the kitchen is possible.)")

        if r["status"] == "dish_not_found":
            if lang == "el":
                return f"Δεν βρήκα το «{r.get('query')}» στο μενού, οπότε δεν μπορώ να επιβεβαιώσω τα αλλεργιογόνα του."
            return f"I couldn't find \"{r.get('query')}\" on our menu, so I can't confirm its allergens."

        if r["status"] == "free_of":   # «ποια πιάτα δεν έχουν X;»
            names = _join([self._dn(d, lang) for d in r["dishes"]], lang)
            al = self.menu.allergen_name(r["allergen"], lang, accusative=True)
            if lang == "el":
                return f"Πιάτα χωρίς {al}: {names}." + note
            return f"Dishes without {al}: {names}." + note

        if r["status"] == "need_more":
            return ("Για ποιο πιάτο ή ποιο αλλεργιογόνο θέλετε να μάθετε;" if lang == "el"
                    else "Which dish or which allergen would you like to know about?")

        d = r["dish"]
        dish_allergens = _join([self._an(a, lang) for a in d["allergens"]], lang)
        asked = r.get("allergen")
        alts = _join([self._dn(x, lang) for x in r.get("alternatives", [])], lang)

        if asked is None:
            if lang == "el":
                if not d["allergens"]:
                    return f"{self._the(d, capital=True)} δεν περιέχει κανένα από τα 14 βασικά αλλεργιογόνα." + note
                return f"{self._the(d, capital=True)} περιέχει: {dish_allergens}." + note
            if not d["allergens"]:
                return f"The {self._dn(d, lang)} contains none of the 14 major allergens." + note
            return f"The {self._dn(d, lang)} contains: {dish_allergens}." + note

        al = self.menu.allergen_name(asked, lang, accusative=True)
        if r["contains"]:
            if lang == "el":
                text = f"Ναι, {self._the(d)} περιέχει {al}. Όλα τα αλλεργιογόνα {self._its(d)}: {dish_allergens}."
                if alts:
                    text += f" Χωρίς {al} σας προτείνω: {alts}."
            else:
                text = f"Yes, the {self._dn(d, lang)} contains {al}. All its allergens: {dish_allergens}."
                if alts:
                    text += f" A {al}-free alternative I can suggest: {alts}."
            return text + note
        if lang == "el":
            rest = f" Τα αλλεργιογόνα {self._its(d)} είναι: {dish_allergens}." if dish_allergens else ""
            return f"Όχι, σύμφωνα με το μενού μας {self._the(d)} δεν περιέχει {al}.{rest}" + note
        rest = f" Its allergens are: {dish_allergens}." if dish_allergens else ""
        return f"No, according to our menu the {self._dn(d, lang)} does not contain {al}.{rest}" + note

    # ---------------- cart ----------------
    def _cart_lines(self, cart, lang):
        return "\n".join(
            f"  • {it['qty']}× {it['name_el'] if lang == 'el' else it['name_en']} – "
            f"{money(it['qty'] * it['unit_price'], lang)}" for it in cart.items)

    def _r_add_item(self, r, lang, _reply):
        added = r.get("added", [])
        missing = r.get("not_found", [])
        parts = []
        if added:
            if lang == "el":
                desc = _join([f"{q}× {self._dn(d, lang)} ({money(d['price'], lang)} το ένα)" if q > 1
                              else f"1× {self._dn(d, lang)} ({money(d['price'], lang)})" for d, q in added], lang)
                parts.append(f"Πρόσθεσα {desc}. Σύνολο: {money(r['total'], lang)}.")
            else:
                desc = _join([f"{q}× {self._dn(d, lang)} ({money(d['price'], lang)} each)" if q > 1
                              else f"1× {self._dn(d, lang)} ({money(d['price'], lang)})" for d, q in added], lang)
                parts.append(f"Added {desc}. Total: {money(r['total'], lang)}.")
        if missing:
            m = ", ".join(f"«{x}»" for x in missing)
            parts.append(f"Δεν βρήκα στο μενού: {m}." if lang == "el" else f"I couldn't find on the menu: {m}.")
        if not added and not missing:
            return ("Δεν βρήκα αυτό το προϊόν στο μενού μας. Θέλετε να σας δείξω το μενού;" if lang == "el"
                    else "I couldn't find that on our menu. Would you like to see the menu?")
        if added:
            parts.append("Να επιβεβαιώσω την παραγγελία ή θέλετε κάτι άλλο;" if lang == "el"
                         else "Shall I confirm the order, or would you like anything else?")
        return " ".join(parts)

    def _r_remove_item(self, r, lang, _reply):
        if not r.get("removed"):
            return ("Δεν βρήκα αυτό το προϊόν στο καλάθι σας." if lang == "el"
                    else "I couldn't find that item in your cart.")
        names = _join([self._dn(d, lang) for d in r["removed"]], lang)
        total = money(r["total"], lang)
        if lang == "el":
            return f"Αφαίρεσα: {names}. Νέο σύνολο: {total}."
        return f"Removed: {names}. New total: {total}."

    def _r_view_cart(self, r, lang, _reply):
        cart = r["cart"]
        if not cart.items:
            return "Το καλάθι σας είναι άδειο." if lang == "el" else "Your cart is empty."
        head = "Η παραγγελία σας:" if lang == "el" else "Your order:"
        tot = "Σύνολο" if lang == "el" else "Total"
        return f"{head}\n{self._cart_lines(cart, lang)}\n{tot}: {money(cart.total(), lang)}"

    def _r_confirm_order(self, r, lang, _reply):
        if r["status"] == "empty":
            return ("Το καλάθι σας είναι άδειο, δεν υπάρχει κάτι να επιβεβαιώσω. Τι θα θέλατε;" if lang == "el"
                    else "Your cart is empty, so there is nothing to confirm yet. What would you like?")
        if lang == "el":
            return (f"Η παραγγελία #{r['order_id']} επιβεβαιώθηκε! Σύνολο: {money(r['total'], lang)}. "
                    "Ευχαριστούμε πολύ!")
        return f"Order #{r['order_id']} confirmed! Total: {money(r['total'], lang)}. Thank you!"

    def _r_cancel_order(self, r, lang, _reply):
        return ("Η παραγγελία ακυρώθηκε και το καλάθι άδειασε." if lang == "el"
                else "Your order has been cancelled and the cart is now empty.")

    # ---------------- reservation ----------------
    def _r_make_reservation(self, r, lang, _reply):
        st = r["status"]
        if st == "ok":
            dt = r["datetime"]
            day = GREEK_DAYS[dt.weekday()] if lang == "el" else EN_DAYS[dt.weekday()]
            when = f"{day} {dt:%d/%m} {'στις' if lang == 'el' else 'at'} {dt:%H:%M}"
            ics = r.get("ics_file", "")
            if lang == "el":
                return (f"Έγινε! Η κράτηση #{r['reservation_id']} για {r['party_size']} άτομα, {when}, "
                        f"καταχωρήθηκε. Το αρχείο ημερολογίου (.ics) είναι: {ics}")
            return (f"Done! Reservation #{r['reservation_id']} for {r['party_size']} people on {when} "
                    f"has been saved. Your calendar file (.ics): {ics}")
        if st == "incomplete":
            missing = r["missing"]
            if lang == "el":
                asks = {"party_size": "για πόσα άτομα", "date": "ποια μέρα", "time": "τι ώρα"}
                return "Ευχαρίστως! Πείτε μου " + _join([asks[m] for m in missing], lang) + ";"
            asks = {"party_size": "for how many people", "date": "which day", "time": "what time"}
            return "Sure! Please tell me " + _join([asks[m] for m in missing], lang) + "."
        if st == "past":
            return ("Η ημερομηνία αυτή έχει περάσει. Ποια άλλη μέρα σας βολεύει;" if lang == "el"
                    else "That date is in the past. Which other day suits you?")
        if st == "closed":
            return (f"Δεχόμαστε κρατήσεις από {config.OPENING_HOUR}:00 έως {config.LAST_SEATING_HOUR}:00. Ποια ώρα σας βολεύει;"
                    if lang == "el" else
                    f"We accept reservations from {config.OPENING_HOUR}:00 to {config.LAST_SEATING_HOUR}:00. What time suits you?")
        if st == "party_too_large":
            return (f"Online κρατήσεις γίνονται έως {config.MAX_PARTY_SIZE} άτομα. Για μεγαλύτερες ομάδες καλέστε μας."
                    if lang == "el" else
                    f"Online reservations are possible for up to {config.MAX_PARTY_SIZE} people. For larger groups please call us.")
        if st == "unparsed":
            return ("Δεν κατάλαβα την ημερομηνία/ώρα. Μπορείτε να τη γράψετε π.χ. «Παρασκευή στις 20:00»;" if lang == "el"
                    else "I didn't understand the date/time. Could you write it like \"Friday at 20:00\"?")
        return FALLBACK_ERROR[lang]

    # ---------------- none ----------------
    def _r_none(self, r, lang, reply):
        return reply or SMALL_TALK_REPLIES["out_of_scope"][lang]
