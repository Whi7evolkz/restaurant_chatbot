# ============ chatbot/tools.py ============
"""
Τα «εργαλεία» (tools) που εκτελεί η Python όταν το μοντέλο αποφασίσει μια ενέργεια.

Το fine-tuned LLM επιστρέφει π.χ. {"action": "get_allergens", "arguments": {...}}
και εδώ εκτελείται η αντίστοιχη συνάρτηση ΠΑΝΩ στο menu.json.
Έτσι οι τιμές και τα αλλεργιογόνα προέρχονται πάντα από τα πραγματικά δεδομένα.
"""
from __future__ import annotations

from chatbot.menu_data import Menu
from data_handlers.order_manager import Cart, OrderManager
from data_handlers.reservation_manager import ReservationManager


def get_menu(menu: Menu, category: str | None = None) -> dict:
    cat = menu.find_category(category) if category else None
    cats = [cat] if cat else list(menu.category_names)
    return {"status": "ok", "categories": {c: menu.dishes_in_category(c) for c in cats}}


def get_dish_info(menu: Menu, dish: str | None) -> dict:
    d = menu.find_dish(dish)
    return {"status": "ok", "dish": d} if d else {"status": "not_found", "query": dish}


def get_price(menu: Menu, dish: str | None) -> dict:
    d = menu.find_dish(dish)
    return {"status": "ok", "dish": d, "price": d["price"]} if d else {"status": "not_found", "query": dish}


def get_allergens(menu: Menu, dish: str | None = None, allergen: str | None = None) -> dict:
    """ΥΠΟΧΡΕΩΤΙΚΟ για κάθε ερώτηση αλλεργιογόνων — ποτέ απάντηση «από μνήμης»."""
    allergen_key = menu.find_allergen(allergen) if allergen else None
    if dish:
        d = menu.find_dish(dish)
        if d is None:
            return {"status": "dish_not_found", "query": dish}
        result = {"status": "ok", "dish": d, "allergen": allergen_key,
                  "contains": (allergen_key in d["allergens"]) if allergen_key else None, "alternatives": []}
        if allergen_key and result["contains"]:
            same_cat = [x for x in menu.dishes if x["category"] == d["category"]
                        and allergen_key not in x["allergens"] and x["id"] != d["id"]]
            result["alternatives"] = same_cat[:2]
        return result
    if allergen_key:
        free = [x for x in menu.dishes if allergen_key not in x["allergens"] and x["category"] != "drink"]
        return {"status": "free_of", "allergen": allergen_key, "dishes": free}
    return {"status": "need_more"}


class ToolExecutor:
    """Εκτελεί μια ενέργεια για μια συγκεκριμένη συνεδρία (session)."""

    def __init__(self, menu: Menu, orders: OrderManager, reservations: ReservationManager):
        self.menu = menu
        self.orders = orders
        self.reservations = reservations

    def execute(self, action: str, args: dict, session) -> dict:
        args = args or {}
        if action == "get_menu":
            return get_menu(self.menu, args.get("category"))
        if action == "get_dish_info":
            return get_dish_info(self.menu, args.get("dish"))
        if action == "get_price":
            return get_price(self.menu, args.get("dish"))
        if action == "get_allergens":
            return get_allergens(self.menu, args.get("dish"), args.get("allergen"))
        if action == "add_item":
            return self._add(args.get("items") or [], session.cart)
        if action == "remove_item":
            return self._remove(args.get("items") or [], session.cart)
        if action == "view_cart":
            return {"status": "ok", "cart": session.cart}
        if action == "confirm_order":
            if not session.cart.items:
                return {"status": "empty"}
            order_id, total = self.orders.confirm(session.cart, session.session_id)
            return {"status": "ok", "order_id": order_id, "total": total}
        if action == "cancel_order":
            session.cart.clear()
            return {"status": "ok"}
        if action == "make_reservation":
            return self.reservations.make(args, session.partial_reservation, session.session_id)
        return {"status": "ok"}   # action == "none"

    def _add(self, items: list, cart: Cart) -> dict:
        added, not_found = [], []
        for it in items:
            if not isinstance(it, dict):
                continue
            d = self.menu.find_dish(it.get("name"))
            if d is None:
                not_found.append(it.get("name") or "?")
                continue
            try:
                qty = int(it.get("qty") or 1)
            except (TypeError, ValueError):
                qty = 1
            cart.add(d, qty)
            added.append((d, max(1, qty)))
        return {"status": "ok", "added": added, "not_found": not_found, "total": cart.total()}

    def _remove(self, items: list, cart: Cart) -> dict:
        removed = []
        for it in items:
            if not isinstance(it, dict):
                continue
            d = self.menu.find_dish(it.get("name"))
            qty = it.get("qty")
            if d and cart.remove(d, int(qty) if isinstance(qty, (int, float)) else None):
                removed.append(d)
        return {"status": "ok", "removed": removed, "total": cart.total()}
