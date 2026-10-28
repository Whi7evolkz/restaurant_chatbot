# ============ data_handlers/order_manager.py ============
"""
Καλάθι αγορών (cart) και αποθήκευση επιβεβαιωμένων παραγγελιών στο orders.xlsx.
Κωδικός παραγγελίας: ORD-YYYYMMDD-NNN (αύξων αριθμός ανά ημέρα).
"""
from __future__ import annotations

from datetime import datetime

import config
from data_handlers.excel_logger import ExcelSheetWriter


class Cart:
    def __init__(self):
        self.items: list[dict] = []   # {id, name_en, name_el, qty, unit_price}

    def add(self, dish: dict, qty: int) -> None:
        qty = max(1, min(int(qty or 1), 50))
        for it in self.items:
            if it["id"] == dish["id"]:
                it["qty"] += qty
                return
        self.items.append({"id": dish["id"], "name_en": dish["name_en"], "name_el": dish["name_el"],
                           "qty": qty, "unit_price": float(dish["price"])})

    def remove(self, dish: dict, qty: int | None = None) -> bool:
        for it in self.items:
            if it["id"] == dish["id"]:
                if qty is None or qty >= it["qty"]:
                    self.items.remove(it)
                else:
                    it["qty"] -= int(qty)
                return True
        return False

    def total(self) -> float:
        return round(sum(it["qty"] * it["unit_price"] for it in self.items), 2)

    def clear(self) -> None:
        self.items.clear()

    def for_prompt(self) -> list[dict]:
        """Μορφή που βλέπει το μοντέλο: [{'name': ..., 'qty': ...}]"""
        return [{"name": it["name_en"], "qty": it["qty"]} for it in self.items]


class OrderManager:
    HEADERS = ["Order ID", "Timestamp", "Session ID", "Item", "Qty", "Unit Price (€)", "Line Total (€)",
               "Order Total (€)"]
    WIDTHS = [20, 20, 14, 30, 8, 14, 14, 16]

    def __init__(self, path=config.ORDERS_XLSX):
        self.writer = ExcelSheetWriter(path, "Orders", self.HEADERS, self.WIDTHS)

    def _next_order_id(self) -> str:
        today = datetime.now().strftime("%Y%m%d")
        prefix = f"ORD-{today}-"
        ids = {r["Order ID"] for r in self.writer.read_rows() if str(r.get("Order ID", "")).startswith(prefix)}
        return f"{prefix}{len(ids) + 1:03d}"

    def confirm(self, cart: Cart, session_id: str) -> tuple[str, float]:
        order_id = self._next_order_id()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total = cart.total()
        rows = [[order_id, ts, session_id, it["name_en"], it["qty"], it["unit_price"],
                 round(it["qty"] * it["unit_price"], 2), total] for it in cart.items]
        self.writer.append_rows(rows)
        cart.clear()
        return order_id, total
