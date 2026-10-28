# ============ chatbot/menu_data.py ============
"""
Φόρτωση του menu.json και «έξυπνη» αναζήτηση πιάτων / αλλεργιογόνων / κατηγοριών.

Χρησιμοποιείται από: tools.py, rule_parser.py, dataset_builder.py, evaluate.py.
Δεν εισάγει torch/transformers, ώστε να φορτώνει γρήγορα.
"""
from __future__ import annotations

import difflib
import json
import re
import unicodedata
from pathlib import Path

import config

# ---------------------------------------------------------------------------
# Κανονικοποίηση κειμένου (πεζά, χωρίς τόνους, ς→σ)
# ---------------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[^\w\s:/]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    """Αφαιρεί τόνους/διαλυτικά: 'γλουτένη' → 'γλουτενη'."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def normalize(text: str) -> str:
    """Κανονικοποίηση για συγκρίσεις: πεζά, χωρίς τόνους, χωρίς σημεία στίξης."""
    text = strip_accents(text.lower()).replace("ς", "σ")
    text = text.replace("-", " ").replace("_", " ")
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def has_greek(text: str) -> bool:
    return any("\u0370" <= ch <= "\u03ff" or "\u1f00" <= ch <= "\u1fff" for ch in text)


# ---------------------------------------------------------------------------
# Λεξικά συνωνύμων
# ---------------------------------------------------------------------------
NUMBER_WORDS = {
    # αγγλικά
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "couple": 2, "pair": 2,
    # ελληνικά (κανονικοποιημένα)
    "ενα": 1, "εναν": 1, "μια": 1, "μιαν": 1, "ενασ": 1, "δυο": 2, "τρια": 3, "τρεισ": 3,
    "τεσσερα": 4, "τεσσερισ": 4, "πεντε": 5, "εξι": 6, "επτα": 7, "εφτα": 7,
    "οκτω": 8, "οχτω": 8, "εννια": 9, "εννεα": 9, "δεκα": 10, "εντεκα": 11, "δωδεκα": 12,
    # greeklish
    "ena": 1, "mia": 1, "dyo": 2, "dio": 2, "duo": 2, "tria": 3, "treis": 3, "tris": 3,
    "tessera": 4, "tesseris": 4, "pente": 5, "eksi": 6, "exi": 6,
}

ALLERGEN_SYNONYMS = {
    "gluten": ["gluten", "wheat", "γλουτενη", "σιταρι", "glouteni", "glutenh"],
    "milk": ["milk", "dairy", "lactose", "cheese", "γαλα", "γαλακτοκομικα", "λακτοζη", "gala", "laktozi", "galaktokomika"],
    "eggs": ["egg", "eggs", "αυγο", "αυγα", "αβγα", "avga", "avgo"],
    "nuts": ["nuts", "nut", "tree nuts", "walnut", "walnuts", "pistachio", "ξηρουσ καρπουσ", "ξηροι καρποι", "καρυδια", "ksirous karpous", "karydia"],
    "peanuts": ["peanut", "peanuts", "φιστικια", "φυστικια", "αραπικα φιστικια", "fistikia"],
    "fish": ["fish", "anchovy", "ψαρι", "ψαρια", "psari"],
    "crustaceans": ["shellfish", "crustacean", "crustaceans", "shrimp", "shrimps", "prawn", "οστρακοειδη", "γαριδεσ", "γαριδα", "garides"],
    "molluscs": ["mollusc", "molluscs", "mollusk", "squid", "μαλακια", "καλαμαρι", "malakia"],
    "sesame": ["sesame", "tahini", "σουσαμι", "ταχινι", "sousami"],
    "soybeans": ["soy", "soya", "soybean", "soybeans", "tofu", "σογια", "sogia", "soya"],
    "celery": ["celery", "σελινο", "selino"],
    "mustard": ["mustard", "μουσταρδα", "moustarda"],
    "sulphites": ["sulphite", "sulphites", "sulfite", "sulfites", "θειωδη", "θειωδεσ", "theiodi"],
    "lupin": ["lupin", "lupine", "λουπινο", "loupino"],
}

CATEGORY_SYNONYMS = {
    "starter": ["starter", "starters", "appetizer", "appetizers", "appetisers", "ορεκτικα", "ορεκτικο", "orektika", "mezedes", "μεζεδεσ"],
    "salad": ["salad", "salads", "σαλατα", "σαλατεσ", "salates", "salata"],
    "main": ["main", "mains", "main course", "main courses", "main dishes", "κυριωσ πιατα", "κυριωσ", "kyrios piata"],
    "pizza": ["pizza", "pizzas", "πιτσα", "πιτσεσ", "pitses", "pitsa"],
    "dessert": ["dessert", "desserts", "sweets", "sweet", "γλυκα", "γλυκο", "επιδορπια", "επιδορπιο", "glyka", "gluka"],
    "drink": ["drink", "drinks", "beverages", "ποτα", "ποτο", "αναψυκτικα", "pota", "anapsyktika"],
}


class Menu:
    """Περιτύλιγμα του menu.json με μεθόδους αναζήτησης."""

    def __init__(self, path: Path | str = config.MENU_PATH):
        with open(path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.dishes: list[dict] = self.data["dishes"]
        self.by_id = {d["id"]: d for d in self.dishes}
        self.category_names = self.data["category_names"]
        self.allergen_names = self.data["allergen_names"]

        # alias (κανονικοποιημένο) → id πιάτου
        self.alias_index: dict[str, str] = {}
        for d in self.dishes:
            for alias in [d["name_en"], d["name_el"], *d.get("aliases", [])]:
                self.alias_index.setdefault(normalize(alias), d["id"])
        # μεγαλύτερα aliases πρώτα, ώστε το "pizza margherita" να κερδίζει το "margherita"
        self._aliases_sorted = sorted(self.alias_index, key=len, reverse=True)
        self._single_word_aliases = [a for a in self._aliases_sorted if " " not in a and len(a) >= 4]

        self._allergen_index = {normalize(s): key for key, syns in ALLERGEN_SYNONYMS.items() for s in syns}
        for key, names in self.allergen_names.items():
            for name in [names["en"], names["el"]]:
                for part in name.split("/"):
                    self._allergen_index.setdefault(normalize(part), key)
        self._category_index = {normalize(s): key for key, syns in CATEGORY_SYNONYMS.items() for s in syns}

    # ------------------------------------------------------------------
    # Ονόματα
    # ------------------------------------------------------------------
    def dish_name(self, dish: dict, lang: str = "en") -> str:
        return dish["name_el"] if lang == "el" else dish["name_en"]

    def allergen_name(self, key: str, lang: str = "en", accusative: bool = False) -> str:
        names = self.allergen_names.get(key)
        if not names:
            return key
        if accusative and lang == "el":
            return names.get("el_acc", names["el"])
        return names[lang]

    def category_name(self, key: str, lang: str = "en") -> str:
        names = self.category_names.get(key)
        return names[lang] if names else key

    # ------------------------------------------------------------------
    # Αναζήτηση
    # ------------------------------------------------------------------
    def find_dish(self, name: str | None) -> dict | None:
        """Βρίσκει πιάτο από όνομα (ακριβές → περιέχεται → fuzzy)."""
        if not name:
            return None
        if name in self.by_id:
            return self.by_id[name]
        norm = normalize(name)
        if not norm:
            return None
        if norm in self.alias_index:
            return self.by_id[self.alias_index[norm]]
        for alias in self._aliases_sorted:
            if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", norm):
                return self.by_id[self.alias_index[alias]]
        match = difflib.get_close_matches(norm, self._aliases_sorted, n=1, cutoff=0.75)
        if match:
            return self.by_id[self.alias_index[match[0]]]
        return None

    def scan_dishes(self, text: str) -> list[tuple[dict, int | None]]:
        """Βρίσκει όλα τα πιάτα μέσα σε ένα κείμενο μαζί με ποσότητα (None αν δεν δόθηκε).
        π.χ. 'θέλω 2 μαργαρίτες και μια κόκα κόλα' → [(margherita, 2), (coca_cola, 1)]"""
        norm = normalize(text)
        taken = [False] * len(norm)
        found: list[tuple[int, dict, int]] = []

        def qty_before(start: int) -> int:
            prev_tokens = norm[:start].split()[-3:]
            for tok in reversed(prev_tokens):
                m = re.fullmatch(r"x?(\d{1,2})x?", tok)
                if m:
                    return max(1, int(m.group(1)))
                if tok in NUMBER_WORDS:
                    return NUMBER_WORDS[tok]
                if tok in ("και", "and", "kai", "with", "με", "me"):
                    break
            return None   # δεν αναφέρθηκε ρητά ποσότητα

        for alias in self._aliases_sorted:
            for m in re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", norm):
                s, e = m.span()
                if any(taken[s:e]):
                    continue
                for i in range(s, e):
                    taken[i] = True
                found.append((s, self.by_id[self.alias_index[alias]], qty_before(s)))

        # fuzzy για λέξεις που δεν ταίριαξαν (π.χ. πληθυντικός 'μουσακάδες')
        pos = 0
        for tok in norm.split():
            s = norm.index(tok, pos)
            e = s + len(tok)
            pos = e
            if any(taken[s:e]) or len(tok) < 5 or tok.isdigit():
                continue
            match = difflib.get_close_matches(tok, self._single_word_aliases, n=1, cutoff=0.82)
            if match:
                for i in range(s, e):
                    taken[i] = True
                found.append((s, self.by_id[self.alias_index[match[0]]], qty_before(s)))

        found.sort(key=lambda x: x[0])
        # ένωση διπλών εμφανίσεων του ίδιου πιάτου
        merged: dict[str, list] = {}
        for _, dish, qty in found:
            if dish["id"] in merged:
                merged[dish["id"]][1] = (merged[dish["id"]][1] or 1) + (qty or 1)
            else:
                merged[dish["id"]] = [dish, qty]
        return [(d, q) for d, q in merged.values()]   # q = None αν δεν δόθηκε ποσότητα

    def find_allergen(self, text: str | None) -> str | None:
        """Επιστρέφει το κλειδί αλλεργιογόνου (π.χ. 'gluten') που αναφέρεται στο κείμενο."""
        if not text:
            return None
        if text in self.allergen_names:
            return text
        norm = normalize(text)
        for syn in sorted(self._allergen_index, key=len, reverse=True):
            if re.search(rf"(?<!\w){re.escape(syn)}", norm):
                return self._allergen_index[syn]
        return None

    def find_category(self, text: str | None) -> str | None:
        if not text:
            return None
        if text in self.category_names:
            return text
        norm = normalize(text)
        for syn in sorted(self._category_index, key=len, reverse=True):
            if re.search(rf"(?<!\w){re.escape(syn)}(?!\w)", norm):
                return self._category_index[syn]
        return None

    def dishes_in_category(self, category: str) -> list[dict]:
        return [d for d in self.dishes if d["category"] == category]

    def strip_dishes(self, text: str) -> str:
        """Αφαιρεί τα ονόματα πιάτων από το κείμενο (για να μη μπερδευτεί π.χ. το
        'shrimp pasta' με το αλλεργιογόνο 'shrimp' όταν ρωτάμε για γλουτένη)."""
        norm = normalize(text)
        for alias in self._aliases_sorted:
            norm = re.sub(rf"(?<!\w){re.escape(alias)}(?!\w)", " ", norm)
        return _SPACE_RE.sub(" ", norm).strip()
