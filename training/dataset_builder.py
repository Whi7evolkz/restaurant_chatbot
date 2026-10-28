# ============ training/dataset_builder.py ============
"""
Δημιουργία ΣΥΝΘΕΤΙΚΟΥ dataset εκπαίδευσης από το menu.json.

Γιατί συνθετικό: δεν έχουμε πραγματικές συνομιλίες πελατών. Παράγουμε χιλιάδες
παραδείγματα από πρότυπα (templates) με παραλλαγές, και επειδή οι ετικέτες
(όνομα πιάτου, ποσότητα, αλλεργιογόνο) βγαίνουν από το ίδιο το μενού,
είναι πάντα σωστές.

Κάθε παράδειγμα = system prompt + (ιστορικό, καλάθι, κατάσταση, μήνυμα) → JSON στόχος.

Εκτέλεση:
    python -m training.dataset_builder               # 2000 παραδείγματα (config)
    python -m training.dataset_builder --n 500       # λιγότερα για CPU
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from chatbot.menu_data import Menu, has_greek, strip_accents  # noqa: E402
from chatbot.prompt_format import build_messages, build_user_content  # noqa: E402
from chatbot.response_builder import SMALL_TALK_REPLIES, ResponseBuilder  # noqa: E402

MENU = Menu()
RB = ResponseBuilder(MENU)
rng = random.Random(config.SEED)

# «γλώσσες» επιφάνειας: en, el (ελληνικά), gl (greeklish → απαντάμε στα ελληνικά)
LANG_WEIGHTS = [("en", 0.42), ("el", 0.43), ("gl", 0.15)]

INTENT_WEIGHTS = {
    "ask_menu": 8, "ask_dish_info": 9, "ask_price": 11, "ask_allergen": 15,
    "order_food": 15, "remove_item": 5, "view_cart": 5, "confirm_order": 7,
    "cancel_order": 4, "make_reservation": 13, "small_talk": 5, "out_of_scope": 3,
}


def choice(seq):
    return rng.choice(seq)


def weighted(pairs):
    items, weights = zip(*pairs)
    return rng.choices(items, weights=weights, k=1)[0]


def code(lang: str) -> str:
    """Γλώσσα απάντησης: greeklish → el."""
    return "en" if lang == "en" else "el"


# ---------------------------------------------------------------------------
# Επιφάνειες (πώς γράφει ο πελάτης ένα πιάτο / ποσότητα / αλλεργιογόνο)
# ---------------------------------------------------------------------------
def dish_surface(d: dict, lang: str) -> str:
    greek = [a for a in d["aliases"] if has_greek(a)]
    ascii_ = [a for a in d["aliases"] if not has_greek(a)]
    if lang == "en":
        return choice([d["name_en"], d["name_en"].lower(), *ascii_])
    if lang == "el":
        return choice([d["name_el"], d["name_el"].lower(), *greek])
    return choice(ascii_)


def article(d: dict, capital: bool = False) -> str:
    a = d.get("article_el", "")
    return a.capitalize() if capital else a


QTY_EN = {1: ["a", "one", "1"], 2: ["two", "2"], 3: ["three", "3"], 4: ["four", "4"]}
QTY_EL = {1: ["ένα", "μία", "μια", "1"], 2: ["δύο", "2"], 3: ["τρία", "τρεις", "3"], 4: ["τέσσερα", "τέσσερις", "4"]}
QTY_GL = {1: ["ena", "mia", "1"], 2: ["dyo", "2"], 3: ["tria", "3"], 4: ["tessera", "4"]}
PEOPLE_EN = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
PEOPLE_EL = {1: "ένα", 2: "δύο", 3: "τρία", 4: "τέσσερα", 5: "πέντε", 6: "έξι", 7: "επτά", 8: "οκτώ", 9: "εννιά", 10: "δέκα"}


def qty_surface(q: int, lang: str) -> str:
    table = {"en": QTY_EN, "el": QTY_EL, "gl": QTY_GL}[lang]
    return choice(table[q])


ALLERGEN_SURF = {
    "en": {"gluten": ["gluten", "wheat"], "milk": ["milk", "dairy", "lactose"], "eggs": ["eggs", "egg"],
           "nuts": ["nuts", "walnuts"], "peanuts": ["peanuts"], "fish": ["fish"],
           "crustaceans": ["shellfish", "shrimp"], "molluscs": ["molluscs", "squid"], "sesame": ["sesame"],
           "soybeans": ["soy"], "celery": ["celery"], "mustard": ["mustard"], "sulphites": ["sulphites", "sulfites"]},
    "el": {"gluten": ["γλουτένη"], "milk": ["γάλα", "γαλακτοκομικά", "λακτόζη"], "eggs": ["αυγά", "αυγό"],
           "nuts": ["ξηρούς καρπούς", "καρύδια"], "peanuts": ["φιστίκια"], "fish": ["ψάρι"],
           "crustaceans": ["οστρακοειδή", "γαρίδες"], "molluscs": ["μαλάκια"], "sesame": ["σουσάμι"],
           "soybeans": ["σόγια"], "celery": ["σέλινο"], "mustard": ["μουστάρδα"], "sulphites": ["θειώδη"]},
    "gl": {"gluten": ["glouteni"], "milk": ["gala", "galaktokomika", "laktozi"], "eggs": ["avga"],
           "nuts": ["ksirous karpous", "karydia"], "peanuts": ["fistikia"], "fish": ["psari"],
           "crustaceans": ["garides"], "molluscs": ["malakia"], "sesame": ["sousami"], "soybeans": ["sogia"],
           "celery": ["selino"], "mustard": ["moustarda"], "sulphites": ["theiodi"]},
}
COMMON_ALLERGENS = ["gluten", "milk", "eggs", "nuts", "fish", "crustaceans", "sesame", "soybeans", "peanuts", "celery"]

CATEGORY_SURF = {
    "en": {"starter": ["starters", "appetizers"], "salad": ["salads"], "main": ["main courses", "mains", "main dishes"],
           "pizza": ["pizzas"], "dessert": ["desserts", "sweets"], "drink": ["drinks", "beverages"]},
    "el": {"starter": ["ορεκτικά", "μεζέδες"], "salad": ["σαλάτες"], "main": ["κυρίως πιάτα"],
           "pizza": ["πίτσες"], "dessert": ["γλυκά", "επιδόρπια"], "drink": ["ποτά", "αναψυκτικά"]},
    "gl": {"starter": ["orektika", "mezedes"], "salad": ["salates"], "main": ["kyrios piata"],
           "pizza": ["pitses"], "dessert": ["glyka"], "drink": ["pota", "anapsyktika"]},
}


# ---------------------------------------------------------------------------
# Augmentation: πεζά, χωρίς τόνους, typos, ευγενικές φράσεις
# ---------------------------------------------------------------------------
PREFIX = {"en": ["Hi, ", "Hello, ", "Excuse me, ", "Hey, "], "el": ["Γεια σας, ", "Καλησπέρα, ", "Συγγνώμη, "],
          "gl": ["Geia sas, ", "Kalispera, "]}
SUFFIX = {"en": [" please", " thanks"], "el": [" παρακαλώ", " ευχαριστώ"], "gl": [" parakalo"]}


def augment(text: str, lang: str, allow_affixes: bool = True) -> str:
    if allow_affixes and rng.random() < 0.12:
        text = choice(PREFIX[lang]) + text[0].lower() + text[1:]
    if allow_affixes and rng.random() < 0.08:
        text = text.rstrip("?.!;") + choice(SUFFIX[lang])
    if rng.random() < 0.25:
        text = text.lower()
    if lang == "el" and rng.random() < 0.3:
        text = strip_accents(text)
    if rng.random() < 0.3:
        text = text.rstrip("?.!;")
    if rng.random() < 0.08:  # μικρό τυπογραφικό λάθος
        words = text.split()
        idx = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
        if idx:
            i = choice(idx)
            w = list(words[i])
            j = rng.randrange(len(w) - 1)
            w[j], w[j + 1] = w[j + 1], w[j]
            words[i] = "".join(w)
            text = " ".join(words)
    return text or "?"


# ---------------------------------------------------------------------------
# Περιβάλλον: ιστορικό & καλάθι
# ---------------------------------------------------------------------------
def random_cart(min_items: int = 0) -> list[dict]:
    k = max(min_items, choice([0, 0, 0, 1, 2, 3]))
    dishes = rng.sample(MENU.dishes, k)
    return [{"name": d["name_en"], "qty": choice([1, 1, 2])} for d in dishes]


def history_pair(lang: str) -> tuple[str, str]:
    c = code(lang)
    kind = choice(["greet", "price", "info", "allergen", "menu"])
    d = choice(MENU.dishes)
    if kind == "greet":
        u = {"en": "Hello", "el": "Γεια σας", "gl": "geia sas"}[lang]
        return u, SMALL_TALK_REPLIES["greeting"][c]
    if kind == "price":
        u = {"en": f"How much is the {d['name_en']}?", "el": f"Πόσο κάνει {article(d)} {d['name_el']};",
             "gl": f"poso kanei {dish_surface(d, 'gl')}?"}[lang]
        return u, RB.build("get_price", {"status": "ok", "dish": d}, c)
    if kind == "info":
        u = {"en": f"What's in the {d['name_en']}?", "el": f"Τι περιέχει {article(d)} {d['name_el']};",
             "gl": f"ti periexei {dish_surface(d, 'gl')}?"}[lang]
        return u, RB.build("get_dish_info", {"status": "ok", "dish": d}, c)
    if kind == "allergen":
        u = {"en": f"What allergens does the {d['name_en']} have?", "el": f"Τι αλλεργιογόνα έχει {article(d)} {d['name_el']};",
             "gl": f"ti allergiogona exei {dish_surface(d, 'gl')}?"}[lang]
        return u, RB.build("get_allergens", {"status": "ok", "dish": d, "allergen": None, "contains": None}, c)
    u = {"en": "Can I see the menu?", "el": "Μπορώ να δω το μενού;", "gl": "mporo na do to menu?"}[lang]
    return u, ("Here is our menu: Starters: Tzatziki – €4.50, Grilled Halloumi – €6.00 …" if c == "en"
               else "Το μενού μας: Ορεκτικά: Τζατζίκι – 4,50 €, Χαλλούμι Σχάρας – 6,00 € …")


def random_history(lang: str) -> list[tuple[str, str]]:
    n_pairs = weighted([(0, 0.45), (1, 0.35), (2, 0.15), (3, 0.05)])
    hist: list[tuple[str, str]] = []
    for _ in range(n_pairs):
        u, a = history_pair(lang)
        hist += [("user", u), ("assistant", a)]
    return hist


# ---------------------------------------------------------------------------
# Γεννήτριες ανά intent. Επιστρέφουν dict με message, target και context.
# ---------------------------------------------------------------------------
def T(intent, lang, action, arguments, reply=None):
    return {"intent": intent, "language": code(lang), "action": action, "arguments": arguments, "reply": reply}


def gen_ask_menu(lang):
    cat = None if rng.random() < 0.5 else choice(list(CATEGORY_SURF["en"]))
    if cat is None:
        msg = choice({
            "en": ["What's on the menu?", "Can I see the menu?", "Show me the menu", "What do you have to eat?",
                   "What food do you serve?", "Could you show me the menu please?", "Menu please", "What can I order?",
                   "What's available today?", "What do you recommend from the menu?"],
            "el": ["Τι έχει το μενού;", "Μπορώ να δω το μενού;", "Δείξε μου τον κατάλογο", "Τι φαγητά έχετε;",
                   "Τι μπορώ να παραγγείλω;", "Έχετε κατάλογο;", "Το μενού παρακαλώ", "Τι έχετε σήμερα;",
                   "Θέλω να δω τι φαγητά έχετε"],
            "gl": ["ti exei to menu?", "mporo na do to menu?", "deikse mou ton katalogo", "ti fagita exete?"],
        }[lang])
    else:
        c = choice(CATEGORY_SURF[lang][cat])
        msg = choice({
            "en": ["What {c} do you have?", "Do you have any {c}?", "Show me the {c}", "Which {c} are on the menu?",
                   "What kind of {c} do you serve?", "List the {c} please"],
            "el": ["Τι {c} έχετε;", "Έχετε {c};", "Δείξε μου {c}", "Τι {c} υπάρχουν;", "Θέλω να δω {c}"],
            "gl": ["ti {c} exete?", "exete {c}?", "deikse mou {c}"],
        }[lang]).format(c=c)
    return {"message": msg, "target": T("ask_menu", lang, "get_menu", {"category": cat})}


def gen_ask_dish_info(lang):
    d = choice(MENU.dishes)
    s = dish_surface(d, lang)
    msg = choice({
        "en": ["What's in the {d}?", "What are the ingredients of the {d}?", "Tell me about the {d}",
               "What does the {d} come with?", "Can you describe the {d}?", "What is {d}?", "{d} ingredients?",
               "How is the {d} made?"],
        "el": ["Τι περιέχει {a} {d};", "Τι υλικά έχει {a} {d};", "Πες μου για {d}", "Τι είναι {a} {d};",
               "Τι έχει μέσα {a} {d};", "Υλικά για {d};", "Περιγράψτε μου {d}"],
        "gl": ["ti periexei {d}?", "ti ylika exei {d}?", "pes mou gia {d}", "ti einai {d}?"],
    }[lang]).format(d=s, a=article(d))
    return {"message": msg, "target": T("ask_dish_info", lang, "get_dish_info", {"dish": d["name_en"]})}


def gen_ask_price(lang):
    d = choice(MENU.dishes)
    if rng.random() < 0.2:  # αναφορά στο προηγούμενο πιάτο: «Πόσο κάνει;»
        u = {"en": f"Tell me about the {d['name_en']}", "el": f"Τι περιέχει {article(d)} {d['name_el']};",
             "gl": f"ti periexei {dish_surface(d, 'gl')}?"}[lang]
        a = RB.build("get_dish_info", {"status": "ok", "dish": d}, code(lang))
        msg = choice({"en": ["How much is it?", "And how much does it cost?", "What's the price?", "How much?"],
                      "el": ["Πόσο κάνει;", "Και πόσο κοστίζει;", "Τι τιμή έχει;", "Πόσο έχει;"],
                      "gl": ["poso kanei?", "poso kostizei?"]}[lang])
        return {"message": msg, "history": [("user", u), ("assistant", a)],
                "target": T("ask_price", lang, "get_price", {"dish": d["name_en"]}), "no_affix": True}
    s = dish_surface(d, lang)
    msg = choice({
        "en": ["How much is the {d}?", "How much does the {d} cost?", "What's the price of the {d}?", "Price of {d}?",
               "How much for a {d}?", "What does the {d} cost?", "Is the {d} expensive?", "how much {d}"],
        "el": ["Πόσο κάνει {a} {d};", "Πόσο κοστίζει {a} {d};", "Τι τιμή έχει {a} {d};", "Πόσο έχει {a} {d};",
               "Τιμή για {d};", "Πόσο πάει {a} {d};"],
        "gl": ["poso kanei {d}?", "poso kostizei {d}?", "ti timi exei {d}?", "poso exei {d}?"],
    }[lang]).format(d=s, a=article(d))
    return {"message": msg, "target": T("ask_price", lang, "get_price", {"dish": d["name_en"]})}


def gen_ask_allergen(lang):
    mode = weighted([("both", 0.55), ("dish", 0.25), ("allergen", 0.20)])
    d = choice(MENU.dishes)
    if d["allergens"] and rng.random() < 0.5:
        al = choice([a for a in d["allergens"] if a in ALLERGEN_SURF[lang]] or COMMON_ALLERGENS)
    else:
        al = choice(COMMON_ALLERGENS)
    al_s = choice(ALLERGEN_SURF[lang][al])
    s = dish_surface(d, lang)
    if mode == "both":
        msg = choice({
            "en": ["Does the {d} contain {al}?", "Is there {al} in the {d}?", "Does the {d} have {al}?",
                   "Is the {d} {al}-free?", "I'm allergic to {al}. Can I eat the {d}?",
                   "Is the {d} safe for someone allergic to {al}?", "Any {al} in the {d}?"],
            "el": ["Έχει {al} {a} {d};", "Περιέχει {al} {a} {d};", "{A} {d} έχει {al};",
                   "Είμαι αλλεργικός σε {al}, μπορώ να φάω {d};", "Έχω αλλεργία σε {al}. Είναι εντάξει {a} {d};",
                   "{A} {d} είναι χωρίς {al};"],
            "gl": ["exei {al} {d}?", "periexei {al} {d}?", "eimai allergikos se {al}, mporo na fao {d}?", "{d} exei {al}?"],
        }[lang]).format(d=s, al=al_s, a=article(d), A=article(d, True))
        args = {"dish": d["name_en"], "allergen": al}
    elif mode == "dish":
        msg = choice({
            "en": ["What allergens does the {d} have?", "Which allergens are in the {d}?",
                   "Does the {d} contain any allergens?", "Allergens in {d}?", "Is the {d} safe for allergies?"],
            "el": ["Τι αλλεργιογόνα έχει {a} {d};", "Ποια αλλεργιογόνα περιέχει {a} {d};",
                   "Έχει αλλεργιογόνα {a} {d};", "Αλλεργιογόνα για {d};"],
            "gl": ["ti allergiogona exei {d}?", "allergiogona {d}?"],
        }[lang]).format(d=s, a=article(d))
        args = {"dish": d["name_en"], "allergen": None}
    else:
        msg = choice({
            "en": ["Which dishes are {al}-free?", "Do you have anything without {al}?",
                   "What can I eat if I'm allergic to {al}?", "Which dishes don't contain {al}?",
                   "I can't eat {al}, what do you recommend?"],
            "el": ["Ποια πιάτα δεν έχουν {al};", "Έχετε κάτι χωρίς {al};", "Τι μπορώ να φάω αν έχω αλλεργία σε {al};",
                   "Ποια πιάτα είναι χωρίς {al};", "Δεν τρώω {al}, τι μου προτείνετε;"],
            "gl": ["ti piata den exoun {al}?", "exete kati xoris {al}?"],
        }[lang]).format(al=al_s)
        args = {"dish": None, "allergen": al}
    return {"message": msg, "target": T("ask_allergen", lang, "get_allergens", args)}


def gen_order_food(lang):
    n = weighted([(1, 0.5), (2, 0.35), (3, 0.15)])
    dishes = rng.sample(MENU.dishes, n)
    qtys = [weighted([(1, 0.6), (2, 0.25), (3, 0.1), (4, 0.05)]) for _ in dishes]
    parts = [f"{qty_surface(q, lang)} {dish_surface(d, lang)}" for d, q in zip(dishes, qtys)]
    conj = {"en": " and ", "el": " και ", "gl": " kai "}[lang]
    items = parts[0] if n == 1 else ", ".join(parts[:-1]) + conj + parts[-1]
    msg = choice({
        "en": ["I want {i}", "I'd like {i}", "Can I get {i}?", "Can I have {i}?", "I'll have {i}", "Add {i} to my order",
               "{i} please", "Please bring {i}", "We would like {i}", "Give me {i}", "I'd like to order {i}"],
        "el": ["Θέλω {i}", "Θα ήθελα {i}", "Βάλε μου {i}", "Πρόσθεσε {i}", "Μπορώ να έχω {i};", "Να πάρω {i}",
               "Φέρτε μας {i}", "Θα πάρω {i}", "{i} παρακαλώ", "Θέλω να παραγγείλω {i}"],
        "gl": ["thelo {i}", "tha ithela {i}", "vale mou {i}", "na paro {i}", "{i} parakalo"],
    }[lang]).format(i=items)
    msg = msg[0].upper() + msg[1:]
    args = {"items": [{"name": d["name_en"], "qty": q} for d, q in zip(dishes, qtys)]}
    return {"message": msg, "target": T("order_food", lang, "add_item", args)}


def gen_remove_item(lang):
    d = choice(MENU.dishes)
    cart = [x for x in random_cart() if x["name"] != d["name_en"]] + [{"name": d["name_en"], "qty": choice([1, 2])}]
    rng.shuffle(cart)
    s = dish_surface(d, lang)
    if rng.random() < 0.7:
        q = None
        tpl = {"en": ["Remove the {d}", "Take the {d} off my order", "I don't want the {d} anymore", "Cancel the {d}",
                      "Delete the {d} from my cart", "No {d}, remove it", "Actually drop the {d}", "Scratch the {d}",
                      "Forget the {d}", "Skip the {d}"],
               "el": ["Αφαίρεσε {d}", "Βγάλε {a} {d}", "Δεν θέλω πια {d}", "Σβήσε {d} από την παραγγελία", "Ακύρωσε {d}",
                      "Τελικά δεν θέλω {d}", "Τελικά βγάλε {d}", "Χωρίς {d} τελικά"],
               "gl": ["vgale {d}", "afairese {d}", "den thelo pia {d}", "telika den thelo {d}"]}[lang]
    else:
        q = 1
        tpl = {"en": ["Remove one {d}", "Take off one {d}", "One less {d} please"],
               "el": ["Βγάλε ένα {d}", "Αφαίρεσε ένα {d}", "Ένα {d} λιγότερο"],
               "gl": ["vgale ena {d}", "afairese ena {d}"]}[lang]
    msg = choice(tpl).format(d=s, a=article(d))
    return {"message": msg, "cart": cart,
            "target": T("remove_item", lang, "remove_item", {"items": [{"name": d["name_en"], "qty": q}]})}


def gen_view_cart(lang):
    msg = choice({
        "en": ["What's in my cart?", "Show me my order", "What have I ordered so far?", "What's my total?",
               "How much is my order in total?", "Can you repeat my order?", "Order summary", "What's the bill so far?"],
        "el": ["Τι έχω στο καλάθι;", "Δείξε μου την παραγγελία μου", "Τι έχω παραγγείλει μέχρι τώρα;",
               "Πόσο είναι το σύνολο;", "Πόσο βγαίνει η παραγγελία;", "Επανάλαβε την παραγγελία μου",
               "Πόσα πληρώνω συνολικά;"],
        "gl": ["ti exo sto kalathi?", "deikse mou tin paraggelia", "poso einai to synolo?"],
    }[lang])
    return {"message": msg, "target": T("view_cart", lang, "view_cart", {})}


def _order_history(lang, cart):
    d = MENU.find_dish(cart[-1]["name"])
    q = cart[-1]["qty"]
    u = {"en": f"I'd like {q} {d['name_en']}", "el": f"Θέλω {q} {d['name_el']}", "gl": f"thelo {q} {dish_surface(d, 'gl')}"}[lang]
    total = sum(MENU.find_dish(x["name"])["price"] * x["qty"] for x in cart)
    a = RB.build("add_item", {"added": [(d, q)], "not_found": [], "total": total}, code(lang))
    return [("user", u), ("assistant", a)]


def gen_confirm_order(lang):
    cart = random_cart(min_items=1)
    if rng.random() < 0.8:
        pending = "awaiting_order_confirmation"
        msg = choice({
            "en": ["Yes", "Yes please", "Yes, confirm", "Confirm", "Confirm the order", "That's all, confirm", "Sure",
                   "Ok", "OK go ahead", "Yep", "Yes that's everything", "Place the order", "Go ahead and confirm",
                   "That's it", "Send it", "Perfect, that's all", "Sounds good", "All good", "Yes, place it"],
            "el": ["Ναι", "Ναι παρακαλώ", "Ναι, επιβεβαίωσε", "Επιβεβαίωση", "Επιβεβαίωσε την παραγγελία", "Εντάξει",
                   "Οκ", "Ναι, αυτά", "Αυτά, επιβεβαίωσε", "Ολοκλήρωσε την παραγγελία", "Σωστά, προχώρα",
                   "Όλα σωστά", "Τέλεια", "Αυτά είναι", "Στείλ' την", "Μια χαρά, προχώρα"],
            "gl": ["nai", "nai parakalo", "ok", "epivevaiose", "nai afta", "teleia", "ola sosta"],
        }[lang])
        history = _order_history(lang, cart)
    else:
        pending = None
        msg = choice({
            "en": ["Confirm my order", "Please place my order", "I'm done, confirm the order", "That's all, please confirm"],
            "el": ["Επιβεβαίωσε την παραγγελία μου", "Ολοκλήρωσε την παραγγελία", "Τελείωσα, στείλε την παραγγελία"],
            "gl": ["oloklirose tin paraggelia", "epivevaiose tin paraggelia"],
        }[lang])
        history = None
    return {"message": msg, "cart": cart, "pending": pending, "history": history, "no_affix": True,
            "target": T("confirm_order", lang, "confirm_order", {})}


def gen_cancel_order(lang):
    cart = random_cart(min_items=1)
    msg = choice({
        "en": ["Cancel the order", "Cancel everything", "Forget it, cancel my order", "Clear my cart", "Empty the cart",
               "I changed my mind, cancel the order", "Cancel"],
        "el": ["Ακύρωσε την παραγγελία", "Ακύρωση", "Άδειασε το καλάθι", "Ακύρωσέ τα όλα",
               "Άλλαξα γνώμη, ακύρωσε την παραγγελία", "Σβήσε όλη την παραγγελία"],
        "gl": ["akyrose tin paraggelia", "akyrosi", "adeiase to kalathi"],
    }[lang])
    pending = "awaiting_order_confirmation" if rng.random() < 0.5 else None
    return {"message": msg, "cart": cart, "pending": pending, "target": T("cancel_order", lang, "cancel_order", {})}


# ----------------------------- reservations -----------------------------
EN_WD = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
EL_WD = [("τη", "Δευτέρα"), ("την", "Τρίτη"), ("την", "Τετάρτη"), ("την", "Πέμπτη"), ("την", "Παρασκευή"),
         ("το", "Σάββατο"), ("την", "Κυριακή")]
GL_WD = ["Deftera", "Triti", "Tetarti", "Pempti", "Paraskevi", "Savvato", "Kyriaki"]
EN_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
             "November", "December"]
EL_MONTHS_GEN = ["Ιανουαρίου", "Φεβρουαρίου", "Μαρτίου", "Απριλίου", "Μαΐου", "Ιουνίου", "Ιουλίου", "Αυγούστου",
                 "Σεπτεμβρίου", "Οκτωβρίου", "Νοεμβρίου", "Δεκεμβρίου"]
NAMES = {"en": ["Maria", "John", "Anna", "Nikos", "Sophia", "George", "Elena", "Chris"],
         "el": ["Μαρία", "Νίκος", "Ελένη", "Γιώργος", "Άννα", "Σοφία", "Ανδρέας", "Χρήστος"],
         "gl": ["Maria", "Nikos", "Eleni", "Giorgos", "Anna"]}


def date_surface(lang) -> tuple[str, str]:
    """(πώς γράφεται στο μήνυμα, τι επιστρέφει το μοντέλο ως date_text)"""
    day, month = rng.randint(1, 28), rng.randint(1, 12)
    k = rng.random()
    if lang == "en":
        if k < 0.25:
            t = choice(["today", "tonight", "tomorrow", "the day after tomorrow"])
            return t, t
        if k < 0.65:
            wd = choice(EN_WD)
            form = choice(["on {w}", "this {w}", "next {w}"])
            s = form.format(w=wd)
            return s, s.replace("on ", "")
        t = choice([f"{day}/{month}", f"{EN_MONTHS[month - 1]} {day}", f"{day} {EN_MONTHS[month - 1]}"])
        return f"on {t}", t
    if lang == "el":
        if k < 0.25:
            t = choice(["σήμερα", "απόψε", "αύριο", "μεθαύριο", "αύριο το βράδυ"])
            return t, t
        if k < 0.65:
            art, wd = choice(EL_WD)
            s = choice([f"{art} {wd}", f"{wd}", f"αυτή {art} {wd}" if art != "το" else f"αυτό το {wd}",
                        f"{'την επόμενη' if art != 'το' else 'το επόμενο'} {wd}"])
            return s, s
        t = choice([f"{day}/{month}", f"{day} {EL_MONTHS_GEN[month - 1]}"])
        return t, t
    if k < 0.3:
        t = choice(["simera", "apopse", "avrio"])
        return t, t
    wd = choice(GL_WD)
    s = choice([f"tin {wd}", wd])
    return s, s


def time_surface(lang) -> str:
    h24 = rng.randint(12, 22)
    h12 = h24 - 12 if h24 > 12 else 12
    half = rng.random() < 0.25
    if lang == "en":
        return choice([f"{h12}pm", f"{h12} pm", f"{h24}:00", f"{h12}:30pm" if half else f"{h12} o'clock",
                       f"{h24}:30" if half else f"{h12} in the evening"])
    if lang == "el":
        return choice([f"{h12} το βράδυ" if h24 >= 18 else f"{h12} το μεσημέρι", f"{h24}:00", f"{h12}",
                       f"{h12}:30" if half else f"{h12} μμ", f"{h12} και μισή" if half else f"{h24}:00"])
    return choice([f"{h12} to vrady", f"{h24}:00", f"{h12}"])


def people_surface(n, lang):
    if lang == "en":
        return choice([str(n), PEOPLE_EN[n]])
    if lang == "el":
        return choice([str(n), PEOPLE_EL[n]])
    return str(n)


def gen_make_reservation(lang):
    n = weighted([(2, 0.3), (4, 0.25), (3, 0.1), (5, 0.08), (6, 0.08), (1, 0.05), (8, 0.06), (10, 0.04), (7, 0.04)])
    ds, dt = date_surface(lang)
    ts = time_surface(lang)
    ns = people_surface(n, lang)
    name = choice(NAMES[lang]) if rng.random() < 0.2 else None
    mode = weighted([("full", 0.5), ("no_party", 0.1), ("only_party", 0.1), ("bare", 0.1), ("followup", 0.2)])
    args = {"party_size": n, "date_text": dt, "time_text": ts, "name": name}
    ctx: dict = {}

    if mode == "full":
        msg = choice({
            "en": ["I'd like to reserve a table for {n} {d} at {t}", "Book a table for {n} people {d} at {t}",
                   "Can I make a reservation for {n} {d} at {t}?", "Table for {n}, {d} at {t} please",
                   "We are {n} people, can we book {d} at {t}?", "I want to book a table for {n} persons {d} at {t}"],
            "el": ["Θέλω να κλείσω τραπέζι για {n} {d} στις {t}", "Κράτηση για {n} άτομα {d} στις {t}",
                   "Μπορώ να κάνω κράτηση για {n} άτομα {d} στις {t};", "Ένα τραπέζι για {n} {d} στις {t} παρακαλώ",
                   "Είμαστε {n} άτομα, έχετε τραπέζι {d} στις {t};", "Θα ήθελα τραπέζι για {n} άτομα {d} στις {t}"],
            "gl": ["thelo na kleiso trapezi gia {n} {d} stis {t}", "kratisi gia {n} atoma {d} stis {t}"],
        }[lang]).format(n=ns, d=ds, t=ts)
    elif mode == "no_party":
        args["party_size"] = None
        msg = choice({
            "en": ["I'd like to book a table {d} at {t}", "Can I reserve {d} at {t}?", "Is there a table free {d} at {t}?"],
            "el": ["Θέλω να κλείσω τραπέζι {d} στις {t}", "Μπορώ να κάνω κράτηση {d} στις {t};", "Έχετε τραπέζι {d} στις {t};"],
            "gl": ["thelo trapezi {d} stis {t}"],
        }[lang]).format(d=ds, t=ts)
    elif mode == "only_party":
        args.update(date_text=None, time_text=None)
        msg = choice({
            "en": ["I want to book a table for {n}", "Reservation for {n} people please", "Can I book a table for {n}?"],
            "el": ["Θέλω τραπέζι για {n} άτομα", "Κράτηση για {n} παρακαλώ", "Μπορώ να κλείσω τραπέζι για {n};"],
            "gl": ["thelo trapezi gia {n} atoma", "kratisi gia {n}"],
        }[lang]).format(n=ns)
    elif mode == "bare":
        args = {"party_size": None, "date_text": None, "time_text": None, "name": None}
        name = None
        msg = choice({
            "en": ["I want to book a table", "Can I make a reservation?", "Do you take reservations?",
                   "I'd like to reserve a table", "Book a table please"],
            "el": ["Θέλω να κλείσω τραπέζι", "Μπορώ να κάνω κράτηση;", "Κάνετε κρατήσεις;",
                   "Θα ήθελα να κλείσω ένα τραπέζι", "Κράτηση τραπεζιού παρακαλώ"],
            "gl": ["thelo na kleiso trapezi", "mporo na kano kratisi?"],
        }[lang])
    else:  # followup: ο βοηθός ρώτησε ήδη τα στοιχεία
        u = {"en": "I want to book a table", "el": "Θέλω να κλείσω τραπέζι", "gl": "thelo na kleiso trapezi"}[lang]
        a = RB.build("make_reservation", {"status": "incomplete", "missing": ["party_size", "date", "time"]}, code(lang))
        ctx = {"history": [("user", u), ("assistant", a)], "pending": "awaiting_reservation_details", "no_affix": True}
        sub = choice(["party", "datetime", "all", "name"])
        if sub == "party":
            args.update(date_text=None, time_text=None)
            msg = choice({"en": ["For {n} people", "{n} people", "{n}", "We are {n}"],
                          "el": ["Για {n} άτομα", "{n} άτομα", "{n}", "Είμαστε {n}"],
                          "gl": ["gia {n} atoma", "{n} atoma", "{n}"]}[lang]).format(n=ns)
        elif sub == "datetime":
            args["party_size"] = None
            msg = choice({"en": ["{d} at {t}", "{d}, {t}"], "el": ["{d} στις {t}"], "gl": ["{d} stis {t}"]}[lang]).format(d=ds, t=ts)
        elif sub == "all":
            msg = choice({"en": ["{n} people, {d} at {t}", "For {n}, {d} at {t}"],
                          "el": ["{n} άτομα, {d} στις {t}", "Για {n}, {d} στις {t}"],
                          "gl": ["{n} atoma, {d} stis {t}"]}[lang]).format(n=ns, d=ds, t=ts)
        else:
            name = choice(NAMES[lang])
            args = {"party_size": None, "date_text": None, "time_text": None, "name": name}
            msg = choice({"en": ["Under the name {x}", "The name is {x}"], "el": ["Στο όνομα {x}", "Όνομα: {x}"],
                          "gl": ["sto onoma {x}"]}[lang]).format(x=name)
            name = None  # ήδη μέσα στο μήνυμα

    if name and mode in ("full", "no_party", "only_party"):
        msg = msg.rstrip("?;") + {"en": f" under the name {name}", "el": f" στο όνομα {name}", "gl": f" sto onoma {name}"}[lang]
    msg = msg[0].upper() + msg[1:]
    return {"message": msg, "target": T("make_reservation", lang, "make_reservation", args), **ctx}


SMALL_TALK_MSGS = {
    "greeting": {"en": ["Hi", "Hello", "Hey", "Good evening", "Good morning", "Hi there!"],
                 "el": ["Γεια", "Γεια σας", "Καλησπέρα", "Καλημέρα", "Γεια σου!"], "gl": ["geia", "geia sas", "kalispera"]},
    "thanks": {"en": ["Thanks", "Thank you", "Thank you very much", "Thanks a lot!"],
               "el": ["Ευχαριστώ", "Ευχαριστώ πολύ", "Σας ευχαριστώ!"], "gl": ["efxaristo", "eyxaristo poly"]},
    "goodbye": {"en": ["Bye", "Goodbye", "See you", "That's all, bye"],
                "el": ["Αντίο", "Τα λέμε", "Γεια χαρά", "Καληνύχτα"], "gl": ["ta leme", "antio"]},
    "capabilities": {"en": ["Who are you?", "What can you do?", "Are you a robot?", "How can you help me?"],
                     "el": ["Ποιος είσαι;", "Τι μπορείς να κάνεις;", "Είσαι ρομπότ;", "Πώς μπορείς να με βοηθήσεις;"],
                     "gl": ["ti mporeis na kaneis?", "poios eisai?"]},
    "hours": {"en": ["What time do you open?", "What are your opening hours?", "When do you close?", "Are you open on Sunday?"],
              "el": ["Τι ώρα ανοίγετε;", "Ποιο είναι το ωράριό σας;", "Τι ώρα κλείνετε;", "Είστε ανοιχτά την Κυριακή;"],
              "gl": ["ti ora anoigete?", "ti ora kleinete?"]},
    "keep_cart": {"en": ["No", "Not yet", "No, wait", "Hold on, not yet"],
                  "el": ["Όχι", "Όχι ακόμα", "Όχι, περίμενε", "Όχι, θέλω κι άλλο"], "gl": ["oxi", "oxi akoma"]},
}


def gen_small_talk(lang):
    kind = choice(list(SMALL_TALK_MSGS))
    ex = {"message": choice(SMALL_TALK_MSGS[kind][lang]), "no_affix": True,
          "target": T("small_talk", lang, "none", {}, SMALL_TALK_REPLIES[kind][code(lang)])}
    if kind == "keep_cart":
        cart = random_cart(min_items=1)
        ex.update(cart=cart, pending="awaiting_order_confirmation", history=_order_history(lang, cart))
    return ex


def gen_out_of_scope(lang):
    msg = choice({
        "en": ["What's the weather like tomorrow?", "Who won the football match yesterday?", "Can you help me with my homework?",
               "Tell me a joke", "What is the capital of France?", "Can you book me a taxi?", "Write me a poem",
               "What's the bitcoin price?", "How do I fix my laptop?"],
        "el": ["Τι καιρό θα κάνει αύριο;", "Ποιος κέρδισε τον αγώνα χθες;", "Βοήθησέ με στην εργασία μου",
               "Πες μου ένα ανέκδοτο", "Ποια είναι η πρωτεύουσα της Γαλλίας;", "Μπορείς να μου καλέσεις ταξί;",
               "Γράψε μου ένα ποίημα", "Πόσο είναι το bitcoin;"],
        "gl": ["ti kairo tha kanei avrio?", "pes mou ena anekdoto", "voithise me stin ergasia mou"],
    }[lang])
    return {"message": msg, "target": T("out_of_scope", lang, "none", {}, SMALL_TALK_REPLIES["out_of_scope"][code(lang)])}


GENERATORS = {
    "ask_menu": gen_ask_menu, "ask_dish_info": gen_ask_dish_info, "ask_price": gen_ask_price,
    "ask_allergen": gen_ask_allergen, "order_food": gen_order_food, "remove_item": gen_remove_item,
    "view_cart": gen_view_cart, "confirm_order": gen_confirm_order, "cancel_order": gen_cancel_order,
    "make_reservation": gen_make_reservation, "small_talk": gen_small_talk, "out_of_scope": gen_out_of_scope,
}


# ---------------------------------------------------------------------------
def make_example(intent: str) -> dict:
    lang = weighted(LANG_WEIGHTS)
    ex = GENERATORS[intent](lang)
    history = ex.get("history")
    if history is None:
        history = random_history(lang)
    cart = ex.get("cart", random_cart())
    pending = ex.get("pending")
    if "pending" not in ex and cart and rng.random() < 0.1:
        pending = "awaiting_order_confirmation"   # ο πελάτης ρωτά κάτι άλλο ενώ περιμένουμε επιβεβαίωση
    message = augment(ex["message"], lang, allow_affixes=not ex.get("no_affix"))
    user_content = build_user_content(history, cart, pending, message)
    return {
        "intent": intent,
        "messages": build_messages(user_content, ex["target"]),
        "target": ex["target"],
        "meta": {"message": message, "pending": pending, "cart": cart, "history": history, "surface_lang": lang},
    }


def build_dataset(n: int) -> tuple[list[dict], list[dict]]:
    intents, weights = zip(*INTENT_WEIGHTS.items())
    seen: set[str] = set()
    examples: list[dict] = []
    attempts = 0
    while len(examples) < n and attempts < n * 20:
        attempts += 1
        intent = rng.choices(intents, weights=weights, k=1)[0]
        ex = make_example(intent)
        key = ex["messages"][1]["content"]
        if key in seen:
            continue
        seen.add(key)
        examples.append(ex)

    # stratified split ανά intent
    by_intent: dict[str, list] = defaultdict(list)
    for ex in examples:
        by_intent[ex["intent"]].append(ex)
    train, val = [], []
    for items in by_intent.values():
        rng.shuffle(items)
        k = max(1, round(len(items) * config.VAL_RATIO))
        val += items[:k]
        train += items[k:]
    rng.shuffle(train)
    rng.shuffle(val)
    for i, ex in enumerate(train):
        ex["id"] = f"train-{i:05d}"
    for i, ex in enumerate(val):
        ex["id"] = f"val-{i:05d}"
    return train, val


def save_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Δημιουργία συνθετικού dataset από το menu.json")
    parser.add_argument("--n", type=int, default=config.NUM_TRAIN_EXAMPLES, help="πλήθος παραδειγμάτων")
    args = parser.parse_args()

    train, val = build_dataset(args.n)
    save_jsonl(train, config.TRAIN_PATH)
    save_jsonl(val, config.VAL_PATH)

    langs = Counter(ex["meta"]["surface_lang"] for ex in train + val)
    print(f"✔ train: {len(train)}  val: {len(val)}  →  {config.DATA_DIR}")
    print("  Γλώσσες:", dict(langs))
    print("  Intents:", dict(Counter(ex["intent"] for ex in train)))
    print("\nΠαράδειγμα:")
    ex = train[0]
    print(ex["messages"][1]["content"])
    print("→", ex["messages"][2]["content"])


if __name__ == "__main__":
    main()
