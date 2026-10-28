# ============ config.py ============
"""
Κεντρικές ρυθμίσεις του project: διαδρομές αρχείων, μοντέλο, υπερπαράμετροι
εκπαίδευσης, ωράριο εστιατορίου και χρώματα επισήμανσης στο Excel.

Οι περισσότερες τιμές μπορούν να αλλάξουν και με μεταβλητές περιβάλλοντος
(π.χ. RC_EPOCHS=1) χωρίς να αλλάξει ο κώδικας.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Διαδρομές (pathlib → δουλεύει ίδια σε Windows και Linux)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

MENU_PATH = BASE_DIR / "menu.json"
TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH = DATA_DIR / "val.jsonl"
TEST_DIALOGUES_PATH = DATA_DIR / "test_dialogues.jsonl"

ADAPTER_DIR = MODELS_DIR / "lora_adapter"
INTENT_MODEL_PATH = MODELS_DIR / "intent_classifier.pt"

CHAT_LOG_XLSX = OUTPUTS_DIR / "chat_log.xlsx"
ORDERS_XLSX = OUTPUTS_DIR / "orders.xlsx"
RESERVATIONS_XLSX = OUTPUTS_DIR / "reservations.xlsx"
ICS_PATH = OUTPUTS_DIR / "restaurant_reservations.ics"
SINGLE_ICS_DIR = OUTPUTS_DIR / "reservations"          # ένα .ics ανά κράτηση
CALENDAR_PNG = OUTPUTS_DIR / "calendar_view.png"
ERROR_LOG = OUTPUTS_DIR / "errors.log"
TRAINING_LOSS_PNG = OUTPUTS_DIR / "training_loss.png"
TRAINING_LOG_CSV = OUTPUTS_DIR / "training_log.csv"
EVAL_XLSX = OUTPUTS_DIR / "eval_report.xlsx"
CONFUSION_PNG = OUTPUTS_DIR / "confusion_matrix.png"

for _d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR, SINGLE_ICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Μοντέλο (τοπικό, χωρίς API)
# ---------------------------------------------------------------------------
# Για καλύτερα ελληνικά: RC_BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
BASE_MODEL_NAME = os.getenv("RC_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ---------------------------------------------------------------------------
# Υπερπαράμετροι εκπαίδευσης
# ---------------------------------------------------------------------------
SEED = 42
NUM_TRAIN_EXAMPLES = int(os.getenv("RC_NUM_EXAMPLES", "2000"))  # σε CPU: 500
EPOCHS = int(os.getenv("RC_EPOCHS", "3"))                         # σε CPU: 1
BATCH_SIZE = int(os.getenv("RC_BATCH_SIZE", "4"))
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 1.0
MAX_SEQ_LEN = 512
VAL_RATIO = 0.10

# Παραγωγή απάντησης
MAX_NEW_TOKENS = 160

# ---------------------------------------------------------------------------
# Συνομιλία
# ---------------------------------------------------------------------------
MEMORY_WINDOW = 8           # πόσα τελευταία μηνύματα θυμάται η συνεδρία
HISTORY_CHARS = 160         # κάθε μήνυμα του ιστορικού κόβεται σε τόσους χαρακτήρες

# ---------------------------------------------------------------------------
# Εστιατόριο
# ---------------------------------------------------------------------------
RESTAURANT_NAME = "Ταβέρνα Άρωμα"
RESTAURANT_NAME_EN = "Taverna Aroma"
RESTAURANT_ADDRESS = "Limassol, Cyprus"
TIMEZONE = "Europe/Nicosia"
OPENING_HOUR = 12           # 12:00
LAST_SEATING_HOUR = 22      # τελευταία κράτηση 22:00
CLOSING_HOUR = 24           # κλείνει μεσάνυχτα
RESERVATION_DURATION_HOURS = 2
MAX_PARTY_SIZE = 12
CURRENCY_SYMBOL = "€"

# ---------------------------------------------------------------------------
# Χρώματα επισήμανσης Excel (hex χωρίς #)
# ---------------------------------------------------------------------------
HIGHLIGHT_COLORS = {
    "Allergen question": "FF0000",      # ΚΟΚΚΙΝΟ
    "Price question": "FFFF00",         # ΚΙΤΡΙΝΟ
    "Reservation request": "00FF00",    # ΠΡΑΣΙΝΟ
    "Order confirmation": "00B0F0",     # ΜΠΛΕ
}

# ---------------------------------------------------------------------------
# Intents & actions (κοινά για dataset, classifier και LLM)
# ---------------------------------------------------------------------------
INTENTS = [
    "ask_menu", "ask_dish_info", "ask_price", "ask_allergen", "order_food",
    "remove_item", "view_cart", "confirm_order", "cancel_order",
    "make_reservation", "small_talk", "out_of_scope",
]

ACTIONS = [
    "get_menu", "get_dish_info", "get_price", "get_allergens", "add_item",
    "remove_item", "view_cart", "confirm_order", "cancel_order",
    "make_reservation", "none",
]

# Προεπιλεγμένη ενέργεια για κάθε intent (χρησιμοποιείται στο fallback)
INTENT_TO_ACTION = {
    "ask_menu": "get_menu",
    "ask_dish_info": "get_dish_info",
    "ask_price": "get_price",
    "ask_allergen": "get_allergens",
    "order_food": "add_item",
    "remove_item": "remove_item",
    "view_cart": "view_cart",
    "confirm_order": "confirm_order",
    "cancel_order": "cancel_order",
    "make_reservation": "make_reservation",
    "small_talk": "none",
    "out_of_scope": "none",
}

# System prompt που βλέπει το μοντέλο (ίδιο στην εκπαίδευση και στη χρήση)
SYSTEM_PROMPT = (
    f"You are the ordering assistant of the restaurant \"{RESTAURANT_NAME_EN}\". "
    "Read the history, cart, pending state and the customer's message, then answer "
    "ONLY with one JSON object: "
    "{\"intent\": str, \"language\": \"el\"|\"en\", \"action\": str, "
    "\"arguments\": object, \"reply\": str|null}. "
    "Intents: " + ", ".join(INTENTS) + ". "
    "Actions: " + ", ".join(ACTIONS) + ". "
    "Use the English menu names for dishes. Never state prices or allergens yourself; "
    "the system reads them from the menu. Use \"reply\" only when action is \"none\"."
)
