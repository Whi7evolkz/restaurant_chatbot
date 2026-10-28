# AI Restaurant Ordering Assistant – Fine-Tuned Conversational Chatbot

Συνομιλιακός βοηθός εστιατορίου (ελληνικά, αγγλικά και greeklish) που:

- απαντά για **μενού, τιμές, υλικά και αλλεργιογόνα**,
- παίρνει **παραγγελίες** με καλάθι, σύνολο και επιβεβαίωση,
- κλείνει **κρατήσεις τραπεζιού**, που εμφανίζονται σε **ημερολόγιο** (.ics + εικόνα μήνα),
- καταγράφει κάθε μήνυμα σε **Excel με χρωματική επισήμανση**.

Το «μυαλό» του είναι ένα **μικρό ανοιχτό γλωσσικό μοντέλο (Qwen2.5-0.5B-Instruct)** που
**εκπαιδεύουμε εμείς (fine-tuning με LoRA)** πάνω σε δεδομένα που παράγονται από το μενού.
Όλα τρέχουν **τοπικά**, χωρίς API και χωρίς API key.

---

## 1. Δομή του project

```
restaurant_chatbot/
├── requirements.txt
├── README.md
├── main.py                       # CLI συνομιλίας (και --demo, --rules-only)
├── config.py                     # διαδρομές, μοντέλο, υπερπαράμετροι, ωράριο, χρώματα
├── menu.json                     # 24 πιάτα/ποτά (ελληνικά + αγγλικά, υλικά, 14 αλλεργιογόνα ΕΕ)
├── data/
│   ├── train.jsonl / val.jsonl   # παράγονται από το dataset_builder
│   └── test_dialogues.jsonl      # 30 χειρόγραφα, ρεαλιστικά μηνύματα για αξιολόγηση
├── models/
│   ├── lora_adapter/             # το fine-tuned adapter (μετά την εκπαίδευση)
│   └── intent_classifier.pt      # ο PyTorch classifier
├── training/
│   ├── dataset_builder.py        # συνθετικό dataset από το menu.json
│   ├── train.py                  # LoRA fine-tuning με καθαρό PyTorch loop
│   ├── train_intent.py           # εκπαίδευση του intent classifier
│   └── evaluate.py               # μετρικές, confusion matrix, base vs fine-tuned
├── chatbot/
│   ├── local_llm.py              # φόρτωση base + LoRA, παραγωγή JSON απόφασης
│   ├── intent_classifier.py      # torch.nn.Module (EmbeddingBag + MLP)
│   ├── prompt_format.py          # ΙΔΙΑ μορφή εισόδου σε εκπαίδευση και χρήση + parser εξόδου
│   ├── menu_data.py              # φόρτωση μενού, fuzzy αναζήτηση πιάτων/αλλεργιογόνων
│   ├── tools.py                  # get_menu, get_allergens, add_item, make_reservation, ...
│   ├── response_builder.py       # τελικές απαντήσεις (EL/EN) από τα πραγματικά δεδομένα
│   ├── rule_parser.py            # fallback: classifier + κανόνες
│   └── conversation.py           # μνήμη συνεδρίας, guard rails, ενορχήστρωση
├── data_handlers/
│   ├── excel_logger.py           # openpyxl + PatternFill
│   ├── order_manager.py          # καλάθι + orders.xlsx
│   └── reservation_manager.py    # ημερομηνίες, .ics (RFC 5545), calendar_view.png
└── outputs/                      # όλα τα παραγόμενα αρχεία
```

---

## 2. Εγκατάσταση (VS Code / terminal)

Χρειάζεστε **Python 3.11 ή νεότερη** (προτείνεται 3.12) και περίπου 3 GB ελεύθερο χώρο.

```bash
cd restaurant_chatbot
python -m venv venv

# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
```

Στο VS Code: `Ctrl+Shift+P` → **Python: Select Interpreter** → διαλέξτε αυτόν μέσα στο `venv`.

---

## 3. Εκτέλεση βήμα-βήμα

| # | Εντολή | Τι κάνει | Χρόνος |
|---|--------|----------|--------|
| 1 | `python -m training.dataset_builder` | Φτιάχνει 2.000 παραδείγματα (`data/train.jsonl`, `data/val.jsonl`) | 2 δευτ. |
| 2 | `python -m training.train_intent` | Εκπαιδεύει τον PyTorch intent classifier | ~10 δευτ. |
| 3 | `python -m training.train` | **LoRA fine-tuning** του Qwen2.5-0.5B | GPU: ~20–40 λεπτά · CPU: αρκετές ώρες |
| 4 | `python -m training.evaluate` | Μετρικές + `eval_report.xlsx` + confusion matrix | GPU: ~3 λεπτά · CPU: ~15 λεπτά |
| 5 | `python main.py` | Συνομιλία με τον fine-tuned βοηθό | 2–5 δευτ. ανά απάντηση σε CPU |

Χρήσιμες επιλογές:

```bash
python main.py --demo          # τρέχει αυτόματα τους διαλόγους-παραδείγματα
python main.py --rules-only    # λειτουργεί ΚΑΙ πριν την εκπαίδευση (classifier + κανόνες)

# Γρήγορη εκπαίδευση σε laptop χωρίς κάρτα γραφικών (λιγότερη ακρίβεια):
python -m training.dataset_builder --n 500
python -m training.train --epochs 1

python -m training.evaluate --compare-base   # σύγκριση με το μοντέλο ΧΩΡΙΣ fine-tuning
```

> Την πρώτη φορά που τρέχει το `train.py` ή το `main.py` κατεβαίνει το base μοντέλο
> (~1 GB) από το Hugging Face. Μετά δουλεύει και χωρίς internet.

---

## 4. Εκπαίδευση στο Google Colab (δωρεάν GPU) – για laptop χωρίς κάρτα γραφικών

1. Τρέξτε τοπικά τα βήματα 1 και 2. Μετά κάντε **zip** τον φάκελο `restaurant_chatbot`
   (χωρίς τον φάκελο `venv`).
2. Πηγαίνετε στο <https://colab.research.google.com> → **New notebook**.
3. **Runtime → Change runtime type → T4 GPU**.
4. Ανεβάστε το `restaurant_chatbot.zip` (εικονίδιο φακέλου αριστερά) και τρέξτε:

```python
!unzip -q restaurant_chatbot.zip
%cd restaurant_chatbot
# κρατάμε το torch του Colab (ταιριάζει με την GPU του) και προσθέτουμε τα υπόλοιπα
!pip install -q "transformers>=4.56" "peft>=0.17" accelerate icalendar dateparser
!python -m training.train
!python -m training.evaluate --compare-base
!zip -r -q results.zip models/lora_adapter outputs
```

5. Κατεβάστε το `results.zip` (δεξί κλικ → Download), κάντε το unzip **μέσα** στον
   φάκελο του project στο laptop σας και τρέξτε `python main.py`.

---

## 5. Τι παράγεται (φάκελος `outputs/`)

| Αρχείο | Περιγραφή |
|--------|-----------|
| `chat_log.xlsx` | Φύλλο **Chat Log**: Timestamp, Session ID, User Message, Detected Intent (classifier + βεβαιότητα), LLM Intent, Action, Bot Response, Highlight Reason. Οι γραμμές χρωματίζονται: 🟥 αλλεργιογόνα, 🟨 τιμές, 🟩 κρατήσεις, 🟦 επιβεβαιώσεις παραγγελίας. Φύλλο **Legend** με την επεξήγηση των χρωμάτων. |
| `orders.xlsx` | Μία γραμμή ανά προϊόν: Order ID (`ORD-20261024-001`), ποσότητα, τιμή μονάδας, σύνολο γραμμής, σύνολο παραγγελίας. |
| `reservations.xlsx` | Κωδικός `RSV-001`, άτομα, ημερομηνία, ώρα, ημέρα, έναρξη/λήξη ISO. |
| `restaurant_reservations.ics` | Όλες οι κρατήσεις σε ένα ημερολόγιο (RFC 5545, με VTIMEZONE Europe/Nicosia). Ανοίγει σε Google Calendar / Outlook / Apple Calendar. |
| `reservations/RSV-00X.ics` | Ένα αρχείο ανά κράτηση (αυτό που «παίρνει» ο πελάτης). |
| `calendar_view.png` | Μηνιαίο ημερολόγιο: οι μέρες με κρατήσεις είναι πράσινες, με πλήθος κρατήσεων και ώρες. |
| `training_loss.png`, `training_log.csv` | Καμπύλη train/validation loss της εκπαίδευσης. |
| `eval_report.xlsx`, `confusion_matrix.png` | Αποτελέσματα αξιολόγησης. |
| `errors.log` | Σφάλματα (π.χ. μη έγκυρο JSON, ανοιχτό αρχείο Excel). |

Παράδειγμα από το `python main.py --demo`:

```
🙂 Εσύ: Does the carbonara contain gluten?
🤖 Bot: Yes, the Spaghetti Carbonara contains gluten. All its allergens: gluten, eggs and
        milk / dairy. A gluten-free alternative I can suggest: Risotto ai Funghi and Grilled Sea Bass.
   [intent=ask_allergen, action=get_allergens, highlight=Allergen question]      → ΚΟΚΚΙΝΗ γραμμή

🙂 Εσύ: I want 2 pizzas margherita and one coke.
🤖 Bot: Added 2× Pizza Margherita (€9.00 each) and 1× Coca-Cola (€2.50). Total: €20.50.
        Shall I confirm the order, or would you like anything else?
🙂 Εσύ: Yes.
🤖 Bot: Order #ORD-20261024-001 confirmed! Total: €20.50. Thank you!                → ΜΠΛΕ γραμμή

🙂 Εσύ: Θέλω τραπέζι για 4 την Παρασκευή στις 8 το βράδυ
🤖 Bot: Έγινε! Η κράτηση #RSV-001 για 4 άτομα, Παρασκευή 30/10 στις 20:00, καταχωρήθηκε.
        Το αρχείο ημερολογίου (.ics) είναι: outputs/reservations/RSV-001.ics       → ΠΡΑΣΙΝΗ γραμμή
```

---

## 6. Πώς λειτουργεί (για την αναφορά)

### 6.1 Αρχιτεκτονική

```
 μήνυμα πελάτη
     │
     ├──► PyTorch intent classifier ───────────────► «Detected Intent» στο Excel
     │
     ├──► Fine-tuned LLM (Qwen2.5-0.5B + LoRA)
     │        είσοδος: ιστορικό (8 μηνύματα) + καλάθι + κατάσταση + μήνυμα
     │        έξοδος : {"intent","language","action","arguments","reply"}
     │        └─ αν το JSON δεν είναι έγκυρο → 2η προσπάθεια → fallback σε classifier + κανόνες
     │
     ├──► Guard rails (π.χ. αλλεργιογόνα ΠΑΝΤΑ μέσω get_allergens)
     ├──► Tools (menu.json, καλάθι, orders.xlsx, reservations.xlsx, .ics, png)
     ├──► Response builder (τιμές/αλλεργιογόνα ΜΟΝΟ από τα δεδομένα)
     └──► chat_log.xlsx με χρωματική επισήμανση
```

### 6.2 Fine-tuning και LoRA

**Fine-tuning** σημαίνει ότι παίρνουμε ένα ήδη εκπαιδευμένο γλωσσικό μοντέλο και το
εκπαιδεύουμε λίγο ακόμα σε δικά μας παραδείγματα, ώστε να εξειδικευτεί σε μια εργασία.
Εδώ η εργασία είναι: *«διάβασε το μήνυμα του πελάτη και αποφάσισε ποια ενέργεια χρειάζεται»*.

**LoRA (Low-Rank Adaptation):** αντί να αλλάξουμε όλα τα ~500 εκατομμύρια βάρη, τα
«παγώνουμε». Σε κάθε πίνακα προσοχής (`q_proj`, `k_proj`, `v_proj`, `o_proj`) προσθέτουμε
δύο μικρούς πίνακες `A` (r×d) και `B` (d×r) με r = 8. Το αποτέλεσμα γίνεται
`W' = W + (α/r)·B·A` και εκπαιδεύονται **μόνο** οι `A`, `B` (~0,2–0,5% των παραμέτρων).

Πλεονεκτήματα:

- χωράει στη μνήμη μιας δωρεάν GPU,
- το αποτέλεσμα είναι λίγα MB και όχι 1 GB,
- δεν «ξεχνάει» τη γενική γνώση της γλώσσας.

### 6.3 Γιατί Qwen2.5-0.5B-Instruct

- **Μικρό μέγεθος:** τρέχει σε CPU και εκπαιδεύεται σε δωρεάν Colab.
- **Είναι ήδη «Instruct»:** ξέρει να συνομιλεί και να ακολουθεί οδηγίες.
- **Πολυγλωσσικό**, με ελληνικά.
- **Καλό σε δομημένη έξοδο (JSON).**
- **Άδεια Apache 2.0:** κατεβαίνει χωρίς έγκριση.

Για καλύτερα ελληνικά αλλάξτε στο `config.py` σε `Qwen/Qwen2.5-1.5B-Instruct`
(ή `set RC_BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct`). Είναι πιο αργό.

### 6.4 Το dataset

Το `dataset_builder.py` παράγει παραδείγματα **από το ίδιο το menu.json**, άρα οι ετικέτες
είναι πάντα σωστές. Περιλαμβάνει:

- 12 intents,
- ~43% ελληνικά, ~42% αγγλικά, ~15% greeklish,
- παραλλαγές: χωρίς τόνους, πεζά, ορθογραφικά λάθη, ευγενικές φράσεις,
- ποσότητες σε λέξεις και αριθμούς,
- σχετικές ημερομηνίες («αύριο», «την Παρασκευή», «next Monday», «15/11»),
- **πολλαπλών βημάτων** παραδείγματα: «Ναι» μετά από «Να επιβεβαιώσω;», «Πόσο κάνει;»
  με αναφορά στο προηγούμενο πιάτο, «Για 4 άτομα» αφού ο βοηθός ρώτησε.

Κάθε παράδειγμα είναι μια συνομιλία *system → user → assistant*, όπου ο assistant
απαντά με ένα JSON, π.χ.:

```json
{"intent":"ask_allergen","language":"el","action":"get_allergens",
 "arguments":{"dish":"Spaghetti Carbonara","allergen":"gluten"},"reply":null}
```

### 6.5 Grounding: γιατί το μοντέλο δεν γράφει το ίδιο τιμές και αλλεργιογόνα

Ένα μικρό LLM μπορεί να «επινοήσει» πληροφορίες (hallucination). Για αλλεργιογόνα αυτό
είναι επικίνδυνο. Γι' αυτό:

1. Το μοντέλο αποφασίζει **μόνο** την ενέργεια και τα ορίσματα.
2. Η Python εκτελεί το εργαλείο πάνω στο `menu.json` (`tools.py`).
3. Η απάντηση γράφεται από τα πραγματικά δεδομένα (`response_builder.py`).
4. Ένας guard rail εξασφαλίζει ότι **κάθε** ερώτηση αλλεργιογόνου περνά από το `get_allergens()`.

### 6.5β Guard rails: «το LLM προτείνει, οι κανόνες επαληθεύουν»

Στο `conversation._guard()` η απόφαση του μοντέλου περνά από ελέγχους πριν εκτελεστεί.

1. **Επαλήθευση οντοτήτων.**
   - Αν το μήνυμα γράφει ρητά πιάτο, αλλεργιογόνο, ποσότητα, άτομα ή ώρα και το μοντέλο έβγαλε κάτι άλλο, υπερισχύει το μήνυμα.
   - Αν το μοντέλο αναφέρει πιάτο που δεν υπάρχει ούτε στο μήνυμα ούτε στη συζήτηση, θεωρείται hallucination και αγνοείται.
2. **Ensemble με τον classifier.** Όταν ο PyTorch classifier είναι σχεδόν βέβαιος (≥ 0,97) και διαφωνεί, ακολουθείται ο classifier. Στα χειρόγραφα test dialogues, οι προβλέψεις του με τόση βεβαιότητα ήταν όλες σωστές.
3. **Ενέργειες που αλλάζουν την παραγγελία** (επιβεβαίωση, ακύρωση, αφαίρεση) εκτελούνται μόνο αν συμφωνούν LLM και classifier.
4. **Small talk:** η απάντηση είναι πάντα ένα από τα εγκεκριμένα κείμενα, όχι ελεύθερο κείμενο του μοντέλου.

Έτσι ένα λάθος του μικρού μοντέλου δεν φτάνει στον πελάτη ως λάθος πληροφορία.

### 6.6 Πού χρησιμοποιείται το PyTorch

- **`training/train.py`:** καθαρό PyTorch training loop.
  - `Dataset` και `DataLoader` με padding,
  - masking του prompt με `-100`, ώστε το loss να μετρά μόνο την απάντηση,
  - `AdamW`, linear warmup scheduler, gradient accumulation και clipping,
  - mixed precision σε GPU, validation ανά epoch, αποθήκευση του καλύτερου checkpoint.
- **`chatbot/intent_classifier.py`:** δικό μας `torch.nn.Module`
  (`EmbeddingBag` πάνω σε hashed n-grams → `Linear` → `ReLU` → `Dropout` → `Linear`).
  - Εκπαιδεύεται σε δευτερόλεπτα.
  - Δίνει το «Detected Intent» και λειτουργεί ως fallback.
- **`chatbot/local_llm.py`:** inference του μοντέλου με PyTorch (CUDA / Apple MPS / CPU).

### 6.7 Excel, ημερολόγιο, μνήμη, σφάλματα

- **Excel:** `openpyxl` με `PatternFill(start_color=..., end_color=..., fill_type="solid")`.
  - Bold επικεφαλίδες και παγωμένη πρώτη γραμμή.
  - Αν το αρχείο είναι ανοιχτό στο Excel, γράφεται αντίγραφο με timestamp αντί για crash.
- **Ημερολόγιο:**
  - Το `icalendar` δημιουργεί έγκυρο RFC 5545 αρχείο (VEVENT με UID, DTSTAMP, DTSTART/DTEND με TZID, VTIMEZONE).
  - Το `matplotlib` σχεδιάζει τον μήνα.
  - Οι ημερομηνίες αναλύονται με κανόνες (ελληνικά/αγγλικά/greeklish) και με τη βιβλιοθήκη `dateparser`.
- **Έλεγχοι κράτησης:** όχι στο παρελθόν, μόνο 12:00–22:00, έως 12 άτομα. Αν λείπει κάτι, ο βοηθός ρωτά και θυμάται τα υπόλοιπα.
- **Μνήμη:** τα τελευταία **8 μηνύματα** ανά συνεδρία (`deque(maxlen=8)`) μπαίνουν στην είσοδο του μοντέλου μαζί με το καλάθι.
- **Σφάλματα:** όλα καταγράφονται στο `outputs/errors.log` και ο πελάτης παίρνει φιλικό μήνυμα.

---

## 7. Αξιολόγηση

Το `python -m training.evaluate --compare-base` γράφει στο `eval_report.xlsx`:

- **Summary:** JSON validity, intent / action accuracy, argument exact match, συμφωνία με τον classifier, χρόνος απόκρισης. Τα αποτελέσματα είναι για:
  - τον classifier,
  - το fine-tuned μοντέλο,
  - το base μοντέλο χωρίς fine-tuning.
- **Per Intent:** precision / recall / F1.
- **Errors:** κάθε λάθος πρόβλεψη με την ακατέργαστη έξοδο του μοντέλου.

Υπάρχουν δύο σύνολα δεδομένων:

- το **validation** (συνθετικό, ίδια κατανομή με την εκπαίδευση),
- τα **30 χειρόγραφα test dialogues**, γραμμένα με διαφορετική διατύπωση από τα templates.

Η διαφορά ανάμεσα στα δύο δείχνει πόσο γενικεύει το μοντέλο. Για παράδειγμα, ο classifier
πετυχαίνει ~98% στο validation αλλά ~90% στα χειρόγραφα. Αυτή η διαφορά είναι
ενδιαφέρουσα για τη συζήτηση στην αναφορά.

Η στήλη **Full pipeline acc** μετρά την τελική απόφαση **μετά** τα guard rails, δηλαδή
αυτό που πραγματικά βιώνει ο πελάτης. Συγκρίνοντάς τη με τις στήλες του σκέτου μοντέλου
φαίνεται πόσο συνεισφέρει κάθε κομμάτι:

- fine-tuning: base vs fine-tuned,
- guard rails: μοντέλο σκέτο vs full pipeline.

---

## 8. Περιορισμοί και μελλοντική δουλειά

- Τα δεδομένα είναι **συνθετικά**. Πραγματικές συνομιλίες θα βελτίωναν τη γενίκευση.
- Το 0.5B μοντέλο έχει περιορισμένα ελληνικά. Το 1.5B ή ένα 3B είναι καλύτερα, αλλά πιο αργά.
- Δεν ελέγχεται η διαθεσιμότητα τραπεζιών (χωρητικότητα ανά ώρα).
- Μελλοντικά: RAG για περισσότερες πληροφορίες, web interface (Streamlit), πληρωμή, σύνδεση με POS.

---

## 9. Πώς η υλοποίηση καλύπτει τις απαιτήσεις της εργασίας

| Απαίτηση | Πού υλοποιείται |
|----------|-----------------|
| Συνομιλία σε φυσική γλώσσα (μενού, τιμές, υλικά, αλλεργιογόνα) | `chatbot/conversation.py` → `RestaurantChatbot.handle()`, `chatbot/tools.py` |
| Παραγγελία χωρίς σερβιτόρο (καλάθι, σύνολο, επιβεβαίωση) | `data_handlers/order_manager.py` (`Cart`, `OrderManager.confirm`) |
| Κράτηση με ημερομηνία/ώρα σε ημερολόγιο | `data_handlers/reservation_manager.py` (`make`, `write_ics`, `render_calendar`) |
| Γλωσσικό μοντέλο **fine-tuned** (αντί για API) | `training/train.py` (LoRA), `chatbot/local_llm.py` |
| Dataset εκπαίδευσης | `training/dataset_builder.py` → `data/train.jsonl`, `data/val.jsonl` |
| Πραγματική χρήση PyTorch | `training/train.py` (training loop), `chatbot/intent_classifier.py` (`nn.Module`), `training/train_intent.py` |
| Intent ως context / fallback | `chatbot/intent_classifier.py`, `chatbot/rule_parser.py`, `conversation._guard()` |
| Tool calling (`get_menu`, `get_allergens`, `add_item`, `make_reservation`, ...) | Το LLM επιστρέφει `action` + `arguments` → `ToolExecutor.execute()` |
| Αλλεργιογόνα ΠΑΝΤΑ μέσω εργαλείου | `conversation._guard()` (κανόνας 1), `tools.get_allergens()` |
| Προστασία από hallucinations | `conversation._verify_entities()`, `response_builder.py` (grounding) |
| Μενού σε `menu.json`, όχι μέσα στο prompt | `menu.json`, `chatbot/menu_data.py` |
| Excel log με στήλες & χρώματα | `data_handlers/excel_logger.py` (`ChatLogger`, `PatternFill`) |
| Ξεχωριστά orders / reservations | `outputs/orders.xlsx`, `outputs/reservations.xlsx` |
| `.ics` RFC 5545 + `calendar_view.png` | `ReservationManager.write_ics()`, `render_calendar()` |
| Μνήμη N = 8 μηνυμάτων | `ChatSession.history = deque(maxlen=config.MEMORY_WINDOW)`, `prompt_format.build_user_content()` |
| Χειρισμός σφαλμάτων + `errors.log` | `main.setup_logging()`, `try/except` στο `conversation.handle()`, fallback μηνύματα |
| Χειρισμός όταν λείπει το μοντέλο | `main.py` (`INSTRUCTIONS_NO_ADAPTER`, `--rules-only`) |
| Αξιολόγηση | `training/evaluate.py` → `eval_report.xlsx`, `confusion_matrix.png` |
| Windows + Linux | `pathlib`, UTF-8 παντού, `tzdata`, `matplotlib` Agg backend |

---

## 10. Συχνά προβλήματα

| Πρόβλημα | Λύση |
|----------|------|
| `Δεν βρέθηκε εκπαιδευμένο μοντέλο` | Τρέξτε το `python -m training.train` ή βάλτε τον φάκελο `lora_adapter` από το Colab μέσα στο `models/`. |
| Πολύ αργή εκπαίδευση | Χρησιμοποιήστε το Colab (ενότητα 4) ή `--n 500` / `--epochs 1`. |
| `PermissionError` στο Excel | Κλείστε το αρχείο στο Excel. Μέχρι τότε γράφεται αντίγραφο με timestamp. |
| Περίεργοι χαρακτήρες στο terminal των Windows | Χρησιμοποιήστε το terminal του VS Code ή τρέξτε `chcp 65001`. |
| `ZoneInfoNotFoundError` στα Windows | `pip install tzdata` (υπάρχει ήδη στο requirements). |
