# ============ data_handlers/reservation_manager.py ============
"""
Κρατήσεις τραπεζιών:
  1. Μετατροπή φράσεων όπως «την Παρασκευή στις 8 το βράδυ» σε datetime.
  2. Έλεγχοι (παρελθόν, ωράριο, μέγεθος παρέας).
  3. Αποθήκευση στο reservations.xlsx (κωδικός RSV-NNN).
  4. Αρχείο ημερολογίου .ics (RFC 5545) – ένα συνολικό + ένα ανά κράτηση.
  5. Μηνιαίο ημερολόγιο calendar_view.png με matplotlib.
"""
from __future__ import annotations

import calendar as pycal
import logging
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import config
from chatbot.menu_data import NUMBER_WORDS, normalize
from data_handlers.excel_logger import ExcelSheetWriter

log = logging.getLogger(__name__)
TZ = ZoneInfo(config.TIMEZONE)

WEEKDAYS = {
    0: ["monday", "mon", "δευτερα", "deftera", "deutera"],
    1: ["tuesday", "tue", "τριτη", "triti"],
    2: ["wednesday", "wed", "τεταρτη", "tetarti"],
    3: ["thursday", "thu", "πεμπτη", "pempti"],
    4: ["friday", "fri", "παρασκευη", "paraskevi", "paraskeui"],
    5: ["saturday", "sat", "σαββατο", "σαβατο", "savvato", "savato"],
    6: ["sunday", "sun", "κυριακη", "kyriaki", "kiriaki"],
}
MONTHS = {
    1: ["january", "jan", "ιανουαριου", "ιανουαριοσ", "ianouariou"],
    2: ["february", "feb", "φεβρουαριου", "φεβρουαριοσ", "fevrouariou"],
    3: ["march", "mar", "μαρτιου", "μαρτιοσ", "martiou"],
    4: ["april", "apr", "απριλιου", "απριλιοσ", "apriliou"],
    5: ["may", "μαιου", "μαιοσ", "maiou"],
    6: ["june", "jun", "ιουνιου", "ιουνιοσ", "iouniou"],
    7: ["july", "jul", "ιουλιου", "ιουλιοσ", "iouliou"],
    8: ["august", "aug", "αυγουστου", "αυγουστοσ", "avgoustou"],
    9: ["september", "sep", "sept", "σεπτεμβριου", "σεπτεμβριοσ", "septemvriou"],
    10: ["october", "oct", "οκτωβριου", "οκτωβριοσ", "oktovriou"],
    11: ["november", "nov", "νοεμβριου", "νοεμβριοσ", "noemvriou"],
    12: ["december", "dec", "δεκεμβριου", "δεκεμβριοσ", "dekemvriou"],
}
GREEK_MONTHS = ["Ιανουάριος", "Φεβρουάριος", "Μάρτιος", "Απρίλιος", "Μάιος", "Ιούνιος", "Ιούλιος",
                "Αύγουστος", "Σεπτέμβριος", "Οκτώβριος", "Νοέμβριος", "Δεκέμβριος"]
GREEK_DAYS_SHORT = ["Δευ", "Τρι", "Τετ", "Πεμ", "Παρ", "Σαβ", "Κυρ"]


# ---------------------------------------------------------------------------
# Ανάλυση ημερομηνίας & ώρας
# ---------------------------------------------------------------------------
def _has(norm: str, words) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", norm) for w in words)


def parse_date(text: str | None, now: datetime, use_dateparser: bool = True) -> date | None:
    if not text:
        return None
    norm = normalize(text)
    today = now.date()
    if _has(norm, ["day after tomorrow", "μεθαυριο", "methavrio", "meuavrio"]):
        return today + timedelta(days=2)
    if _has(norm, ["tomorrow", "αυριο", "avrio", "aurio"]):
        return today + timedelta(days=1)
    if _has(norm, ["today", "tonight", "σημερα", "αποψε", "simera", "apopse"]):
        return today

    # dd/mm(/yyyy) ή dd.mm ή dd-mm
    m = re.search(r"(?<!\d)(\d{1,2})\s*[/.\-]\s*(\d{1,2})(?:\s*[/.\-]\s*(\d{2,4}))?(?!\d)", text)
    if m:
        d, mo = int(m.group(1)), int(m.group(2))
        y = int(m.group(3)) if m.group(3) else today.year
        if y < 100:
            y += 2000
        try:
            result = date(y, mo, d)
            if not m.group(3) and result < today:
                result = date(y + 1, mo, d)
            return result
        except ValueError:
            pass   # π.χ. «20.00» είναι ώρα, όχι ημερομηνία → συνεχίζουμε

    # «20 Νοεμβρίου», «November 20», «5th of December»
    for mo, names in MONTHS.items():
        for name in names:
            if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", norm):
                dm = re.search(r"(?<!\d)(\d{1,2})(?:st|nd|rd|th|η)?(?!\d)", norm)
                if dm:
                    try:
                        result = date(today.year, mo, int(dm.group(1)))
                    except ValueError:
                        return None
                    return result if result >= today else date(today.year + 1, mo, int(dm.group(1)))

    # ημέρα της εβδομάδας
    for wd, names in WEEKDAYS.items():
        if _has(norm, names):
            ahead = (wd - today.weekday()) % 7
            if ahead == 0 and _has(norm, ["next", "επομενη", "επομενο", "epomeni", "epomeno"]):
                ahead = 7
            return today + timedelta(days=ahead)

    # τελευταία λύση: βιβλιοθήκη dateparser (αν είναι εγκατεστημένη)
    if not use_dateparser:
        return None
    try:
        import dateparser
        parsed = dateparser.parse(text, languages=["el", "en"],
                                  settings={"PREFER_DATES_FROM": "future",
                                            "RELATIVE_BASE": now.replace(tzinfo=None)})
        return parsed.date() if parsed else None
    except Exception:  # noqa: BLE001
        return None


def parse_time(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    norm = normalize(text).replace("o clock", "").replace("oclock", "")
    hour = minute = None
    m = re.search(r"(?<!\d)(\d{1,2})\s*[:.]\s*(\d{2})(?!\d)", norm)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
    else:
        m = re.search(r"(?<![\d/])(\d{1,2})(?![\d/])", norm)
        if m:
            hour, minute = int(m.group(1)), 0
        else:
            for tok in norm.split():
                if tok in NUMBER_WORDS and NUMBER_WORDS[tok] <= 12 and tok not in ("a", "an"):
                    hour, minute = NUMBER_WORDS[tok], 0
                    break
    if hour is None:
        return None
    if _has(norm, ["και μιση", "kai misi", "half past"]):
        minute = 30
    is_am = re.search(r"(?<!\w)(am|πμ|π μ)(?!\w)", norm) is not None
    if not is_am and 1 <= hour < 12:
        # «στις 8», «8pm», «8 το βράδυ» → 20:00 (το εστιατόριο ανοίγει 12:00)
        hour += 12
    if hour > 23 or minute > 59:
        return None
    return hour, minute


def to_int(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    s = normalize(str(value))
    m = re.search(r"\d+", s)
    if m:
        return int(m.group())
    for tok in s.split():
        if tok in NUMBER_WORDS:
            return NUMBER_WORDS[tok]
    return None


# ---------------------------------------------------------------------------
class ReservationManager:
    HEADERS = ["Reservation ID", "Created At", "Session ID", "Name", "Party Size", "Date", "Time", "Day",
               "Start (ISO)", "End (ISO)"]
    WIDTHS = [16, 20, 14, 16, 11, 12, 8, 12, 28, 28]

    def __init__(self, xlsx=config.RESERVATIONS_XLSX, ics=config.ICS_PATH, png=config.CALENDAR_PNG,
                 single_dir=config.SINGLE_ICS_DIR):
        self.writer = ExcelSheetWriter(xlsx, "Reservations", self.HEADERS, self.WIDTHS)
        self.ics_path = Path(ics)
        self.png_path = Path(png)
        self.single_dir = Path(single_dir)
        self.single_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def make(self, arguments: dict, partial: dict | None, session_id: str, now: datetime | None = None) -> dict:
        now = now or datetime.now(TZ)
        merged = dict(partial or {})
        for k, v in (arguments or {}).items():
            if v not in (None, "", []):
                merged[k] = v

        party = to_int(merged.get("party_size"))
        date_text, time_text = merged.get("date_text"), merged.get("time_text")
        d = parse_date(date_text, now)
        t = parse_time(time_text)

        if date_text and d is None:
            merged.pop("date_text", None)
            return {"status": "unparsed", "partial": merged}
        if time_text and t is None:
            merged.pop("time_text", None)
            return {"status": "unparsed", "partial": merged}
        missing = [k for k, v in (("party_size", party), ("date", d), ("time", t)) if v is None]
        if missing:
            return {"status": "incomplete", "missing": missing, "partial": merged}

        start = datetime.combine(d, time(*t), tzinfo=TZ)
        if start < now:
            merged.pop("date_text", None)
            merged.pop("time_text", None)
            return {"status": "past", "partial": merged}
        if not (config.OPENING_HOUR <= t[0] <= config.LAST_SEATING_HOUR) or (t[0] == config.LAST_SEATING_HOUR and t[1] > 0):
            merged.pop("time_text", None)
            return {"status": "closed", "partial": merged}
        if party < 1 or party > config.MAX_PARTY_SIZE:
            merged.pop("party_size", None)
            return {"status": "party_too_large", "partial": merged}

        rid = self._next_id()
        end = start + timedelta(hours=config.RESERVATION_DURATION_HOURS)
        name = merged.get("name") or ""
        self.writer.append_rows([[rid, now.strftime("%Y-%m-%d %H:%M:%S"), session_id, name, party,
                                  start.strftime("%Y-%m-%d"), start.strftime("%H:%M"), start.strftime("%A"),
                                  start.isoformat(), end.isoformat()]],
                                fill_color=config.HIGHLIGHT_COLORS["Reservation request"])
        single = self.single_dir / f"{rid}.ics"
        try:
            self.write_ics([self._row_dict(rid, name, party, start, end)], single)
            self.regenerate_ics()
            self.render_calendar(start.year, start.month)
        except Exception:  # noqa: BLE001  — η κράτηση έχει ήδη σωθεί στο Excel
            log.exception("Σφάλμα στη δημιουργία .ics / ημερολογίου")
        return {"status": "ok", "reservation_id": rid, "party_size": party, "datetime": start, "name": name,
                "ics_file": str(single.relative_to(config.BASE_DIR)) if single.is_relative_to(config.BASE_DIR) else str(single)}

    def _next_id(self) -> str:
        return f"RSV-{len(self.writer.read_rows()) + 1:03d}"

    @staticmethod
    def _row_dict(rid, name, party, start, end) -> dict:
        return {"Reservation ID": rid, "Name": name, "Party Size": party,
                "Start (ISO)": start.isoformat(), "End (ISO)": end.isoformat()}

    def all_reservations(self) -> list[dict]:
        return self.writer.read_rows()

    # ------------------------------------------------------------------
    # ICS (RFC 5545)
    # ------------------------------------------------------------------
    def write_ics(self, rows: list[dict], path: Path) -> Path:
        from icalendar import Calendar, Event

        cal = Calendar()
        cal.add("prodid", f"-//{config.RESTAURANT_NAME_EN}//Reservation Chatbot//EN")
        cal.add("version", "2.0")
        cal.add("calscale", "GREGORIAN")
        cal.add("method", "PUBLISH")
        cal.add("x-wr-calname", f"{config.RESTAURANT_NAME_EN} – Reservations")
        cal.add("x-wr-timezone", config.TIMEZONE)
        for r in rows:
            # astimezone(TZ) → TZID=Europe/Nicosia (και όχι σκέτο offset +02:00)
            start = datetime.fromisoformat(str(r["Start (ISO)"])).astimezone(TZ)
            end = datetime.fromisoformat(str(r["End (ISO)"])).astimezone(TZ)
            ev = Event()
            ev.add("uid", f"{r['Reservation ID']}@taverna-aroma.local")
            ev.add("dtstamp", datetime.now(timezone.utc))
            ev.add("dtstart", start)
            ev.add("dtend", end)
            who = f" ({r['Name']})" if r.get("Name") else ""
            ev.add("summary", f"Table reservation {r['Reservation ID']} – {r['Party Size']} people{who}")
            ev.add("description", f"{config.RESTAURANT_NAME_EN}: table for {r['Party Size']} people. "
                                  f"Reservation {r['Reservation ID']} made by the AI ordering assistant.")
            ev.add("location", f"{config.RESTAURANT_NAME_EN}, {config.RESTAURANT_ADDRESS}")
            ev.add("status", "CONFIRMED")
            cal.add_component(ev)
        if hasattr(cal, "add_missing_timezones"):
            cal.add_missing_timezones()   # προσθέτει VTIMEZONE για το Europe/Nicosia
        Path(path).write_bytes(cal.to_ical())
        return Path(path)

    def regenerate_ics(self) -> Path:
        return self.write_ics(self.all_reservations(), self.ics_path)

    # ------------------------------------------------------------------
    # Μηνιαίο ημερολόγιο (matplotlib)
    # ------------------------------------------------------------------
    def render_calendar(self, year: int, month: int) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        by_day: dict[int, list[str]] = {}
        for r in self.all_reservations():
            start = datetime.fromisoformat(str(r["Start (ISO)"]))
            if start.year == year and start.month == month:
                by_day.setdefault(start.day, []).append(f"{start:%H:%M} · {r['Party Size']} άτ.")

        weeks = pycal.Calendar(firstweekday=0).monthdayscalendar(year, month)
        fig, ax = plt.subplots(figsize=(10, 1.6 + 1.25 * len(weeks)))
        ax.set_xlim(0, 7)
        ax.set_ylim(0, len(weeks) + 0.6)
        ax.axis("off")
        today = datetime.now(TZ).date()

        for c, name in enumerate(GREEK_DAYS_SHORT):
            ax.text(c + 0.5, len(weeks) + 0.3, name, ha="center", va="center", fontsize=11, fontweight="bold",
                    color="#1F4E78")
        for w, week in enumerate(weeks):
            y = len(weeks) - 1 - w
            for c, day in enumerate(week):
                if day == 0:
                    continue
                booked = day in by_day
                face = "#B7E1A1" if booked else ("#F2F2F2" if c < 5 else "#E8E8F0")
                ax.add_patch(Rectangle((c + 0.03, y + 0.03), 0.94, 0.94, facecolor=face,
                                       edgecolor="#2E7D32" if booked else "#BBBBBB", linewidth=2 if booked else 1))
                is_today = date(year, month, day) == today
                ax.text(c + 0.08, y + 0.85, str(day), fontsize=11, va="top",
                        fontweight="bold" if is_today else "normal", color="#C62828" if is_today else "black")
                if booked:
                    items = sorted(by_day[day])
                    ax.text(c + 0.92, y + 0.85, f"{len(items)}", fontsize=10, ha="right", va="top",
                            fontweight="bold", color="#2E7D32")
                    shown = items[:3] + ([f"+{len(items) - 3} ακόμα"] if len(items) > 3 else [])
                    ax.text(c + 0.5, y + 0.38, "\n".join(shown), fontsize=7.5, ha="center", va="center")
        ax.set_title(f"{config.RESTAURANT_NAME} – Κρατήσεις {GREEK_MONTHS[month - 1]} {year}",
                     fontsize=14, fontweight="bold", pad=12)
        fig.tight_layout()
        fig.savefig(self.png_path, dpi=150)
        plt.close(fig)
        return self.png_path
