"""Check (and optionally fill) earnings_calendar.csv against SEC EDGAR.

Every US issuer files its earnings press release as an 8-K under Item 2.02 ("Results of
Operations and Financial Condition"), usually within minutes of publishing it. The 8-K's
acceptance time therefore gives both the report date and whether it came before the open (BMO)
or after the close (AMC). This script reads those filings from the EDGAR submissions API and
compares them with the calendar.

Run it on your own machine (SEC requires a User-Agent with a contact address; it is sent only to
sec.gov):
    python tools/build_earnings_calendar.py --user-agent "Jane Doe jane@example.com"
    python tools/build_earnings_calendar.py --user-agent "..." --symbols META AMZN --since 2019-07-01
    python tools/build_earnings_calendar.py --user-agent "..." --write      # add missing past dates

Limits, so read the report rather than trusting --write blindly:
  * An 8-K filed late (e.g. the next morning for an evening release) shifts the date by a day and
    flips AMC to BMO. Differences from the calendar are printed for you to resolve.
  * Item 2.02 also covers preliminary results and restatements: extra rows show up as
    "EDGAR only" and usually are not scheduled earnings reports.
  * EDGAR only knows the past. Upcoming dates come from each company's investor-relations
    announcement and must be added by hand (the live bot refuses entries without one).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.request
from typing import Iterable, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from src.events import load_earnings_calendar  # noqa: E402

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
ET = "America/New_York"
OPEN, CLOSE = dt.time(9, 30), dt.time(16, 0)
MATCH_DAYS = 20          # a calendar row and an 8-K this close together describe the same quarter


# ----------------------------------------------------------------------------- pure functions
def acceptance_et(value: str) -> pd.Timestamp:
    """EDGAR acceptance times are Eastern wall-clock times even though the string ends in "Z"
    (e.g. Apple's 2024-02-01 4:30 p.m. release is stamped "2024-02-01T16:30:52.000Z")."""
    return pd.Timestamp(value[:19]).tz_localize(ET)


def timing_of(accepted: pd.Timestamp) -> str:
    t = accepted.time()
    if t < OPEN:
        return "BMO"
    if t >= CLOSE:
        return "AMC"
    return ""            # during the session: unusual for a scheduled report, left blank for review


def earnings_filings(submissions: Iterable[dict], since: dt.date) -> pd.DataFrame:
    """8-K filings with Item 2.02 from one or more submissions JSON pages ("filings.recent" of
    the main file, or the bare column dict of an older page)."""
    rows = []
    for page in submissions:
        cols = page.get("filings", {}).get("recent", page)
        for form, items, accepted, accession in zip(cols.get("form", []), cols.get("items", []),
                                                    cols.get("acceptanceDateTime", []),
                                                    cols.get("accessionNumber", [])):
            if form != "8-K" or "2.02" not in str(items).split(","):
                continue
            ts = acceptance_et(accepted)
            if ts.date() < since:
                continue
            rows.append({"date": ts.date(), "timing": timing_of(ts), "accepted_et": ts.strftime("%Y-%m-%d %H:%M"),
                         "accession": accession})
    out = pd.DataFrame(rows, columns=["date", "timing", "accepted_et", "accession"])
    return out.drop_duplicates("accession").sort_values("date").reset_index(drop=True)


def compare(calendar_rows: list[tuple[dt.date, str]], edgar: pd.DataFrame, since: dt.date,
            until: dt.date) -> pd.DataFrame:
    """One row per quarter: match / timing differs / date differs / calendar only / EDGAR only.
    Only calendar dates inside [since, until] are checked (EDGAR cannot know future dates)."""
    cal = sorted((d, t) for d, t in calendar_rows if since <= d <= until)
    used, out = set(), []
    for d, t in cal:
        best, gap = None, MATCH_DAYS + 1
        for i, row in edgar.iterrows():
            g = abs((row["date"] - d).days)
            if i not in used and g < gap:
                best, gap = i, g
        if best is None:
            out.append({"calendar": d, "calendar_timing": t, "edgar": None, "edgar_timing": "", "accepted_et": "",
                        "status": "CALENDAR ONLY"})
            continue
        used.add(best)
        e = edgar.loc[best]
        status = "ok" if gap == 0 and (e["timing"] == t or not e["timing"]) else \
            "TIMING DIFFERS" if gap == 0 else "DATE DIFFERS"
        out.append({"calendar": d, "calendar_timing": t, "edgar": e["date"], "edgar_timing": e["timing"],
                    "accepted_et": e["accepted_et"], "status": status})
    for i, e in edgar.iterrows():
        if i not in used:
            out.append({"calendar": None, "calendar_timing": "", "edgar": e["date"], "edgar_timing": e["timing"],
                        "accepted_et": e["accepted_et"], "status": "EDGAR ONLY"})
    return pd.DataFrame(out, columns=["calendar", "calendar_timing", "edgar", "edgar_timing", "accepted_et", "status"])


def merged_rows(existing: pd.DataFrame, symbol: str, report: pd.DataFrame) -> pd.DataFrame:
    """Existing calendar plus EDGAR-only dates for `symbol` (never deletes or rewrites a row)."""
    add = report[report["status"] == "EDGAR ONLY"]
    new = pd.DataFrame({"symbol": symbol, "date": [d.isoformat() for d in add["edgar"]],
                        "timing": add["edgar_timing"].tolist(),
                        "source": [f"SEC 8-K Item 2.02 accepted {a} ET" for a in add["accepted_et"]]})
    return pd.concat([existing, new], ignore_index=True)


# ----------------------------------------------------------------------------- network
class Edgar:
    def __init__(self, user_agent: str, pause: float = 0.15):
        self.headers = {"User-Agent": user_agent, "Accept-Encoding": "identity"}
        self.pause = pause                        # SEC allows 10 requests per second

    def get(self, url: str) -> dict:
        time.sleep(self.pause)
        with urllib.request.urlopen(urllib.request.Request(url, headers=self.headers), timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def ciks(self) -> dict[str, int]:
        return {row["ticker"].upper(): int(row["cik_str"]) for row in self.get(TICKERS_URL).values()}

    def submissions(self, cik: int, since: dt.date) -> list[dict]:
        """The main file holds the latest ~1000 filings; issuers that file a lot (JPM's structured
        notes) need the older pages too."""
        main = self.get(SUBMISSIONS_URL.format(name=f"CIK{cik:010d}.json"))
        pages = [main]
        for extra in main.get("filings", {}).get("files", []):
            if pd.Timestamp(extra["filingTo"]).date() >= since:
                pages.append(self.get(SUBMISSIONS_URL.format(name=extra["name"])))
        return pages


# ----------------------------------------------------------------------------- CLI
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user-agent", default=os.getenv("SEC_USER_AGENT"),
                        help='"Name email@example.com" (SEC policy; or set SEC_USER_AGENT)')
    parser.add_argument("--symbols", nargs="+",
                        default=[s for s in config.UNIVERSES["expanded12"] if s not in config.ETF_SYMBOLS])
    parser.add_argument("--since", default="2019-07-01")
    parser.add_argument("--calendar", default=config.EARNINGS_CALENDAR_PATH)
    parser.add_argument("--write", action="store_true", help="append EDGAR-only dates to the calendar")
    args = parser.parse_args(argv)
    if not args.user_agent or "@" not in args.user_agent:
        print('SEC requires a User-Agent with a contact e-mail: --user-agent "Jane Doe jane@example.com"')
        return 2
    since, today = pd.Timestamp(args.since).date(), dt.date.today()
    calendar = load_earnings_calendar(args.calendar)
    existing = (pd.read_csv(args.calendar, comment="#", dtype=str, skipinitialspace=True).fillna("")
                if os.path.exists(args.calendar) else pd.DataFrame(columns=["symbol", "date", "timing", "source"]))
    edgar = Edgar(args.user_agent)
    ciks = edgar.ciks()
    problems = 0
    for symbol in args.symbols:
        if symbol not in ciks:
            print(f"{symbol}: no CIK in {TICKERS_URL}")
            continue
        filings = earnings_filings(edgar.submissions(ciks[symbol], since), since)
        report = compare(calendar.get(symbol, []), filings, since, today)
        hours = pd.to_datetime(filings["accepted_et"]).dt.hour.value_counts().sort_index()
        print(f"\n{symbol}: {len(filings)} Item 2.02 8-Ks since {since}; acceptance hours (ET) "
              + ", ".join(f"{h:02d}h x{n}" for h, n in hours.items()))
        bad = report[report["status"] != "ok"]
        problems += len(bad)
        print(bad.to_string(index=False) if len(bad) else "  every calendar date matches an 8-K")
        if args.write:
            existing = merged_rows(existing, symbol, report)
    if args.write:
        header = [line for line in open(args.calendar)] if os.path.exists(args.calendar) else []
        header = [line for line in header if line.startswith("#")]          # keep the file's notes
        with open(args.calendar, "w", newline="") as fh:
            fh.writelines(header)
            existing.sort_values(["symbol", "date"]).to_csv(fh, index=False)
        print(f"\nWrote {args.calendar}. Check the EDGAR-only rows: some are not scheduled reports.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
