# Era-matched timetables

What reading the archive CIFs changes, measured 2026-09-22. Companion to
`PAPER_METHODS.md`, whose limitations list this closes two entries of.

The model read one timetable: the daily feed's newline-delimited JSON, a May-26
extract. The counts it is matched against span **April 2024 to August 2026 --
six published timetables**. Every count was therefore resolved against a
timetable that, for 88% of rows, was not running when the count was taken.

The archive extracts that fix this are published in the original 80-column
fixed-width CIF, which the parser could not read. It can now
(`src/ortla/cif_archive.py`).

---

## 1. The reader, and how it is known to be correct

Fixed-width parsing fails silently: a column offset one character out still
parses, and the run completes with wrong times. So it is not asserted, it is
tested against the JSON issue of the same feed.

`scripts/verify_cif_archive.py` parses `CIF_ALL_FULL_DAILY_toc-full.CIF` and
`...json` -- the same daily extract in both formats -- and compares them record
by record.

| | result |
|---|---|
| location reference records compared | 12,091 |
| disagreements in CRS, description, STANOX, NALCO | **0** |
| schedules compared | 244,618 |
| schedules found in only one extract | 5,382 (2.2%) |
| field disagreements | ~3,500, across ~1% of schedules |

The two files are extracts taken four days apart, so they genuinely differ. The
disagreements are confined to the fields an extract revises -- passing times,
arrival, departure, platform, days run -- and one, `C01277`, was decoded by hand
from the raw record to confirm the reader and not the file was right. **A wrong
column offset would disagree in 100% of records, not 1%.**

Two format differences were found and are handled:

- **An unknown STANOX** is five zeroes in the CIF and null in the feed.
- **A published-timetable extract leaves the STP indicator blank.** The daily
  feed always fills it (C/N/O/P). Every `BS` record in the Dec-25 and May-26
  extracts has column 80 blank, because those extracts carry the permanent plan
  only. Every filter in this model selects permanent schedules by `"P"`, so
  reading blank as missing discards the entire extract -- silently. Blank is
  read as permanent.

The archives also carry the short-term plan: Dec-23 holds 982,853 schedules of
which **87,502 are permanent**, against 86,832 in the permanent-only Dec-25
extract. The two kinds of file agree on the quantity the model uses.

## 2. Era coverage of the counts

Six published timetables cover the counting period. Four are held:

| timetable | in force | share of the 22.3m loading rows | held |
|---|---|---:|---|
| May-23 | 21 May 2023 – 9 Dec 2023 | 0.0% | yes |
| Dec-23 | 10 Dec 2023 – 1 Jun 2024 | 4.7% | yes |
| Jun-24 | 2 Jun 2024 – 14 Dec 2024 | 19.7% | yes |
| **Dec-24** | 15 Dec 2024 – 17 May 2025 | **16.0%** | **no** |
| **May-25** | 18 May 2025 – 13 Dec 2025 | **27.0%** | **no** |
| Dec-25 | 14 Dec 2025 – 16 May 2026 | 20.6% | yes |
| May-26 | 17 May 2026 – 12 Dec 2026 | 12.0% | yes (the current feed) |

**43.0% of the counts fall in the Dec-24 / May-25 gap** and are still resolved
against the May-26 extract. Two more archive downloads close it.

## 3. What it buys: coverage

`configs/paper/eras/` cleans one calls table per era;
`LoadingConfig.timetable_eras` supplies them with their validity windows, and
each count is resolved against the timetable in force when it was taken.
Measured by `scripts/era_matched_departures.py`:

| operator | rows | fell back to the origin departure | era matched |
|---|---:|---:|---:|
| Greater Anglia | 2,119,397 | 28.4% | **21.1%** |
| Northern | 4,487,097 | 26.9% | **25.2%** |
| Southeastern | 4,121,043 | 20.3% | **16.1%** |

The origin fallback is the failure mode: it times a boarding a median 29 minutes
early and puts about 15% of calls in the wrong period.

For Northern the mix also improves in *quality*, which matters more than the
headline: matches on the exact rung rise **60.8% -> 66.2%**, while the loose
nearest-origin rung falls 11.0% -> 7.5%. Those rungs are not equally good --
0.2% of exact matches land in the wrong period against 1.3% of nearest-origin
ones -- so moving 3.5% of rows up the ladder is worth more than the coverage
figure alone suggests.

## 4. What it buys: accuracy

Coverage alone would reward a looser match, so the test is a hold-out: take the
2,251,943 Southeastern rows that *do* publish a departure, hide it, re-impute,
and score against the truth.

| | matched to the timetable | exact to the minute | in the wrong time period |
|---|---:|---:|---:|
| single extract | 71.2% | 94.2% | 6.35% |
| era matched | **74.2%** | **95.2%** | **5.91%** |

Coverage and accuracy both improve, which is the outcome a looser match cannot
produce.

**The design carries its own control.** Split by the era a count falls in:

| era the count falls in | rows | wrong period, single | wrong period, era matched |
|---|---:|---:|---:|
| Dec-25 | 897,293 | 6.26% | **5.17%** |
| May-26 | 455,908 | 6.35% | 6.35% |
| no era held (the Dec-24 / May-25 gap) | 898,742 | 6.43% | 6.43% |

Rows whose era *is* the baseline extract are unchanged to four figures, and so
are rows with no era held. Only the rows that got a different timetable move,
and they improve by **17.4% relative**. Nothing else in the pipeline could
produce that pattern.

60.1% of these rows fall outside a held era. Closing the Dec-24 / May-25 gap
would extend a 17% reduction to them too.

## 5. What is now possible that was not

- **A real before-network.** `StripRule` synthesises one by deleting a service
  from the May-26 timetable. Case 01 (Northumberland Line, opened Dec 2024) can
  now be run against the **Jun-24 timetable as published**, which is what was
  actually running the day before it opened. That retires the caveat in
  `PAPER_METHODS.md` correction 9 for the cases an archive covers.
- **A demand panel.** Six timetables against seven years of ODM is the input to
  the within-pair frequency elasticity in `ACCURACY_IMPROVEMENTS.md` §7, which
  is the first estimate in this study that is not cross-sectional.

## 6. Ask

Two files close the remaining 43%: the **Dec-24** and **May-25** published
archive CIFs, in the same form as the ones already held.
