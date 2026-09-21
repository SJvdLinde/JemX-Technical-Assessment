"""Every rule and magic number lives here, so nothing is buried inside logic.

The values in RULES were not chosen by us. They were reverse-engineered from
`weekly_summary.csv`, which is the output of the client's existing system, and
they reproduce it on all 2,122 rows with zero difference. See test_hours.py.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

# --- The seven files a client export always contains. -----------------------
EXPECTED_FILES: dict[str, list[str]] = {
    "shifts": [
        "shift_id", "employee_id", "site_id",
        "shift_date", "clock_in_time", "clock_out_time",
    ],
    "employees": [
        "employee_id", "full_name", "id_number", "role", "primary_site_id",
        "shift_pattern", "contract_ordinary_hours", "employment_type",
    ],
    "sites": ["site_id", "site_name", "province"],
    "shift_notes": ["shift_id", "logged_by", "note"],
    "public_holidays": ["date", "name"],
    "weekly_summary": [
        "employee_id", "week_starting", "total_hours", "overtime_hours", "breached",
    ],
    "payroll_details": [
        "employee_id", "full_name", "id_number", "bank_name", "branch_code",
        "account_number", "account_type", "tax_number", "hourly_rate", "pay_frequency",
    ],
}

# Only `weekly_summary` is genuinely optional: it is the client's own roll-up,
# useful as a cross-check, but every number we publish is derived from `shifts`.
REQUIRED_FILES = {"shifts", "employees"}

# Columns that must never reach the API response or the frontend. Synthetic here,
# but this is payroll data and Jem builds payroll software.
PII_COLUMNS = {
    "id_number", "bank_name", "branch_code",
    "account_number", "account_type", "tax_number",
}

# --- Basic Conditions of Employment Act, as the client's engine implements it. ---
ORDINARY_HOURS_CAP = 45.0   # per Mon-Sun week
OVERTIME_CAP = 10.0         # per Mon-Sun week
BREACH_TOTAL_HOURS = ORDINARY_HOURS_CAP + OVERTIME_CAP  # 55.0

# Pay multipliers. COST ONLY -- they play no part in the breach test.
# The client's engine demonstrably ignores them when computing total_hours.
OVERTIME_MULTIPLIER = 1.5
SUNDAY_MULTIPLIER = 2.0
PUBLIC_HOLIDAY_MULTIPLIER = 2.0

# --- Data-quality rules discovered in Phase 0. ------------------------------
# No shift anywhere exceeds 13.5h, across every role and site. 169 shifts sit at
# 13.00h, 33 at 13.25h, then 366 at exactly 13.50h -- an 11x pile-up against a
# wall. That is truncation, not a distribution. Real hours above it are
# unrecoverable, so our totals are a LOWER BOUND.
SHIFT_LENGTH_CEILING = 13.5

# A shift whose clock-out is earlier than its clock-in crossed midnight; add 24h.
# All 1,010 such shifts belong to night-pattern security guards, and wrapping
# gives sane 3.5-13.5h durations, so this is safe.
MIDNIGHT_WRAP_HOURS = 24.0

# Sanity bounds. Anything outside these is a parse failure, not a long shift.
MIN_PLAUSIBLE_SHIFT_HOURS = 0.25
MAX_PLAUSIBLE_SHIFT_HOURS = 24.0

# --- Imputation of unclosed shifts. -----------------------------------------
# 184 shifts have a clock-in and no clock-out. The client's system drops them,
# which records them as ZERO hours rather than unknown. That bias is strictly
# one-directional: imputing flips 13 employee-weeks from non-breach to breach
# across the 9 complete weeks, and zero the other way -- so their system has been
# hiding roughly 1 in 6 real breaches.
#
# We fill with the employee's own median closed-shift length. Every employee has
# at least 26 closed shifts (median 41), so this is always well supported and no
# role-level fallback is needed -- but one is implemented for future exports.
IMPUTE_UNCLOSED_SHIFTS = True
MIN_SHIFTS_FOR_PERSONAL_MEDIAN = 5

# Weeks run Monday to Sunday (the brief says so; all ten weeks in the data agree).
WEEK_START_WEEKDAY = 0  # Monday
