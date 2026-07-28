from datetime import date
import re

def calc_age(dob_str):
    # Try full date first: DD-MM-YYYY
    full_match = re.search(r'(\d{2})-(\d{2})-(\d{4})', dob_str)
    if full_match:
        d, m, y = map(int, full_match.groups())
        try:
            dob = date(y, m, d)
        except ValueError:
            return None
        today = date.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

    # Fallback: year only (less accurate, off by max 1 year)
    year_match = re.search(r'(19|20)\d{2}', dob_str)
    if year_match:
        return date.today().year - int(year_match.group())

    return None


def validate_age_category(dob, stated_category):
    age = calc_age(dob)
    if age is None:
        return {"field": "age_category", "valid": False, "reason": "DOB unreadable"}

    ranges = {"child": (0, 17), "adult": (18, 59), "senior": (60, 120)}
    if stated_category not in ranges:
        return {"field": "age_category", "valid": False, "reason": f"Unknown category '{stated_category}'"}

    lo, hi = ranges[stated_category]
    valid = lo <= age <= hi
    return {
        "field": "age_category",
        "computed_age": age,
        "stated_category": stated_category,
        "valid": valid,
        "reason": None if valid else f"Age {age} doesn't match '{stated_category}'"
    }


def validate_expiry(expiry_date_str):
    match = re.search(r'(\d{2})-(\d{2})-(\d{4})', expiry_date_str)
    if not match:
        return {"field": "expiry_date", "valid": False, "reason": "Date unreadable"}
    d, m, y = map(int, match.groups())
    try:
        exp = date(y, m, d)
    except ValueError:
        return {"field": "expiry_date", "valid": False, "reason": "Invalid date"}
    valid = exp >= date.today()
    return {
        "field": "expiry_date",
        "expiry": str(exp),
        "valid": valid,
        "reason": None if valid else "Document expired"
    }


def validate_all(fields, stated_category=None):
    results = []
    if fields.get('dob') and stated_category:
        results.append(validate_age_category(fields['dob'], stated_category))
    if fields.get('expiry_date'):
        results.append(validate_expiry(fields['expiry_date']))
    return results