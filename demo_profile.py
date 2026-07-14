"""Deterministic Demo Profile validation.

The LLM-facing extraction boundary may propose Feature Candidates, but this
module is the authority for units, field state, derivation, and confirmation.
See the accepted design at commit 79b16b4, sections 5, 10, 11, and 13.
"""

import json
import re
import time
from collections import defaultdict, deque
from copy import deepcopy


BLOCKING_STATUSES = {"ambiguous", "out_of_range", "conflicting"}
USABLE_STATUSES = {"valid", "outside_reference_support"}

FEATURE_EXTRACTION_SYSTEM_PROMPT = """You extract Feature Candidates for a research Demo Profile.
Return one JSON object with a single `candidates` array. Each candidate must contain:
`field`, `raw_value`, `raw_unit`, `source_text`, and `operation`.

Allowed operations are `set`, `correct`, and `remove`. Copy `source_text` exactly from
the visitor message. Use a unit only when it is explicit in that exact source text.
Do not validate, normalize, calculate, infer missing values, confirm a profile, match a
cohort, infer sex, or generate ICD codes. Preserve unsupported fields as candidates.
If the message contains no profile field, including a request such as "predict", return
exactly {"candidates": []}.
Return JSON only.
"""

# These support bounds come from the current committed fibrotic demo cohort.
# They flag cohort coverage only; they are not medical reference intervals.
REFERENCE_SUPPORT = {
    "age": (33, 90),
    "height": (140, 210),
    "weight": (37.6, 165.8),
    "bmi": (15, 60),
    "waist": (50.7, 163.1),
    "hip": (65.3, 177.5),
    "dbp": (46, 120),
    "sbp": (80, 220),
    "creatinine": (20.4, 500),
    "hba1c": (20, 113),
}

# Operational parser bounds reject obvious unit/entry errors before cohort
# support is considered. They are deliberately broad, are tested to contain the
# full current demo-cohort support, and are not healthy or diagnostic ranges.
# A Dataset Release change must rerun that validation before these are retained.
INPUT_BOUNDS = {
    "age": (0, 130),
    "height": (50, 300),
    "weight": (1, 500),
    "bmi": (5, 150),
    "waist": (20, 300),
    "hip": (20, 300),
    "dbp": (20, 250),
    "sbp": (30, 350),
    "creatinine": (1, 5000),
    "hba1c": (1, 250),
}

FIELD_ALIASES = {
    "age": "age",
    "age_recruit": "age",
    "sex": "sex",
    "height": "height",
    "weight": "weight",
    "bmi": "bmi",
    "waist": "waist",
    "waist_circumference": "waist",
    "hip": "hip",
    "hip_circumference": "hip",
    "blood_pressure": "blood_pressure",
    "bp": "blood_pressure",
    "sbp": "sbp",
    "systolic_blood_pressure": "sbp",
    "dbp": "dbp",
    "diastolic_blood_pressure": "dbp",
    "smoking": "smoking_status",
    "smoking_status": "smoking_status",
    "alcohol": "alcohol_frequency",
    "alcohol_freq": "alcohol_frequency",
    "alcohol_frequency": "alcohol_frequency",
    "affected_relative": "affected_relative",
    "affected_relative_status": "affected_relative",
    "family_history": "affected_relative",
    "has_affected_rel": "affected_relative",
    "creatinine": "creatinine",
    "hba1c": "hba1c",
    "hba1c_percent": "hba1c",
}

FIELD_META = {
    "age": ("Age", "demographics", "years"),
    "sex": ("Sex", "demographics", None),
    "height": ("Height", "body_composition", "cm"),
    "weight": ("Weight", "body_composition", "kg"),
    "bmi": ("BMI", "body_composition", "kg/m²"),
    "waist": ("Waist circumference", "body_composition", "cm"),
    "hip": ("Hip circumference", "body_composition", "cm"),
    "sbp": ("Systolic blood pressure", "blood_pressure", "mmHg"),
    "dbp": ("Diastolic blood pressure", "blood_pressure", "mmHg"),
    "smoking_status": ("Smoking status", "lifestyle", None),
    "alcohol_frequency": ("Alcohol frequency", "lifestyle", None),
    "affected_relative": ("Affected relative", "family_history", None),
    "creatinine": ("Creatinine", "optional_laboratory", "µmol/L"),
    "hba1c": ("HbA1c", "optional_laboratory", "mmol/mol"),
}


class AmbiguousValue(ValueError):
    pass


class OutsideInputRange(ValueError):
    pass


class ProfileRateLimiter:
    def __init__(self, limit=20, window_seconds=60):
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests = defaultdict(deque)

    def allow(self, key, now=None):
        now = time.monotonic() if now is None else now
        requests = self._requests[key]
        while requests and requests[0] <= now - self.window_seconds:
            requests.popleft()
        if len(requests) >= self.limit:
            return False
        requests.append(now)
        return True


def parse_feature_candidates(message, raw_content):
    content = raw_content.strip()
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("Feature extraction did not return valid JSON") from error
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(candidates, list):
        raise ValueError("Feature extraction did not return a candidates array")
    required = {"field", "raw_value", "raw_unit", "source_text", "operation"}
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != required:
            raise ValueError("Feature Candidate contract is invalid")
        if candidate["operation"] not in {"set", "correct", "remove"}:
            raise ValueError("Feature Candidate operation is invalid")
        if not candidate["source_text"] or candidate["source_text"] not in message:
            raise ValueError("Feature Candidate source text must be copied from the message")
    return candidates


def _number(value):
    if isinstance(value, bool) or value is None or value == "":
        raise AmbiguousValue("missing numeric value")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AmbiguousValue("value is not numeric") from error
    if result != result or result in {float("inf"), float("-inf")}:
        raise AmbiguousValue("value is not finite")
    return result


def _unit_key(unit):
    return (unit or "").strip().lower().replace(" ", "")


def _unit_is_explicit(field, unit, source_text):
    key = _unit_key(unit).replace("μ", "µ")
    patterns = {
        "cm": r"\bcm\b|\bcentimeters?\b",
        "m": r"\bmeters?\b|(?<![a-z])m(?![a-z])",
        "in": r"\bin(?:ch|ches)?\b|\"",
        "ft": r"\bft\b|\bfoot\b|\bfeet\b|'",
        "kg": r"\bkg\b|\bkilograms?\b",
        "lb": r"\blbs?\b|\bpounds?\b",
        "µmol/l": r"(?:µ|μ|u)mol\s*/\s*l|micromol\s*/\s*l",
        "mg/dl": r"mg\s*/\s*dl",
        "%": r"%|\bpercent\b|\bpct\b",
        "mmol/mol": r"mmol\s*/\s*mol",
        "years": r"\byears?\b|\byrs?\b",
        "kg/m2": r"kg\s*/\s*m(?:2|²)",
        "mmhg": r"\bmm\s*hg\b",
    }
    aliases = {
        "centimeter": "cm",
        "centimeters": "cm",
        "meter": "m",
        "meters": "m",
        "inch": "in",
        "inches": "in",
        "foot": "ft",
        "feet": "ft",
        "kilogram": "kg",
        "kilograms": "kg",
        "lbs": "lb",
        "pound": "lb",
        "pounds": "lb",
        "umol/l": "µmol/l",
        "micromol/l": "µmol/l",
        "mgdl": "mg/dl",
        "percent": "%",
        "pct": "%",
        "mmolmol": "mmol/mol",
        "year": "years",
        "yr": "years",
        "yrs": "years",
        "kg/m²": "kg/m2",
        "kgm2": "kg/m2",
        "kgm²": "kg/m2",
        "mmhg.": "mmhg",
    }
    if field in {"height", "waist", "hip"} and key in {"ft/in", "ft+in"}:
        return bool(re.search(patterns["ft"], source_text, re.IGNORECASE)) and bool(
            re.search(patterns["in"], source_text, re.IGNORECASE)
        )
    canonical = aliases.get(key, key)
    pattern = patterns.get(canonical)
    return bool(pattern and re.search(pattern, source_text, re.IGNORECASE))


def _with_unit(value, unit, factors, label):
    value = _number(value)
    key = _unit_key(unit)
    if key not in factors:
        raise AmbiguousValue(f"unsupported or missing {label} unit")
    return value * factors[key]


def _normalize_length(value, unit):
    if isinstance(value, str):
        compound = re.fullmatch(
            r"\s*(\d+(?:\.\d+)?)\s*(?:ft|foot|feet|')\s*"
            r"(\d+(?:\.\d+)?)\s*(?:in|inch|inches|\")\s*",
            value,
            re.IGNORECASE,
        )
        if compound:
            feet = float(compound.group(1))
            inches = float(compound.group(2))
            return round(feet * 30.48 + inches * 2.54, 2)
    factors = {
        "cm": 1,
        "centimeter": 1,
        "centimeters": 1,
        "m": 100,
        "meter": 100,
        "meters": 100,
        "in": 2.54,
        "inch": 2.54,
        "inches": 2.54,
        "ft": 30.48,
        "foot": 30.48,
        "feet": 30.48,
    }
    return round(_with_unit(value, unit, factors, "length"), 2)


def _normalize_weight(value, unit):
    factors = {
        "kg": 1,
        "kilogram": 1,
        "kilograms": 1,
        "lb": 0.45359237,
        "lbs": 0.45359237,
        "pound": 0.45359237,
        "pounds": 0.45359237,
    }
    return round(_with_unit(value, unit, factors, "weight"), 2)


def _normalize_plain(value, unit, accepted_units, label):
    value = _number(value)
    if _unit_key(unit) not in accepted_units:
        raise AmbiguousValue(f"unsupported or missing {label} unit")
    return value


def _normalize_creatinine(value, unit):
    value = _number(value)
    key = _unit_key(unit).replace("μ", "µ")
    if key in {"µmol/l", "umol/l", "micromol/l"}:
        return round(value, 2)
    if key in {"mg/dl", "mgdl"}:
        return round(value * 88.4, 2)
    raise AmbiguousValue("unsupported or missing creatinine unit")


def _normalize_hba1c(value, unit):
    value = _number(value)
    key = _unit_key(unit)
    if key in {"mmol/mol", "mmolmol"}:
        return round(value, 1)
    if key in {"%", "percent", "pct"}:
        return round((value - 2.15) * 10.929, 1)
    raise AmbiguousValue("unsupported or missing HbA1c unit")


def _canonical_text(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def _normalize_category(field, value):
    key = _canonical_text(value)
    maps = {
        "sex": {
            "female": "female",
            "woman": "female",
            "male": "male",
            "man": "male",
        },
        "smoking_status": {
            "never": "never",
            "never smoked": "never",
            "former": "former",
            "former smoker": "former",
            "previous": "former",
            "current": "current",
            "current smoker": "current",
        },
        "alcohol_frequency": {
            "never": "never",
            "special occasions only": "special_occasions",
            "special occasions": "special_occasions",
            "monthly": "one_to_three_per_month",
            "1 3 times a month": "one_to_three_per_month",
            "once or twice a week": "one_to_two_per_week",
            "1 2 times a week": "one_to_two_per_week",
            "three or four times a week": "three_to_four_per_week",
            "3 4 times a week": "three_to_four_per_week",
            "daily": "daily_or_almost_daily",
            "daily or almost daily": "daily_or_almost_daily",
        },
        "affected_relative": {
            "yes": True,
            "true": True,
            "affected": True,
            "no": False,
            "false": False,
            "none": False,
        },
    }
    if key not in maps[field]:
        raise AmbiguousValue(f"unsupported {field} category")
    return maps[field][key]


def _normalize(field, value, unit):
    if field in {"sex", "smoking_status", "alcohol_frequency", "affected_relative"}:
        return _normalize_category(field, value)
    if field in {"height", "waist", "hip"}:
        return _normalize_length(value, unit)
    if field == "weight":
        return _normalize_weight(value, unit)
    if field == "age":
        value = _normalize_plain(value, unit, {"", "year", "years", "yr", "yrs"}, "age")
        if not value.is_integer():
            raise AmbiguousValue("age must be expressed in whole years")
        return int(value)
    if field == "bmi":
        return round(
            _normalize_plain(
                value,
                unit,
                {"", "kg/m2", "kg/m²", "kgm2", "kgm²"},
                "BMI",
            ),
            1,
        )
    if field in {"sbp", "dbp"}:
        return round(
            _normalize_plain(value, unit, {"", "mmhg", "mmhg."}, "blood pressure")
        )
    if field == "creatinine":
        return _normalize_creatinine(value, unit)
    if field == "hba1c":
        return _normalize_hba1c(value, unit)
    raise AmbiguousValue("unsupported field")


def _status_for_value(field, value):
    if field not in INPUT_BOUNDS:
        return "valid"
    lower, upper = INPUT_BOUNDS[field]
    if value < lower or value > upper:
        raise OutsideInputRange(f"{field} falls outside accepted input bounds")
    support_lower, support_upper = REFERENCE_SUPPORT[field]
    if value < support_lower or value > support_upper:
        return "outside_reference_support"
    return "valid"


def _derived_status(field, value):
    try:
        return _status_for_value(field, value), None
    except OutsideInputRange as error:
        return "out_of_range", str(error)


def _base_feature(field, candidate):
    label, domain, unit = FIELD_META.get(
        field, (field.replace("_", " ").title(), "unsupported", None)
    )
    return {
        "label": label,
        "domain": domain,
        "status": "unsupported" if field not in FIELD_META else "ambiguous",
        "original_value": candidate.get("raw_value"),
        "original_unit": candidate.get("raw_unit"),
        "normalized_value": None,
        "normalized_unit": None,
        "expected_unit": unit,
        "source_text": candidate.get("source_text", ""),
        "operation": candidate.get("operation", "set"),
        "source_history": [
            {
                "source_text": candidate.get("source_text", ""),
                "original_value": candidate.get("raw_value"),
                "original_unit": candidate.get("raw_unit"),
                "operation": candidate.get("operation", "set"),
            }
        ],
    }


def _candidate_feature(field, candidate):
    feature = _base_feature(field, candidate)
    if field not in FIELD_META:
        return feature
    try:
        supplied_unit = candidate.get("raw_unit")
        if (
            supplied_unit
            and not candidate.get("_unit_from_existing")
            and not candidate.get("_unit_from_schema")
            and not _unit_is_explicit(
                field, supplied_unit, candidate.get("source_text", "")
            )
        ):
            raise AmbiguousValue("measurement unit is not explicit in the source text")
        if (
            field in {"height", "weight", "waist", "hip", "creatinine", "hba1c"}
            and not supplied_unit
            and not candidate.get("_unit_from_existing")
        ):
            raise AmbiguousValue("measurement unit is missing from the source text")
        value = _normalize(field, candidate.get("raw_value"), candidate.get("raw_unit"))
        feature["normalized_value"] = value
        feature["normalized_unit"] = FIELD_META[field][2]
        feature["status"] = _status_for_value(field, value)
    except AmbiguousValue as error:
        feature["status"] = "ambiguous"
        feature["message"] = str(error)
    except OutsideInputRange as error:
        feature["status"] = "out_of_range"
        feature["message"] = str(error)
    return feature


def _same_value(left, right):
    return (
        left.get("normalized_value") == right.get("normalized_value")
        and left.get("normalized_unit") == right.get("normalized_unit")
        and left.get("status") == right.get("status")
    )


def _merge_feature(current, incoming):
    operation = incoming.get("operation", "set")
    if current is None:
        return incoming
    if operation == "correct":
        incoming["source_history"] = (
            current.get("source_history", []) + incoming.get("source_history", [])
        )
        return incoming
    if _same_value(current, incoming):
        return current
    alternatives = list(current.get("alternatives", []))
    if not alternatives:
        alternatives.append(current.get("normalized_value", current.get("original_value")))
    value = incoming.get("normalized_value", incoming.get("original_value"))
    if value not in alternatives:
        alternatives.append(value)
    merged = deepcopy(current)
    merged["status"] = "conflicting"
    merged["alternatives"] = alternatives
    merged["source_texts"] = list(
        dict.fromkeys(
            current.get("source_texts", [current.get("source_text", "")])
            + [incoming.get("source_text", "")]
        )
    )
    merged["source_history"] = (
        current.get("source_history", []) + incoming.get("source_history", [])
    )
    merged["message"] = "Conflicting values require an explicit correction."
    return merged


def _expand_blood_pressure(candidate):
    raw = str(candidate.get("raw_value", ""))
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*", raw)
    if not match:
        return None
    expanded = []
    for field, value in (("sbp", match.group(1)), ("dbp", match.group(2))):
        part = deepcopy(candidate)
        part["field"] = field
        part["raw_value"] = float(value)
        part["raw_unit"] = candidate.get("raw_unit") or "mmHg"
        if candidate.get("raw_unit") is None:
            part["_unit_from_schema"] = True
        expanded.append(part)
    return expanded


def _missing_fields(reported, derived):
    missing = [field for field in FIELD_META if field not in reported]
    if "bmi" in derived and "bmi" in missing:
        missing.remove("bmi")
    return missing


def build_profile_draft(candidates):
    reported = {}
    for original in candidates:
        candidate = deepcopy(original)
        raw_field = re.sub(r"[^a-z0-9]+", "_", str(candidate.get("field", "")).lower()).strip("_")
        field = FIELD_ALIASES.get(raw_field, raw_field or "unknown")
        operation = candidate.get("operation", "set")
        if operation not in {"set", "correct", "remove"}:
            feature = _base_feature(field, candidate)
            feature.update(
                status="ambiguous",
                message="Feature Candidate operation is invalid.",
            )
            reported[field] = _merge_feature(reported.get(field), feature)
            continue
        candidate["operation"] = operation

        if field == "blood_pressure":
            expanded = _expand_blood_pressure(candidate)
            if expanded is None:
                feature = _base_feature(field, candidate)
                feature.update(
                    status="ambiguous",
                    message="Blood pressure must contain visible SBP/DBP values.",
                )
                reported[field] = _merge_feature(reported.get(field), feature)
                continue
            candidates_to_apply = expanded
        else:
            candidate["field"] = field
            candidates_to_apply = [candidate]

        for item in candidates_to_apply:
            item_field = item["field"]
            if operation == "remove":
                reported.pop(item_field, None)
                continue
            current = reported.get(item_field)
            if (
                operation == "correct"
                and item.get("raw_unit") is None
                and current
                and current.get("normalized_unit")
            ):
                item["raw_unit"] = current["normalized_unit"]
                item["_unit_from_existing"] = True
            incoming = _candidate_feature(item_field, item)
            reported[item_field] = _merge_feature(current, incoming)

    derived = {}
    height = reported.get("height")
    weight = reported.get("weight")
    if (
        height
        and weight
        and height["status"] in USABLE_STATUSES
        and weight["status"] in USABLE_STATUSES
    ):
        bmi = weight["normalized_value"] / (height["normalized_value"] / 100) ** 2
        bmi = round(bmi, 1)
        bmi_status, bmi_message = _derived_status("bmi", bmi)
        derived["bmi"] = {
            "label": "BMI",
            "value": bmi,
            "unit": "kg/m²",
            "domain": "body_composition",
            "status": bmi_status,
            "derived_from": ["height", "weight"],
        }
        if bmi_message:
            derived["bmi"]["message"] = bmi_message
        reported_bmi = reported.get("bmi")
        if (
            reported_bmi
            and reported_bmi["status"] in USABLE_STATUSES
            and abs(reported_bmi["normalized_value"] - bmi) > 0.1
        ):
            reported_bmi["calculated_value"] = bmi
            if reported_bmi.get("operation") == "correct":
                reported_bmi["mismatch_acknowledged"] = True
                reported_bmi["message"] = (
                    "Reported BMI was explicitly confirmed despite differing from "
                    "the calculated BMI."
                )
            else:
                reported_bmi["status"] = "conflicting"
                reported_bmi["message"] = (
                    "Reported BMI differs from BMI calculated from height and weight."
                )

    waist = reported.get("waist")
    hip = reported.get("hip")
    if (
        waist
        and hip
        and waist["status"] in USABLE_STATUSES
        and hip["status"] in USABLE_STATUSES
        and hip["normalized_value"] > 0
    ):
        derived["waist_to_hip_ratio"] = {
            "label": "Waist-to-hip ratio",
            "value": round(waist["normalized_value"] / hip["normalized_value"], 2),
            "unit": "ratio",
            "domain": "body_composition",
            "status": "valid",
            "derived_from": ["waist", "hip"],
        }

    statuses = {feature["status"] for feature in reported.values()} | {
        feature["status"] for feature in derived.values()
    }
    usable_count = sum(
        feature["status"] in USABLE_STATUSES for feature in reported.values()
    )
    return {
        "state": "draft",
        "candidates": deepcopy(candidates),
        "reported_features": reported,
        "derived_features": derived,
        "missing_fields": _missing_fields(reported, derived),
        "can_confirm": usable_count > 0 and not (statuses & BLOCKING_STATUSES),
    }


def confirm_profile(draft):
    validated = build_profile_draft(draft.get("candidates", []))
    if not validated["can_confirm"]:
        raise ValueError("Profile Draft is not eligible for confirmation")
    validated["state"] = "confirmed"
    validated["matching_started"] = False
    return validated
