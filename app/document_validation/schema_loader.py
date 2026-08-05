import json
import os
from functools import lru_cache

SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "id_schemas")


class SchemaNotFoundError(Exception):
    pass


@lru_cache(maxsize=32)
def load_schema(country_code: str, doc_type: str) -> dict:
    filename = f"{country_code.lower()}_{doc_type.lower()}.json"
    path = os.path.join(SCHEMA_DIR, filename)
    if not os.path.exists(path):
        raise SchemaNotFoundError(
            f"No schema for country={country_code}, doc_type={doc_type}. Expected file: {filename}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_supported_docs() -> list:
    result = []
    for fname in os.listdir(SCHEMA_DIR):
        if fname.endswith(".json"):
            with open(os.path.join(SCHEMA_DIR, fname), "r", encoding="utf-8") as f:
                cfg = json.load(f)
                result.append({
                    "country_code": cfg["country_code"],
                    "doc_type": cfg["doc_type"],
                    "doc_name": cfg["doc_name"]
                })
    return result

def list_countries() -> list:
    """Distinct countries only — for dropdown 1."""
    seen = {}
    for fname in os.listdir(SCHEMA_DIR):
        if fname.endswith(".json"):
            with open(os.path.join(SCHEMA_DIR, fname), "r", encoding="utf-8") as f:
                cfg = json.load(f)
                seen[cfg["country_code"]] = cfg["country_code"]
    return sorted(seen.keys())


def list_doc_types_for_country(country_code: str) -> list:
    """Doc types available for one country — for dropdown 2, after country picked."""
    result = []
    for fname in os.listdir(SCHEMA_DIR):
        if fname.startswith(f"{country_code.lower()}_") and fname.endswith(".json"):
            with open(os.path.join(SCHEMA_DIR, fname), "r", encoding="utf-8") as f:
                cfg = json.load(f)
                result.append({"doc_type": cfg["doc_type"], "doc_name": cfg["doc_name"]})
    return result