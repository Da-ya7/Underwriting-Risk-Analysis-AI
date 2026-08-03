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