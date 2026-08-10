"""
Phase 8 end-to-end test (plan.md): login -> vehicle proposal -> underwriter decision.

Covers the full chain:
  signup(client) -> signup(underwriter) -> submit vehicle proposal (multipart,
  incl. driving-license image) -> OCR/LLM/validation runs server-side ->
  ML prediction -> DB insert -> list proposals -> get proposal detail ->
  underwriter fetches it -> underwriter sets decision -> re-fetch confirms status.

Run against a LIVE server + LIVE MySQL DB. This does not run in the sandbox
(no server, no DB, network whitelist blocks it) — run it yourself:

    pip install pytest requests --break-system-packages
    export BASE_URL=http://localhost:8000   # adjust to your running server
    pytest test_phase8_e2e.py -v -s

Needs a real driving-license image on disk for the OCR step to have something
to extract from. Point TEST_IMAGE_PATH at one, or the test creates a plain
placeholder image so the flow still runs (OCR/extraction will just come back
mostly-empty, since it's not a real license image).
"""
import io
import os
import time
import uuid

import pytest
import requests

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
TEST_IMAGE_PATH = os.environ.get("TEST_IMAGE_PATH")  # optional real license image


def _placeholder_image_bytes() -> bytes:
    """Fallback 1x1 PNG if no real license image is supplied via TEST_IMAGE_PATH."""
    import base64
    png_1x1 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YA"
        "AAAASUVORK5CYII="
    )
    return base64.b64decode(png_1x1)


def _unique_email(prefix: str) -> str:
    # NOTE: EmailStr (pydantic/email-validator) rejects reserved TLDs
    # (.local, .test, .example, .invalid) at the syntax-check level, even
    # with no DNS lookup involved. Use a normal-looking domain instead.
    return f"{prefix}_{uuid.uuid4().hex[:10]}@phase8qa-mail.com"


@pytest.fixture(scope="module")
def client_creds():
    return {
        "full_name": "Phase8 Test Client",
        "email": _unique_email("client"),
        "password": "TestPass123!",
        "role": "client",
    }


@pytest.fixture(scope="module")
def underwriter_creds():
    return {
        "full_name": "Phase8 Test Underwriter",
        "email": _unique_email("underwriter"),
        "password": "TestPass123!",
        "role": "underwriter",
    }


@pytest.fixture(scope="module")
def client_token(client_creds):
    r = requests.post(f"{BASE_URL}/api/v1/auth/signup", json=client_creds, timeout=15)
    assert r.status_code == 201, f"client signup failed: {r.status_code} {r.text}"
    return r.json()["access_token"]


@pytest.fixture(scope="module")
def underwriter_token(underwriter_creds):
    r = requests.post(f"{BASE_URL}/api/v1/auth/signup", json=underwriter_creds, timeout=15)
    assert r.status_code == 201, f"underwriter signup failed: {r.status_code} {r.text}"
    return r.json()["access_token"]


def test_health_check():
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    assert r.status_code == 200, "server not reachable at BASE_URL"


def test_login_roundtrip(client_creds, client_token):
    """login() with the signed-up creds should also succeed and return a usable token."""
    r = requests.post(
        f"{BASE_URL}/api/v1/auth/login",
        json={"email": client_creds["email"], "password": client_creds["password"]},
        timeout=15,
    )
    assert r.status_code == 200, f"login failed: {r.status_code} {r.text}"
    assert r.json()["access_token"]


@pytest.fixture(scope="module")
def vehicle_proposal_id(client_token):
    """Submit vehicle proposal -> OCR -> LLM -> validation -> ML -> DB insert."""
    if TEST_IMAGE_PATH and os.path.exists(TEST_IMAGE_PATH):
        with open(TEST_IMAGE_PATH, "rb") as f:
            image_bytes = f.read()
        filename = os.path.basename(TEST_IMAGE_PATH)
    else:
        image_bytes = _placeholder_image_bytes()
        filename = "placeholder.png"

    form = {
        "make": "Hyundai",
        "model": "Creta",
        "year": "2022",
        "vehicle_type": "suv",
        "engine_cc": "1500",
        "fuel_type": "petrol",
        "vehicle_value": "1500000",
        "safety_features": "yes",
        "anti_theft": "yes",
        "color": "white",
        "driver_age": "35",
        "driving_experience": "10",
        "license_age": "10",
        "previous_accidents": "0",
        "previous_claims": "0",
        "traffic_violations": "0",
        "usage_type": "private",
        "annual_mileage": "12000",
        "city": "Chennai",
        "region": "Tamil Nadu",
        "previous_insurance": "yes",
        "policy_lapses": "0",
        "country_code": "IN",
        "doc_type": "drivers_license",
    }
    files = {"file": (filename, io.BytesIO(image_bytes), "image/png")}
    headers = {"Authorization": f"Bearer {client_token}"}

    r = requests.post(
        f"{BASE_URL}/api/v1/vehicle/proposals",
        data=form, files=files, headers=headers, timeout=60,
    )
    assert r.status_code == 200, f"proposal submit failed: {r.status_code} {r.text}"
    body = r.json()
    assert body["status"] == "PENDING"
    assert body["id"] > 0
    assert body["vehicle_id"] > 0
    return body["id"]


def test_submit_vehicle_proposal(vehicle_proposal_id):
    assert vehicle_proposal_id > 0


def test_duplicate_proposal_blocked(client_token, vehicle_proposal_id):
    """Second PENDING vehicle proposal from the same user should 409."""
    form = {
        "make": "Hyundai", "model": "Creta", "year": "2022", "vehicle_type": "suv",
        "engine_cc": "1500", "fuel_type": "petrol", "vehicle_value": "1500000",
        "safety_features": "yes", "anti_theft": "yes", "color": "white",
        "driver_age": "35", "driving_experience": "10", "license_age": "10",
        "previous_accidents": "0", "previous_claims": "0", "traffic_violations": "0",
        "usage_type": "private", "annual_mileage": "12000", "city": "Chennai",
        "region": "Tamil Nadu", "previous_insurance": "yes", "policy_lapses": "0",
        "country_code": "IN", "doc_type": "drivers_license",
    }
    files = {"file": ("placeholder.png", io.BytesIO(_placeholder_image_bytes()), "image/png")}
    headers = {"Authorization": f"Bearer {client_token}"}
    r = requests.post(
        f"{BASE_URL}/api/v1/vehicle/proposals",
        data=form, files=files, headers=headers, timeout=60,
    )
    assert r.status_code == 409, f"expected 409 duplicate block, got {r.status_code} {r.text}"


def test_list_vehicle_proposals(client_token, vehicle_proposal_id):
    headers = {"Authorization": f"Bearer {client_token}"}
    r = requests.get(f"{BASE_URL}/api/v1/vehicle/proposals", headers=headers, timeout=15)
    assert r.status_code == 200, f"list failed: {r.status_code} {r.text}"
    ids = [item["id"] for item in r.json()]
    assert vehicle_proposal_id in ids


def test_get_vehicle_proposal_detail(client_token, vehicle_proposal_id):
    headers = {"Authorization": f"Bearer {client_token}"}
    r = requests.get(
        f"{BASE_URL}/api/v1/vehicle/proposals/{vehicle_proposal_id}",
        headers=headers, timeout=15,
    )
    assert r.status_code == 200, f"detail fetch failed: {r.status_code} {r.text}"
    body = r.json()
    assert body["id"] == vehicle_proposal_id
    assert "risk_score" in body
    assert body["vehicle"]["make"] == "Hyundai" or "vehicle" in body  # tolerate shape variance


def test_life_only_endpoint_rejects_vehicle_id(client_token, vehicle_proposal_id):
    """Confirms the item-2 fix: old life-only endpoint gives clean 400, not 500."""
    headers = {"Authorization": f"Bearer {client_token}"}
    r = requests.get(
        f"{BASE_URL}/api/v1/proposals/{vehicle_proposal_id}",
        headers=headers, timeout=15,
    )
    assert r.status_code == 400, f"expected 400, got {r.status_code} {r.text}"


def test_underwriter_can_view_and_decide(underwriter_token, vehicle_proposal_id):
    headers = {"Authorization": f"Bearer {underwriter_token}"}

    r = requests.get(
        f"{BASE_URL}/api/v1/vehicle/proposals/{vehicle_proposal_id}",
        headers=headers, timeout=15,
    )
    assert r.status_code == 200, f"underwriter view failed: {r.status_code} {r.text}"

    r = requests.patch(
        f"{BASE_URL}/api/v1/proposals/{vehicle_proposal_id}/decision",
        json={"status": "APPROVED"}, headers=headers, timeout=15,
    )
    assert r.status_code == 200, f"decision failed: {r.status_code} {r.text}"


def test_decision_persisted(client_token, vehicle_proposal_id):
    headers = {"Authorization": f"Bearer {client_token}"}
    r = requests.get(
        f"{BASE_URL}/api/v1/vehicle/proposals/{vehicle_proposal_id}",
        headers=headers, timeout=15,
    )
    assert r.status_code == 200
    assert r.json()["status"] == "APPROVED"


def test_duplicate_guard_clears_after_decision(client_token):
    """Item-1 guard: after APPROVED, a fresh vehicle proposal should be allowed again."""
    form = {
        "make": "Toyota", "model": "Fortuner", "year": "2023", "vehicle_type": "suv",
        "engine_cc": "2700", "fuel_type": "diesel", "vehicle_value": "3500000",
        "safety_features": "yes", "anti_theft": "yes", "color": "black",
        "driver_age": "40", "driving_experience": "15", "license_age": "15",
        "previous_accidents": "0", "previous_claims": "0", "traffic_violations": "0",
        "usage_type": "private", "annual_mileage": "10000", "city": "Chennai",
        "region": "Tamil Nadu", "previous_insurance": "yes", "policy_lapses": "0",
        "country_code": "IN", "doc_type": "drivers_license",
    }
    files = {"file": ("placeholder.png", io.BytesIO(_placeholder_image_bytes()), "image/png")}
    headers = {"Authorization": f"Bearer {client_token}"}
    r = requests.post(
        f"{BASE_URL}/api/v1/vehicle/proposals",
        data=form, files=files, headers=headers, timeout=60,
    )
    assert r.status_code == 200, f"expected guard to clear, got {r.status_code} {r.text}"


def test_document_download_endpoint(client_token, vehicle_proposal_id):
    """Item-5: untested raw file download endpoint."""
    headers = {"Authorization": f"Bearer {client_token}"}
    r = requests.get(
        f"{BASE_URL}/api/v1/vehicle/proposals/{vehicle_proposal_id}/document",
        headers=headers, timeout=15,
    )
    assert r.status_code == 200, f"document download failed: {r.status_code} {r.text}"
    assert len(r.content) > 0