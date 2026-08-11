"""
Fleet-model e2e test for the vehicle module.

Covers the NEW behavior after the fleet rewrite:
  vehicle #1 (needs a license file) creates a NEW proposal
    -> vehicle #2 (no file needed) attaches to the SAME open proposal
    -> proposal detail shows BOTH vehicles, each with its own risk score
    -> underwriter approves vehicle #1, rejects vehicle #2 (per-vehicle decision)
    -> once every vehicle has a decision, proposal auto-closes
    -> a NEW vehicle submission after that starts a brand-new proposal
       (requires a file again, since the old one is closed)

Run against a LIVE server + LIVE MySQL DB — this does not run in the sandbox.

    pip install pytest requests --break-system-packages
    $env:BASE_URL="http://localhost:8000"
    python -m pytest test_fleet_e2e.py -v -s
"""
import io
import os
import uuid

import pytest
import requests

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")


def _placeholder_image_bytes() -> bytes:
    import base64
    png_1x1 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YA"
        "AAAASUVORK5CYII="
    )
    return base64.b64decode(png_1x1)


def _unique_email(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}@fleetqa-mail.com"


def _vehicle_form(make="Hyundai", model="Creta", value="1500000"):
    return {
        "make": make, "model": model, "year": "2022", "vehicle_type": "suv",
        "engine_cc": "1500", "fuel_type": "petrol", "vehicle_value": value,
        "safety_features": "yes", "anti_theft": "yes", "color": "white",
        "driver_age": "35", "driving_experience": "10", "license_age": "10",
        "previous_accidents": "0", "previous_claims": "0", "traffic_violations": "0",
        "usage_type": "private", "annual_mileage": "12000", "city": "Chennai",
        "region": "Tamil Nadu", "previous_insurance": "yes", "policy_lapses": "0",
        "country_code": "IN", "doc_type": "drivers_license",
    }


@pytest.fixture(scope="module")
def client_token():
    creds = {
        "full_name": "Fleet Test Client", "email": _unique_email("client"),
        "password": "TestPass123!", "role": "client",
    }
    r = requests.post(f"{BASE_URL}/api/v1/auth/signup", json=creds, timeout=15)
    assert r.status_code == 201, f"client signup failed: {r.status_code} {r.text}"
    return r.json()["access_token"]


@pytest.fixture(scope="module")
def underwriter_token():
    creds = {
        "full_name": "Fleet Test Underwriter", "email": _unique_email("underwriter"),
        "password": "TestPass123!", "role": "underwriter",
    }
    r = requests.post(f"{BASE_URL}/api/v1/auth/signup", json=creds, timeout=15)
    assert r.status_code == 201, f"underwriter signup failed: {r.status_code} {r.text}"
    return r.json()["access_token"]


def test_health_check():
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    assert r.status_code == 200, "server not reachable at BASE_URL"


@pytest.fixture(scope="module")
def proposal_and_vehicles(client_token):
    headers = {"Authorization": f"Bearer {client_token}"}

    # Vehicle #1: no open proposal exists yet -> file REQUIRED -> creates new proposal
    files = {"file": ("license.png", io.BytesIO(_placeholder_image_bytes()), "image/png")}
    r1 = requests.post(
        f"{BASE_URL}/api/v1/vehicle/proposals",
        data=_vehicle_form(make="Hyundai", model="Creta", value="1500000"),
        files=files, headers=headers, timeout=60,
    )
    assert r1.status_code == 200, f"vehicle 1 submit failed: {r1.status_code} {r1.text}"
    body1 = r1.json()
    proposal_id = body1["id"]
    vehicle1_id = body1["vehicle_id"]

    # Vehicle #2: same user has an OPEN proposal now -> file NOT required -> attaches to it
    r2 = requests.post(
        f"{BASE_URL}/api/v1/vehicle/proposals",
        data=_vehicle_form(make="Toyota", model="Fortuner", value="3500000"),
        headers=headers, timeout=60,
    )
    assert r2.status_code == 200, f"vehicle 2 submit failed: {r2.status_code} {r2.text}"
    body2 = r2.json()

    assert body2["id"] == proposal_id, "vehicle 2 should attach to the SAME open proposal"
    vehicle2_id = body2["vehicle_id"]
    assert vehicle2_id != vehicle1_id

    return proposal_id, vehicle1_id, vehicle2_id


def test_two_vehicles_share_one_proposal(proposal_and_vehicles):
    proposal_id, v1, v2 = proposal_and_vehicles
    assert proposal_id and v1 and v2


def test_second_vehicle_did_not_need_a_file(proposal_and_vehicles):
    """Just confirms the fixture above didn't fail — file omission on vehicle 2
    is the core fleet-model behavior being tested."""
    pass


def test_proposal_detail_shows_both_vehicles(client_token, proposal_and_vehicles):
    proposal_id, v1, v2 = proposal_and_vehicles
    headers = {"Authorization": f"Bearer {client_token}"}
    r = requests.get(f"{BASE_URL}/api/v1/vehicle/proposals/{proposal_id}", headers=headers, timeout=15)
    assert r.status_code == 200, f"detail fetch failed: {r.status_code} {r.text}"
    body = r.json()
    vehicle_ids = [v["id"] for v in body["vehicles"]]
    assert v1 in vehicle_ids and v2 in vehicle_ids
    assert len(body["vehicles"]) == 2
    for v in body["vehicles"]:
        assert v["status"] == "PENDING"
        assert v["risk_score"] is not None


def test_list_shows_fleet_aggregates(client_token, proposal_and_vehicles):
    proposal_id, _, _ = proposal_and_vehicles
    headers = {"Authorization": f"Bearer {client_token}"}
    r = requests.get(f"{BASE_URL}/api/v1/vehicle/proposals", headers=headers, timeout=15)
    assert r.status_code == 200
    match = next(p for p in r.json() if p["id"] == proposal_id)
    assert match["vehicle_count"] == 2
    assert match["total_value"] == 1500000 + 3500000
    assert match["flagged_count"] == 2  # both still PENDING


def test_underwriter_decides_each_vehicle_separately(underwriter_token, proposal_and_vehicles):
    proposal_id, v1, v2 = proposal_and_vehicles
    headers = {"Authorization": f"Bearer {underwriter_token}"}

    r = requests.patch(
        f"{BASE_URL}/api/v1/vehicle/vehicles/{v1}/decision",
        json={"status": "APPROVED"}, headers=headers, timeout=15,
    )
    assert r.status_code == 200, f"decision v1 failed: {r.status_code} {r.text}"

    r = requests.patch(
        f"{BASE_URL}/api/v1/vehicle/vehicles/{v2}/decision",
        json={"status": "REJECTED"}, headers=headers, timeout=15,
    )
    assert r.status_code == 200, f"decision v2 failed: {r.status_code} {r.text}"


def test_proposal_auto_closed_after_all_decided(client_token, proposal_and_vehicles):
    proposal_id, v1, v2 = proposal_and_vehicles
    headers = {"Authorization": f"Bearer {client_token}"}
    r = requests.get(f"{BASE_URL}/api/v1/vehicle/proposals/{proposal_id}", headers=headers, timeout=15)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "CLOSED", f"expected proposal auto-closed, got {body['status']}"
    statuses = {v["id"]: v["status"] for v in body["vehicles"]}
    assert statuses[v1] == "APPROVED"
    assert statuses[v2] == "REJECTED"


def test_new_vehicle_after_close_starts_a_new_proposal(client_token, proposal_and_vehicles):
    old_proposal_id, _, _ = proposal_and_vehicles
    headers = {"Authorization": f"Bearer {client_token}"}

    # Old proposal is CLOSED now, so this should require a file again
    r_no_file = requests.post(
        f"{BASE_URL}/api/v1/vehicle/proposals",
        data=_vehicle_form(make="Honda", model="City", value="1200000"),
        headers=headers, timeout=60,
    )
    assert r_no_file.status_code == 422, "expected file required for a fresh proposal"

    files = {"file": ("license2.png", io.BytesIO(_placeholder_image_bytes()), "image/png")}
    r = requests.post(
        f"{BASE_URL}/api/v1/vehicle/proposals",
        data=_vehicle_form(make="Honda", model="City", value="1200000"),
        files=files, headers=headers, timeout=60,
    )
    assert r.status_code == 200, f"new proposal submit failed: {r.status_code} {r.text}"
    new_proposal_id = r.json()["id"]
    assert new_proposal_id != old_proposal_id, "should be a brand-new proposal, not reused"


def test_old_shared_decision_endpoint_rejects_vehicle_proposal(underwriter_token, proposal_and_vehicles):
    """Confirms main.py's old life-only decision endpoint won't silently
    mishandle a vehicle proposal anymore."""
    proposal_id, _, _ = proposal_and_vehicles
    headers = {"Authorization": f"Bearer {underwriter_token}"}
    r = requests.patch(
        f"{BASE_URL}/api/v1/proposals/{proposal_id}/decision",
        json={"status": "APPROVED"}, headers=headers, timeout=15,
    )
    assert r.status_code == 400, f"expected 400, got {r.status_code} {r.text}"