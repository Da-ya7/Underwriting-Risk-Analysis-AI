from fastapi import APIRouter, HTTPException, Depends
from mysql.connector import Error as MySQLError

from .schemas import SignupRequest, LoginRequest, TokenResponse, CurrentUser
from .security import hash_password, verify_password, create_access_token
from .dependencies import get_current_user
from ..db import get_connection

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

ALLOWED_ROLES = {"client", "underwriter"}


@router.post("/signup", response_model=TokenResponse, status_code=201)
def signup(payload: SignupRequest):
    if payload.role not in ALLOWED_ROLES:
        raise HTTPException(status_code=422, detail=f"role must be one of {ALLOWED_ROLES}")

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT id FROM users WHERE email=%s", (payload.email,))
    if cur.fetchone():
        cur.close()
        conn.close()
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    hashed = hash_password(payload.password)

    try:
        cur.execute(
            "INSERT INTO users (full_name, email, password_hash, role) VALUES (%s,%s,%s,%s)",
            (payload.full_name, payload.email, hashed, payload.role),
        )
        conn.commit()
        new_id = cur.lastrowid
    except MySQLError as e:
        conn.rollback()
        cur.close()
        conn.close()
        raise HTTPException(status_code=500, detail=f"Could not create account: {e}")

    cur.close()
    conn.close()

    token = create_access_token({"sub": str(new_id)})
    return TokenResponse(access_token=token, role=payload.role, full_name=payload.full_name)


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest):
    if payload.role not in ALLOWED_ROLES:
        raise HTTPException(status_code=422, detail=f"role must be one of {ALLOWED_ROLES}")

    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id, full_name, password_hash, role FROM users WHERE email=%s", (payload.email,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    # Same error for "no such email" and "wrong password" — do NOT reveal
    # which one it was, prevents attackers from probing which emails exist.
    invalid = HTTPException(status_code=401, detail="Incorrect email or password")

    if not row:
        raise invalid
    if not verify_password(payload.password, row["password_hash"]):
        raise invalid

    # Role-based login enforcement: the role picked on the login screen
    # (Client / Underwriter box) must match the account's actual role.
    # A client account cannot sign in through the Underwriter tab and
    # vice versa — password is already verified above, so this message
    # is safe to be explicit (the caller already proved account ownership).
    if row["role"] != payload.role:
        raise HTTPException(
            status_code=403,
            detail=f"This account is registered as '{row['role']}'. "
                   f"Please select '{row['role']}' on the login screen.",
        )

    token = create_access_token({"sub": str(row["id"])})
    return TokenResponse(access_token=token, role=row["role"], full_name=row["full_name"])


@router.get("/me", response_model=CurrentUser)
def get_me(current_user: CurrentUser = Depends(get_current_user)):
    return current_user