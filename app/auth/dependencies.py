from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from .security import decode_access_token
from .schemas import CurrentUser
from ..db import get_connection

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def get_current_user(token: str = Depends(oauth2_scheme)) -> CurrentUser:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    user_id = payload.get("sub")
    if user_id is None:
        raise credentials_exception

    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id, full_name, email, role FROM users WHERE id=%s", (user_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row is None:
        raise credentials_exception

    return CurrentUser(**row)


def require_role(*allowed_roles: str):
    """
    Usage: Depends(require_role("underwriter"))
    Raises 403 if current user's role isn't in allowed_roles.
    """
    def _check(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This action requires role: {' or '.join(allowed_roles)}",
            )
        return current_user
    return _check