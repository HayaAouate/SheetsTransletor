"""Access control: one shared API key, sent by the caller in the `X-API-Key` header."""
import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from . import settings

# auto_error=False: we answer 401 ourselves; declaring the scheme also gives /docs its "Authorize" button.
_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(_header)) -> None:
    # compare_digest is constant time: the response delay does not leak how many characters matched
    if not key or not secrets.compare_digest(key.encode(), settings.API_KEY.encode()):
        raise HTTPException(401, "Clé d'API absente ou invalide (en-tête X-API-Key).")
