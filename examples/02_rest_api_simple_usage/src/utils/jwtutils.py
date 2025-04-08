from datetime import timedelta, timezone, datetime
import jwt
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.hazmat.primitives.serialization import PublicFormat
from cryptography.hazmat.backends import default_backend
import base64
from getpass import getpass
import hashlib


def generate_JWT_token(private_key_path: str, account: str, user: str) -> str:
    """
    Generate a valid JWT token from snowflake account name, user name, private key and private key passphrase.
    """

    # Prompt for private key passphrase
    def get_private_key_passphrase():
        return getpass("Private Key Passphrase: ")

    # Generate encoded private key
    with open(private_key_path, "rb") as pem_in:
        pemlines = pem_in.read()
        try:
            private_key = load_pem_private_key(pemlines, None, default_backend())
        except TypeError:
            private_key = load_pem_private_key(
                pemlines, get_private_key_passphrase().encode(), default_backend()
            )
    public_key_raw = private_key.public_key().public_bytes(
        Encoding.DER, PublicFormat.SubjectPublicKeyInfo
    )
    sha256hash = hashlib.sha256()
    sha256hash.update(public_key_raw)
    public_key_fp = "SHA256:" + base64.b64encode(sha256hash.digest()).decode("utf-8")

    # Generate JWT payload
    qualified_username = account + "." + user
    now = datetime.now(timezone.utc)
    lifetime = timedelta(minutes=60)
    payload = {
        "iss": qualified_username + "." + public_key_fp,
        "sub": qualified_username,
        "iat": now,
        "exp": now + lifetime,
    }

    # Return the encoded JWT token
    return jwt.encode(payload, key=private_key, algorithm="RS256")
