"""Encryption at rest: AES-256-GCM with a random 96-bit nonce per record and the record id as associated data (a
ciphertext moved to another row does not decrypt). The key comes from NOTEWELL_DATA_KEY; for the local demo a key file
is generated in the data directory. In production the key belongs in a KMS/HSM, not on the same disk."""

from __future__ import annotations

import base64
import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_VERSION = b"\x01"


class Cipher:
    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            raise ValueError("the data key must be 32 bytes (AES-256)")
        self._aead = AESGCM(key)

    @classmethod
    def from_settings(cls, data_key: str, data_dir: Path) -> Cipher:
        if data_key:
            return cls(base64.b64decode(data_key))
        path = data_dir / "data.key"
        if not path.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(base64.b64encode(AESGCM.generate_key(bit_length=256)))
            path.chmod(0o600)
        return cls(base64.b64decode(path.read_bytes()))

    def encrypt(self, plaintext: bytes, aad: str) -> bytes:
        nonce = os.urandom(12)
        return _VERSION + nonce + self._aead.encrypt(nonce, plaintext, aad.encode("utf-8"))

    def decrypt(self, blob: bytes, aad: str) -> bytes:
        if blob[:1] != _VERSION:
            raise ValueError("unknown ciphertext version")
        return self._aead.decrypt(blob[1:13], blob[13:], aad.encode("utf-8"))
