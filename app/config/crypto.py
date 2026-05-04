"""API Key encryption using a machine-derived Fernet key."""
from __future__ import annotations
import base64
import hashlib
import platform

from cryptography.fernet import Fernet, InvalidToken


def _machine_id() -> str:
    """Return a stable machine identifier. Uses Windows Machine GUID on Windows."""
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Cryptography",
            )
            guid, _ = winreg.QueryValueEx(key, "MachineGuid")
            winreg.CloseKey(key)
            return str(guid)
        except Exception:
            pass
    return f"{platform.node()}-{platform.machine()}-DocDiffAgent"


def _derive_key() -> bytes:
    digest = hashlib.sha256(_machine_id().encode()).digest()
    return base64.urlsafe_b64encode(digest)


_fernet = Fernet(_derive_key())


def encrypt(plaintext: str) -> str:
    """Encrypt plaintext string, return base64 ciphertext string."""
    return _fernet.encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    """Decrypt ciphertext string back to plaintext. Returns empty string on failure."""
    try:
        return _fernet.decrypt(ciphertext.encode()).decode()
    except (InvalidToken, Exception):
        return ""
