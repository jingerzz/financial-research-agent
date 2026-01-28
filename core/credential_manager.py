"""
Secure Credential Manager for Financial Research Agent.

Provides persistent, secure storage for API keys using:
1. OS Keyring (primary) - macOS Keychain, Windows Credential Manager, Linux Secret Service
2. Encrypted local file (fallback) - AES-256 encryption

Multi-user support: Keys are stored per-OS-user.
"""

import os
import json
import base64
import hashlib
import logging
import getpass
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

# Service name for keyring storage
KEYRING_SERVICE = "financial-research-agent"

# Supported providers and their display names
PROVIDERS = {
    "anthropic": {"name": "Anthropic", "env_var": "ANTHROPIC_API_KEY", "prefix": "sk-ant-"},
    "openai": {"name": "OpenAI", "env_var": "OPENAI_API_KEY", "prefix": "sk-"},
    "google": {"name": "Google AI", "env_var": "GOOGLE_API_KEY", "prefix": ""},
    "newsapi": {"name": "NewsAPI", "env_var": "NEWSAPI_KEY", "prefix": ""},
}


@dataclass
class StoredCredential:
    """Metadata about a stored credential."""
    provider: str
    stored_at: str
    storage_method: str  # "keyring" or "encrypted_file"
    key_preview: str  # First/last few chars for identification


class CredentialManager:
    """
    Manages secure storage and retrieval of API keys.
    
    Priority order for retrieval:
    1. Environment variables
    2. Streamlit secrets
    3. OS Keyring
    4. Encrypted local file
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the credential manager.
        
        Args:
            config_dir: Directory for encrypted fallback storage.
                       Defaults to ~/.financial-research-agent/
        """
        self.config_dir = config_dir or Path.home() / ".financial-research-agent"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure secure permissions on config directory (Unix only)
        try:
            os.chmod(self.config_dir, 0o700)
        except (OSError, AttributeError):
            pass  # Windows or permission error
        
        self._keyring_available = self._check_keyring()
        self._encryption_key: Optional[bytes] = None
        
        logger.info(f"CredentialManager initialized. Keyring available: {self._keyring_available}")
    
    def _check_keyring(self) -> bool:
        """Check if OS keyring is available and functional."""
        try:
            import keyring
            from keyring.errors import NoKeyringError, KeyringError
            
            # Try to access the keyring
            keyring.get_keyring()
            
            # Do a test write/read/delete to verify it works
            test_key = f"_test_{datetime.now().timestamp()}"
            keyring.set_password(KEYRING_SERVICE, test_key, "test")
            result = keyring.get_password(KEYRING_SERVICE, test_key)
            keyring.delete_password(KEYRING_SERVICE, test_key)
            
            return result == "test"
        except ImportError:
            logger.warning("keyring package not installed")
            return False
        except (NoKeyringError, KeyringError) as e:
            logger.warning(f"Keyring not available: {e}")
            return False
        except Exception as e:
            logger.warning(f"Keyring check failed: {e}")
            return False
    
    def _get_encryption_key(self) -> bytes:
        """
        Get or derive the encryption key for fallback storage.
        
        Uses machine-specific information + username to derive a key.
        This isn't perfect security, but provides reasonable protection
        for a local application.
        """
        if self._encryption_key:
            return self._encryption_key
        
        # Combine machine-specific info with username
        username = getpass.getuser()
        
        # Try to get machine ID (varies by OS)
        machine_id = ""
        try:
            # Linux
            machine_id_file = Path("/etc/machine-id")
            if machine_id_file.exists():
                machine_id = machine_id_file.read_text().strip()
        except Exception:
            pass
        
        if not machine_id:
            try:
                # macOS
                import subprocess
                result = subprocess.run(
                    ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                    capture_output=True, text=True
                )
                for line in result.stdout.split("\n"):
                    if "IOPlatformUUID" in line:
                        machine_id = line.split('"')[-2]
                        break
            except Exception:
                pass
        
        if not machine_id:
            # Fallback: use hostname + config dir path
            import socket
            machine_id = f"{socket.gethostname()}-{self.config_dir}"
        
        # Derive key using PBKDF2
        salt = f"{KEYRING_SERVICE}-{username}".encode()
        key_material = f"{machine_id}-{username}-{KEYRING_SERVICE}".encode()
        
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        self._encryption_key = kdf.derive(key_material)
        return self._encryption_key
    
    def _get_encrypted_file_path(self) -> Path:
        """Get path to encrypted credentials file for current user."""
        username = getpass.getuser()
        # Hash username to avoid filesystem issues with special characters
        user_hash = hashlib.sha256(username.encode()).hexdigest()[:12]
        return self.config_dir / f"credentials_{user_hash}.enc"
    
    def _encrypt(self, data: str) -> bytes:
        """Encrypt data using AES-256-GCM."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        import secrets
        
        key = self._get_encryption_key()
        nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
        
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, data.encode(), None)
        
        # Prepend nonce to ciphertext
        return nonce + ciphertext
    
    def _decrypt(self, encrypted_data: bytes) -> str:
        """Decrypt data using AES-256-GCM."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        
        key = self._get_encryption_key()
        
        # Extract nonce and ciphertext
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]
        
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        
        return plaintext.decode()
    
    def _load_encrypted_credentials(self) -> Dict[str, str]:
        """Load credentials from encrypted file."""
        file_path = self._get_encrypted_file_path()
        
        if not file_path.exists():
            return {}
        
        try:
            encrypted_data = file_path.read_bytes()
            decrypted = self._decrypt(encrypted_data)
            return json.loads(decrypted)
        except Exception as e:
            logger.error(f"Failed to load encrypted credentials: {e}")
            return {}
    
    def _save_encrypted_credentials(self, credentials: Dict[str, str]) -> bool:
        """Save credentials to encrypted file."""
        file_path = self._get_encrypted_file_path()
        
        try:
            json_data = json.dumps(credentials)
            encrypted = self._encrypt(json_data)
            
            # Write with secure permissions
            file_path.write_bytes(encrypted)
            try:
                os.chmod(file_path, 0o600)
            except (OSError, AttributeError):
                pass
            
            return True
        except Exception as e:
            logger.error(f"Failed to save encrypted credentials: {e}")
            return False
    
    def _key_preview(self, key: str) -> str:
        """Generate a safe preview of an API key."""
        if len(key) <= 8:
            return "*" * len(key)
        return f"{key[:4]}...{key[-4:]}"
    
    def store_key(self, provider: str, key: str) -> tuple[bool, str]:
        """
        Store an API key securely.
        
        Args:
            provider: Provider name (anthropic, openai, google, newsapi)
            key: The API key to store
            
        Returns:
            Tuple of (success, storage_method or error message)
        """
        provider = provider.lower()
        
        if provider not in PROVIDERS:
            return False, f"Unknown provider: {provider}"
        
        if not key or not key.strip():
            return False, "Empty key provided"
        
        key = key.strip()
        username = getpass.getuser()
        keyring_key = f"{provider}_{username}"
        
        # Try keyring first
        if self._keyring_available:
            try:
                import keyring
                keyring.set_password(KEYRING_SERVICE, keyring_key, key)
                logger.info(f"Stored {provider} key in keyring for user {username}")
                return True, "keyring"
            except Exception as e:
                logger.warning(f"Keyring storage failed, falling back to encrypted file: {e}")
        
        # Fallback to encrypted file
        try:
            credentials = self._load_encrypted_credentials()
            credentials[provider] = key
            
            if self._save_encrypted_credentials(credentials):
                logger.info(f"Stored {provider} key in encrypted file for user {username}")
                return True, "encrypted_file"
            else:
                return False, "Failed to save encrypted credentials"
        except Exception as e:
            logger.error(f"Failed to store key: {e}")
            return False, str(e)
    
    def get_key(self, provider: str) -> tuple[Optional[str], str]:
        """
        Retrieve an API key with priority order.
        
        Args:
            provider: Provider name (anthropic, openai, google, newsapi)
            
        Returns:
            Tuple of (key or None, source)
            Source is one of: environment, secrets, keyring, encrypted_file, none
        """
        provider = provider.lower()
        
        if provider not in PROVIDERS:
            return None, "unknown_provider"
        
        provider_info = PROVIDERS[provider]
        
        # 1. Check environment variable
        env_key = os.environ.get(provider_info["env_var"])
        if env_key:
            return env_key, "environment"
        
        # 2. Check Streamlit secrets
        try:
            import streamlit as st
            secret_key_name = f"{provider}_api_key"
            if secret_key_name in st.secrets:
                return st.secrets[secret_key_name], "secrets"
        except Exception:
            pass
        
        username = getpass.getuser()
        keyring_key = f"{provider}_{username}"
        
        # 3. Check keyring
        if self._keyring_available:
            try:
                import keyring
                key = keyring.get_password(KEYRING_SERVICE, keyring_key)
                if key:
                    return key, "keyring"
            except Exception as e:
                logger.debug(f"Keyring retrieval failed: {e}")
        
        # 4. Check encrypted file
        try:
            credentials = self._load_encrypted_credentials()
            if provider in credentials:
                return credentials[provider], "encrypted_file"
        except Exception as e:
            logger.debug(f"Encrypted file retrieval failed: {e}")
        
        return None, "none"
    
    def delete_key(self, provider: str) -> tuple[bool, str]:
        """
        Delete a stored API key.
        
        Args:
            provider: Provider name
            
        Returns:
            Tuple of (success, message)
        """
        provider = provider.lower()
        
        if provider not in PROVIDERS:
            return False, f"Unknown provider: {provider}"
        
        username = getpass.getuser()
        keyring_key = f"{provider}_{username}"
        deleted_from = []
        
        # Try to delete from keyring
        if self._keyring_available:
            try:
                import keyring
                keyring.delete_password(KEYRING_SERVICE, keyring_key)
                deleted_from.append("keyring")
            except Exception as e:
                logger.debug(f"Keyring deletion failed (may not exist): {e}")
        
        # Try to delete from encrypted file
        try:
            credentials = self._load_encrypted_credentials()
            if provider in credentials:
                del credentials[provider]
                if self._save_encrypted_credentials(credentials):
                    deleted_from.append("encrypted_file")
        except Exception as e:
            logger.debug(f"Encrypted file deletion failed: {e}")
        
        if deleted_from:
            return True, f"Deleted from: {', '.join(deleted_from)}"
        else:
            return False, "Key not found in any storage"
    
    def list_stored_credentials(self) -> List[StoredCredential]:
        """
        List all stored credentials (metadata only, not the actual keys).
        
        Returns:
            List of StoredCredential objects
        """
        credentials = []
        username = getpass.getuser()
        
        for provider in PROVIDERS:
            key, source = self.get_key(provider)
            
            if key and source in ("keyring", "encrypted_file"):
                credentials.append(StoredCredential(
                    provider=provider,
                    stored_at=datetime.now().isoformat(),  # We don't track actual store time
                    storage_method=source,
                    key_preview=self._key_preview(key)
                ))
        
        return credentials
    
    def get_storage_status(self) -> Dict[str, Any]:
        """
        Get status of credential storage system.
        
        Returns:
            Dictionary with storage system status
        """
        return {
            "keyring_available": self._keyring_available,
            "config_dir": str(self.config_dir),
            "encrypted_file_exists": self._get_encrypted_file_path().exists(),
            "current_user": getpass.getuser(),
        }
    
    def test_key(self, provider: str, key: str) -> tuple[bool, str]:
        """
        Test if an API key is valid by making a minimal API call.
        
        Args:
            provider: Provider name
            key: API key to test
            
        Returns:
            Tuple of (success, message)
        """
        provider = provider.lower()
        
        if provider not in PROVIDERS:
            return False, f"Unknown provider: {provider}"
        
        try:
            if provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=key)
                # Make a minimal request
                client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1,
                    messages=[{"role": "user", "content": "Hi"}]
                )
                return True, "Key is valid"
                
            elif provider == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=key)
                # List models is a lightweight call
                client.models.list()
                return True, "Key is valid"
                
            elif provider == "google":
                import google.generativeai as genai
                genai.configure(api_key=key)
                # List models is lightweight
                list(genai.list_models())
                return True, "Key is valid"
                
            elif provider == "newsapi":
                import requests
                response = requests.get(
                    "https://newsapi.org/v2/top-headlines",
                    params={"apiKey": key, "country": "us", "pageSize": 1},
                    timeout=10
                )
                if response.status_code == 200:
                    return True, "Key is valid"
                elif response.status_code == 401:
                    return False, "Invalid API key"
                else:
                    return False, f"API returned status {response.status_code}"
            
            return False, "Testing not implemented for this provider"
            
        except Exception as e:
            error_msg = str(e)
            if "invalid" in error_msg.lower() or "unauthorized" in error_msg.lower():
                return False, "Invalid API key"
            return False, f"Test failed: {error_msg}"


# Global instance
_credential_manager: Optional[CredentialManager] = None


def get_credential_manager() -> CredentialManager:
    """Get or create the global credential manager instance."""
    global _credential_manager
    if _credential_manager is None:
        _credential_manager = CredentialManager()
    return _credential_manager
