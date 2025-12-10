"""
Utility Functions
Helper functions for the Optimizer service.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from config import (
    MODEL_DIR,
    CACHE_DIR,
    METADATA_PATH,
    GLOBAL_PREF_PATH
)

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """
    Configure logging for the application.
    
    Args:
        level: Logging level string
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )


def ensure_directories():
    """Create all required directories."""
    dirs = [
        MODEL_DIR,
        CACHE_DIR,
        os.path.join(CACHE_DIR, "clip"),
        os.path.join(CACHE_DIR, "attributes")
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.debug(f"Ensured directory: {dir_path}")


def get_service_info() -> Dict[str, Any]:
    """
    Get service status information.
    
    Returns:
        Dict with service info
    """
    from feature_extraction import get_cache_stats
    from inference import is_model_loaded, get_model_metadata
    
    info = {
        "service": "optimizer-agent",
        "version": "1.0.0",
        "model_loaded": is_model_loaded(),
        "cache_stats": get_cache_stats()
    }
    
    # Add model info if available
    if is_model_loaded():
        metadata = get_model_metadata()
        if metadata:
            info["last_trained"] = metadata.get("trained_at")
            info["model_metrics"] = metadata.get("metrics", {})
    
    return info


def validate_image_url(url: str) -> bool:
    """
    Validate that a URL looks like an image URL.
    
    Args:
        url: URL string
        
    Returns:
        True if valid image URL
    """
    if not url or not isinstance(url, str):
        return False
    
    if not url.startswith(("http://", "https://")):
        return False
    
    # Check for common image hosting domains
    valid_domains = [
        "replicate.delivery",
        "cloudinary.com",
        "airtableusercontent.com",
        "res.cloudinary.com"
    ]
    
    return any(domain in url for domain in valid_domains)


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to max length with ellipsis.
    
    Args:
        text: Input text
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Format datetime as ISO string.
    
    Args:
        dt: Datetime object (defaults to now)
        
    Returns:
        ISO formatted string
    """
    if dt is None:
        dt = datetime.utcnow()
    
    return dt.isoformat() + "Z"


def load_json_file(filepath: str) -> Optional[Dict]:
    """
    Load JSON file safely.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dict or None if file doesn't exist
    """
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file {filepath}: {e}")
        return None


def save_json_file(filepath: str, data: Dict):
    """
    Save data to JSON file.
    
    Args:
        filepath: Path to JSON file
        data: Dict to save
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


class HealthCheck:
    """Health check utility."""
    
    @staticmethod
    def check_airtable() -> Dict[str, Any]:
        """Check Airtable connectivity."""
        try:
            from airtable_client import AirtableClient
            client = AirtableClient()
            # Try a simple query
            structures = client.fetch_all_structures(active_only=True)
            return {
                "status": "healthy",
                "structures_count": len(structures)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    @staticmethod
    def check_openai() -> Dict[str, Any]:
        """Check OpenAI API connectivity."""
        try:
            from openai import OpenAI
            from config import OPENAI_API_KEY
            
            if not OPENAI_API_KEY:
                return {
                    "status": "unconfigured",
                    "error": "OPENAI_API_KEY not set"
                }
            
            client = OpenAI(api_key=OPENAI_API_KEY)
            # Just check we can create a client
            return {"status": "healthy"}
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    @staticmethod
    def check_model() -> Dict[str, Any]:
        """Check if trained model exists."""
        from inference import is_model_loaded, get_model_metadata
        
        if is_model_loaded():
            metadata = get_model_metadata()
            return {
                "status": "healthy",
                "trained_at": metadata.get("trained_at") if metadata else None
            }
        else:
            return {
                "status": "not_trained",
                "message": "Model not found. Call POST /train to create."
            }
    
    @staticmethod
    def full_check() -> Dict[str, Any]:
        """Run all health checks."""
        return {
            "airtable": HealthCheck.check_airtable(),
            "openai": HealthCheck.check_openai(),
            "model": HealthCheck.check_model()
        }
