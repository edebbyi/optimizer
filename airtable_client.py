"""
Airtable Client
Handles all data fetching from Airtable.
"""

import logging
from typing import Dict, List, Optional, Any
from pyairtable import Api, Table
from pyairtable.formulas import match

from config import (
    AIRTABLE_API_KEY,
    AIRTABLE_BASE_ID,
    AIRTABLE_IMAGES_TABLE,
    AIRTABLE_STRUCTURES_TABLE,
    AIRTABLE_IMAGES_VIEW,
    AIRTABLE_STRUCTURES_VIEW_ACTIVE,
    AIRTABLE_STRUCTURES_VIEW_ALL,
    ImageFields,
    StructureFields,
    STATUS_OUTLIER
)

logger = logging.getLogger(__name__)


class AirtableClient:
    """Client for fetching data from Airtable."""
    
    def __init__(self):
        if not AIRTABLE_API_KEY:
            raise ValueError("AIRTABLE_API_KEY environment variable not set")
        
        self.api = Api(AIRTABLE_API_KEY)
        self.images_table = self.api.table(AIRTABLE_BASE_ID, AIRTABLE_IMAGES_TABLE)
        self.structures_table = self.api.table(AIRTABLE_BASE_ID, AIRTABLE_STRUCTURES_TABLE)
    
    def fetch_all_images(self, view: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch all images from Airtable.
        
        Args:
            view: Optional view ID to filter by
            
        Returns:
            List of image records with fields
        """
        view = view or AIRTABLE_IMAGES_VIEW
        
        logger.info(f"Fetching images from view: {view}")
        
        try:
            records = self.images_table.all(view=view)
            logger.info(f"Fetched {len(records)} image records")
            
            # Extract fields from records
            images = []
            for record in records:
                fields = record.get("fields", {})
                fields["_record_id"] = record.get("id")
                images.append(fields)
            
            return images
            
        except Exception as e:
            logger.error(f"Error fetching images: {e}")
            raise
    
    def fetch_all_structures(self, view: Optional[str] = None, active_only: bool = False) -> List[Dict[str, Any]]:
        """
        Fetch all structures from Airtable.
        
        Args:
            view: Optional view ID to filter by
            active_only: If True, fetch only active structures
            
        Returns:
            List of structure records with fields
        """
        if active_only:
            view = view or AIRTABLE_STRUCTURES_VIEW_ACTIVE
        else:
            view = view or AIRTABLE_STRUCTURES_VIEW_ALL
        
        logger.info(f"Fetching structures from view: {view}")
        
        try:
            records = self.structures_table.all(view=view)
            logger.info(f"Fetched {len(records)} structure records")
            
            # Extract fields from records
            structures = []
            for record in records:
                fields = record.get("fields", {})
                fields["_record_id"] = record.get("id")
                structures.append(fields)
            
            return structures
            
        except Exception as e:
            logger.error(f"Error fetching structures: {e}")
            raise
    
    def get_structures_by_id(self, active_only: bool = False) -> Dict[int, Dict[str, Any]]:
        """
        Fetch structures indexed by Structure ID.
        
        Args:
            active_only: If True, fetch only active structures
            
        Returns:
            Dict mapping Structure ID to structure record
        """
        structures = self.fetch_all_structures(active_only=active_only)
        
        structures_by_id = {}
        for struct in structures:
            struct_id = struct.get(StructureFields.STRUCTURE_ID)
            if struct_id is not None:
                # Handle if struct_id comes as string or int
                try:
                    struct_id = int(struct_id)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid Structure ID: {struct_id}")
                    continue
                structures_by_id[struct_id] = struct
        
        logger.info(f"Indexed {len(structures_by_id)} structures by ID")
        return structures_by_id
    
    def fetch_training_data(self) -> tuple[List[Dict], Dict[int, Dict]]:
        """
        Fetch all data needed for training.
        
        Returns:
            Tuple of (images list, structures dict by ID)
        """
        images = self.fetch_all_images()
        structures_by_id = self.get_structures_by_id(active_only=False)  # Need ALL for historical joins
        
        return images, structures_by_id
    
    def fetch_scoring_data(self) -> List[Dict[str, Any]]:
        """
        Fetch active structures for scoring.
        
        Returns:
            List of active structure records
        """
        return self.fetch_all_structures(active_only=True)
    
    def count_outliers(self) -> int:
        """
        Count total outlier images.
        
        Returns:
            Number of images with Status = Outlier
        """
        try:
            # Use formula to filter
            formula = match({ImageFields.STATUS: STATUS_OUTLIER})
            records = self.images_table.all(formula=formula)
            return len(records)
        except Exception as e:
            logger.error(f"Error counting outliers: {e}")
            return 0


def get_image_url(image_record: Dict[str, Any]) -> Optional[str]:
    """
    Extract the best image URL from an image record.
    
    Args:
        image_record: Image record from Airtable
        
    Returns:
        Image URL string or None
    """
    # Prefer attachment URLs (more likely to be fresh)
    image_field = image_record.get("Image")
    # Case 1: list of attachment dicts
    if isinstance(image_field, list) and image_field:
        for att in image_field:
            if isinstance(att, dict):
                att_url = att.get("url")
                if att_url and isinstance(att_url, str) and att_url.startswith("http"):
                    return att_url
    # Case 2: string like "filename.jpg (https://...)"
    if image_field and isinstance(image_field, str):
        if "(" in image_field and ")" in image_field:
            start = image_field.rfind("(") + 1
            end = image_field.rfind(")")
            if start < end:
                extracted_url = image_field[start:end]
                if extracted_url.startswith("http"):
                    return extracted_url

    # Try direct ImageUrl field first
    url = image_record.get(ImageFields.IMAGE_URL)
    if url and isinstance(url, str) and url.startswith("http"):
        return url
    
    return None


def get_image_attachment_data(image_record: Dict[str, Any]) -> Optional[bytes]:
    """
    Download image data directly from Airtable attachment.
    This uses authenticated requests through the Airtable SDK.
    
    Args:
        image_record: Image record from Airtable
        
    Returns:
        Image bytes or None
    """
    import requests
    from config import AIRTABLE_API_KEY
    
    # Get the first attachment from Image field
    image_field = image_record.get("Image")
    if not isinstance(image_field, list) or not image_field:
        return None
    
    attachment = image_field[0]
    if not isinstance(attachment, dict):
        return None
    
    url = attachment.get("url")
    if not url or not isinstance(url, str):
        return None
    
    try:
        # Use Airtable API key for authentication
        headers = {}
        if AIRTABLE_API_KEY:
            headers['Authorization'] = f'Bearer {AIRTABLE_API_KEY}'
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.warning(f"Failed to download attachment from {url}: {e}")
        return None


def get_structure_id(image_record: Dict[str, Any]) -> Optional[int]:
    """
    Extract Structure ID from image record.
    
    Args:
        image_record: Image record from Airtable
        
    Returns:
        Structure ID as int or None
    """
    struct_id = image_record.get(ImageFields.STRUCTURE_ID)
    
    if struct_id is None:
        return None
    
    # Handle various formats (linked records often come back as lists)
    candidates = struct_id if isinstance(struct_id, list) else [struct_id]
    
    for candidate in candidates:
        try:
            if isinstance(candidate, (int, float)):
                return int(candidate)
            if isinstance(candidate, str):
                # Might be "69", "69.0", or numeric string from lookup
                return int(float(candidate))
            if isinstance(candidate, dict):
                # If a dict is returned, try common keys
                for key in ("Structure ID", "id", "value"):
                    if key in candidate:
                        return int(float(candidate[key]))
        except (ValueError, TypeError):
            continue
    
    return None


def get_label(image_record: Dict[str, Any]) -> int:
    """
    Get training label from image record.
    
    Args:
        image_record: Image record from Airtable
        
    Returns:
        1 for Outlier (liked), 0 for anything else
    """
    status = image_record.get(ImageFields.STATUS, "")
    
    if status and isinstance(status, str):
        if status.lower() == "outlier":
            return 1
    
    return 0


def parse_ai_score(ai_score_str: str) -> float:
    """
    Parse AI Score from string format like "8/10".
    
    Args:
        ai_score_str: AI Score string
        
    Returns:
        Float value 0-10
    """
    if not ai_score_str:
        return 5.0  # Default middle value
    
    try:
        if "/" in str(ai_score_str):
            # Format: "8/10"
            parts = str(ai_score_str).split("/")
            return float(parts[0])
        else:
            return float(ai_score_str)
    except (ValueError, IndexError):
        return 5.0


def get_structure_metadata(structure: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract metadata features from structure record.
    
    Args:
        structure: Structure record from Airtable
        
    Returns:
        Dict with metadata values
    """
    return {
        "usage_count": float(structure.get(StructureFields.USAGE_COUNT, 0) or 0),
        "outlier_count": float(structure.get(StructureFields.OUTLIER_COUNT, 0) or 0),
        "age_weeks": float(structure.get(StructureFields.AGE_WEEKS, 0) or 0),
        "z_score": float(structure.get(StructureFields.Z_SCORE, 0) or 0),
        "ai_score": parse_ai_score(structure.get(StructureFields.AI_SCORE, "5/10")),
        "renderer": structure.get(StructureFields.RENDERER, "ImageFX"),
        "structure_id": int(structure.get(StructureFields.STRUCTURE_ID, 0) or 0)
    }
