"""
Feature Extraction Module
Handles CLIP embeddings, GPT-4V attribute extraction, and text embeddings.
"""

import os
import json
import hashlib
import logging
import re
import base64
from typing import Dict, List, Optional, Any
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import time

from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    CLIP_MODEL,
    TEXT_MODEL,
    CLIP_CACHE_DIR,
    ATTRIBUTES_CACHE_DIR,
    CLIP_EMBEDDING_DIM,
    TEXT_EMBEDDING_DIM,
    ATTRIBUTE_VOCABULARY,
    ATTRIBUTE_TO_INDEX,
    get_attribute_extraction_prompt
)

logger = logging.getLogger(__name__)

# =============================================================================
# LAZY LOADING OF MODELS (to reduce startup time)
# =============================================================================

_clip_model = None
_text_model = None


def get_clip_model():
    """Lazy load CLIP model."""
    global _clip_model
    if _clip_model is None:
        logger.info(f"Loading CLIP model: {CLIP_MODEL}")
        from sentence_transformers import SentenceTransformer
        _clip_model = SentenceTransformer(CLIP_MODEL)
        logger.info("CLIP model loaded")
    return _clip_model


def get_text_model():
    """Lazy load text embedding model."""
    global _text_model
    if _text_model is None:
        logger.info(f"Loading text model: {TEXT_MODEL}")
        from sentence_transformers import SentenceTransformer
        _text_model = SentenceTransformer(TEXT_MODEL)
        logger.info("Text model loaded")
    return _text_model


# =============================================================================
# CACHING UTILITIES
# =============================================================================

def ensure_cache_dirs():
    """Create cache directories if they don't exist."""
    os.makedirs(CLIP_CACHE_DIR, exist_ok=True)
    os.makedirs(ATTRIBUTES_CACHE_DIR, exist_ok=True)


def url_to_hash(url: str) -> str:
    """Convert URL to MD5 hash for cache key."""
    return hashlib.md5(url.encode()).hexdigest()


def get_cached_clip_embedding(url: str) -> Optional[np.ndarray]:
    """Get CLIP embedding from cache."""
    ensure_cache_dirs()
    cache_path = os.path.join(CLIP_CACHE_DIR, f"{url_to_hash(url)}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)
    return None


def save_clip_embedding_to_cache(url: str, embedding: np.ndarray):
    """Save CLIP embedding to cache."""
    ensure_cache_dirs()
    cache_path = os.path.join(CLIP_CACHE_DIR, f"{url_to_hash(url)}.npy")
    np.save(cache_path, embedding)


def get_cached_attributes(url: str) -> Optional[Dict[str, float]]:
    """Get GPT-4V attributes from cache."""
    ensure_cache_dirs()
    cache_path = os.path.join(ATTRIBUTES_CACHE_DIR, f"{url_to_hash(url)}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    return None


def save_attributes_to_cache(url: str, attributes: Dict[str, float]):
    """Save GPT-4V attributes to cache."""
    ensure_cache_dirs()
    cache_path = os.path.join(ATTRIBUTES_CACHE_DIR, f"{url_to_hash(url)}.json")
    with open(cache_path, "w") as f:
        json.dump(attributes, f)


def get_cache_stats() -> Dict[str, int]:
    """Get cache statistics."""
    ensure_cache_dirs()
    
    clip_count = len([f for f in os.listdir(CLIP_CACHE_DIR) if f.endswith(".npy")])
    attr_count = len([f for f in os.listdir(ATTRIBUTES_CACHE_DIR) if f.endswith(".json")])
    
    return {
        "clip_embeddings": clip_count,
        "gpt4v_attributes": attr_count
    }


# =============================================================================
# IMAGE LOADING
# =============================================================================

def load_image_from_url(url: str, timeout: int = 30) -> Optional[Image.Image]:
    """
    Load image from URL.
    
    Args:
        url: Image URL
        timeout: Request timeout in seconds
        
    Returns:
        PIL Image or None on failure
    """
    try:
        # Add Airtable authentication if it's an Airtable URL
        headers = {}
        if 'airtable.com' in url:
            from config import AIRTABLE_API_KEY
            if AIRTABLE_API_KEY:
                headers['Authorization'] = f'Bearer {AIRTABLE_API_KEY}'
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image.convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to load image from {url}: {e}")
        return None


# =============================================================================
# CLIP VISUAL EMBEDDING
# =============================================================================

def extract_clip_embedding(image_url: str, use_cache: bool = True) -> np.ndarray:
    """
    Extract CLIP visual embedding from image URL.
    
    Args:
        image_url: URL of the image
        use_cache: Whether to use cache
        
    Returns:
        numpy array of shape (512,)
    """
    # Check cache first
    if use_cache:
        cached = get_cached_clip_embedding(image_url)
        if cached is not None:
            return cached
    
    # Load image
    image = load_image_from_url(image_url)
    if image is None:
        logger.warning(f"Returning zero vector for failed image: {image_url}")
        return np.zeros(CLIP_EMBEDDING_DIM)
    
    # Get embedding
    try:
        model = get_clip_model()
        embedding = model.encode(image)
        embedding = np.array(embedding).astype(np.float32)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        # Cache result
        if use_cache:
            save_clip_embedding_to_cache(image_url, embedding)
        
        return embedding
        
    except Exception as e:
        logger.error(f"CLIP extraction failed for {image_url}: {e}")
        return np.zeros(CLIP_EMBEDDING_DIM)


# =============================================================================
# TEXT EMBEDDING
# =============================================================================

def extract_text_embedding(text: str) -> np.ndarray:
    """
    Extract text embedding using sentence-transformers.
    
    Args:
        text: Input text
        
    Returns:
        numpy array of shape (384,)
    """
    if not text or not isinstance(text, str):
        return np.zeros(TEXT_EMBEDDING_DIM)
    
    try:
        model = get_text_model()
        embedding = model.encode(text)
        embedding = np.array(embedding).astype(np.float32)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
        
    except Exception as e:
        logger.error(f"Text embedding failed: {e}")
        return np.zeros(TEXT_EMBEDDING_DIM)


# =============================================================================
# GPT-4V JSON CLEANING & REPAIR
# =============================================================================

def clean_gpt4v_json(raw_content: str) -> str:
    """
    Clean and repair GPT-4V response to extract valid JSON.
    
    Handles:
    - Markdown code fences
    - Trailing commas
    - Comments
    - BOM characters
    - Explanatory text before/after JSON
    
    Args:
        raw_content: Raw response from GPT-4V
        
    Returns:
        Cleaned JSON string
    """
    content = raw_content.strip()
    
    # Remove BOM and hidden unicode characters
    content = content.encode('utf-8', 'ignore').decode('utf-8')
    content = content.replace('\ufeff', '').replace('\u200b', '')
    
    # Remove markdown code fences (multiple patterns)
    # Pattern 1: ```json ... ```
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n', '', content)
        content = re.sub(r'\n```\s*$', '', content)
    
    # Pattern 2: Remove any remaining backticks
    content = content.strip('`').strip()
    
    # Remove "json" prefix if present
    if content.lower().startswith("json"):
        content = content[4:].strip()
    
    # Extract JSON object between first { and last }
    if "{" in content and "}" in content:
        start = content.index("{")
        end = content.rindex("}") + 1
        content = content[start:end]
    
    # Remove trailing commas before closing braces/brackets
    # Pattern: , followed by optional whitespace then }
    content = re.sub(r',(\s*[}\]])', r'\1', content)
    
    # Remove single-line comments (// style)
    content = re.sub(r'//[^\n]*', '', content)
    
    # Remove multi-line comments (/* ... */)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    
    return content.strip()


def extract_attributes_with_regex(content: str) -> Optional[Dict[str, float]]:
    """
    Fallback: Extract attributes using regex when JSON parsing fails.
    
    Args:
        content: Content to extract from
        
    Returns:
        Dict of attributes or None if extraction fails
    """
    attributes = {}
    
    # Pattern: "attribute_name": 0.5 or "attribute_name":0.5
    pattern = r'"([^"]+)"\s*:\s*([0-9.]+)'
    matches = re.findall(pattern, content)
    
    if not matches:
        return None
    
    for attr_name, value_str in matches:
        # Only accept known attributes
        if attr_name in ATTRIBUTE_VOCABULARY:
            try:
                value = float(value_str)
                # Clamp to [0, 1]
                attributes[attr_name] = max(0.0, min(1.0, value))
            except ValueError:
                continue
    
    # Only return if we got a reasonable number of attributes
    if len(attributes) >= len(ATTRIBUTE_VOCABULARY) // 2:
        return attributes
    
    return None


def save_debug_response(image_url: str, raw_content: str, attempt: int):
    """
    Save raw GPT-4V response to debug log for inspection.
    
    Args:
        image_url: Image URL (for identification)
        raw_content: Raw response content
        attempt: Attempt number
    """
    debug_dir = os.path.join(ATTRIBUTES_CACHE_DIR, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    url_hash = url_to_hash(image_url)
    debug_file = os.path.join(debug_dir, f"{url_hash}_attempt{attempt}.txt")
    
    try:
        with open(debug_file, "w", encoding="utf-8") as f:
            f.write(f"Image URL: {image_url}\n")
            f.write(f"Attempt: {attempt}\n")
            f.write("=" * 60 + "\n")
            f.write(raw_content)
            f.write("\n" + "=" * 60 + "\n")
        logger.info(f"Debug response saved to {debug_file}")
    except Exception as e:
        logger.warning(f"Could not save debug response: {e}")


def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """
    Convert PIL Image to base64 string.
    
    Args:
        image: PIL Image object
        format: Image format (JPEG or PNG)
        
    Returns:
        Base64 encoded string
    """
    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    image_bytes = buffer.read()
    base64_string = base64.b64encode(image_bytes).decode('utf-8')
    return base64_string


# =============================================================================
# GPT-4V ATTRIBUTE EXTRACTION
# =============================================================================

def extract_attributes_gpt4v(
    image_url: str, 
    use_cache: bool = True,
    max_retries: int = 3,
    image_data: Optional[bytes] = None
) -> Dict[str, float]:
    """
    Extract fashion attributes from image using GPT-4V.
    
    Args:
        image_url: URL of the image (used for cache key)
        use_cache: Whether to use cache
        max_retries: Number of retry attempts
        image_data: Optional pre-downloaded image bytes (bypasses URL download)
        
    Returns:
        Dict mapping attribute names to scores (0.0-1.0)
    """
    # Check cache first
    if use_cache:
        cached = get_cached_attributes(image_url)
        if cached is not None:
            return cached
    
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set, returning zero attributes")
        return {attr: 0.0 for attr in ATTRIBUTE_VOCABULARY}
    
    # Load image - either from provided bytes or download from URL
    if image_data:
        logger.info(f"Using provided image data ({len(image_data)} bytes)")
        try:
            image = Image.open(BytesIO(image_data))
            image = image.convert("RGB")
        except Exception as e:
            logger.error(f"Could not load image from provided data: {e}")
            return {attr: 0.0 for attr in ATTRIBUTE_VOCABULARY}
    else:
        logger.info(f"Downloading image from: {image_url}")
        image = load_image_from_url(image_url)
        if image is None:
            logger.error(f"Could not download image from {image_url}")
            return {attr: 0.0 for attr in ATTRIBUTE_VOCABULARY}
    
    # Convert to base64
    base64_image = image_to_base64(image, format="JPEG")
    logger.info(f"Converted image to base64 ({len(base64_image)} chars)")
    
    # Prepare prompt
    prompt = get_attribute_extraction_prompt()
    
    # Call GPT-4V with retries
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.1  # Low temperature for consistent scoring
            )
            
            # Get raw response
            raw_content = response.choices[0].message.content or ""
            
            # Save debug response for first attempt (helps diagnose issues)
            if attempt == 0:
                save_debug_response(image_url, raw_content, attempt + 1)
            
            # TIER 1: Use enhanced JSON cleaning
            cleaned_content = clean_gpt4v_json(raw_content)
            logger.debug(f"Cleaned content length: {len(cleaned_content)} chars")
            
            # TIER 2: Try standard JSON parsing
            attributes = None
            try:
                attributes = json.loads(cleaned_content)
                logger.debug(f"JSON parsed successfully with {len(attributes)} keys")
            except json.JSONDecodeError as json_err:
                logger.debug(f"Standard JSON parse failed: {json_err}")
                
                # TIER 3: Try regex fallback extraction
                logger.debug("Attempting regex fallback extraction...")
                attributes = extract_attributes_with_regex(cleaned_content)
                
                if attributes:
                    logger.info(f"Regex extraction recovered {len(attributes)} attributes")
                else:
                    logger.warning("Regex extraction also failed")
                    # Save failed response for analysis
                    save_debug_response(image_url, raw_content, attempt + 1)
                    raise ValueError("All parsing methods failed")
            
            # Validate that we got a dict
            if not isinstance(attributes, dict):
                raise ValueError(f"Expected dict, got {type(attributes)}")
            
            # Validate and normalize all 100 attributes
            validated = {}
            for attr in ATTRIBUTE_VOCABULARY:
                value = attributes.get(attr, 0.0)
                try:
                    # Ensure value is in [0, 1]
                    validated[attr] = max(0.0, min(1.0, float(value)))
                except (ValueError, TypeError):
                    logger.warning(f"Invalid value for {attr}: {value}, using 0.0")
                    validated[attr] = 0.0
            
            # Log success
            non_zero = sum(1 for v in validated.values() if v > 0.1)
            logger.info(f"Successfully extracted attributes: {non_zero}/{len(ATTRIBUTE_VOCABULARY)} non-zero")
            
            # Cache result
            if use_cache:
                save_attributes_to_cache(image_url, validated)
            
            return validated
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                
        except Exception as e:
            logger.warning(f"GPT-4V error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    # All retries failed
    logger.error(f"GPT-4V extraction failed after {max_retries} attempts for {image_url}")
    return {attr: 0.0 for attr in ATTRIBUTE_VOCABULARY}


def attributes_to_vector(attributes: Dict[str, float]) -> np.ndarray:
    """
    Convert attribute dict to fixed-size vector.
    
    Args:
        attributes: Dict mapping attribute names to scores
        
    Returns:
        numpy array of shape (100,)
    """
    vector = np.zeros(len(ATTRIBUTE_VOCABULARY))
    
    for attr, score in attributes.items():
        if attr in ATTRIBUTE_TO_INDEX:
            vector[ATTRIBUTE_TO_INDEX[attr]] = score
    
    return vector.astype(np.float32)


# =============================================================================
# COMBINED FEATURE EXTRACTION
# =============================================================================

def extract_all_features(
    image_url: str,
    prompt_text: str,
    skeleton_text: str,
    structure_metadata: Dict[str, Any],
    use_cache: bool = True
) -> np.ndarray:
    """
    Extract complete feature vector for an image.
    
    Args:
        image_url: URL of the image
        prompt_text: Prompt used to generate the image
        skeleton_text: Skeleton template from structure
        structure_metadata: Dict with usage_count, outlier_count, etc.
        use_cache: Whether to use cache for CLIP and GPT-4V
        
    Returns:
        numpy array of shape (1387,)
    """
    from config import encode_renderer
    
    # 1. Visual embedding (512D)
    visual_emb = extract_clip_embedding(image_url, use_cache=use_cache)
    
    # 2. Text embedding (384D)
    text_emb = extract_text_embedding(prompt_text)
    
    # 3. Attribute vector (100D)
    attributes = extract_attributes_gpt4v(image_url, use_cache=use_cache)
    attr_vector = attributes_to_vector(attributes)
    
    # 4. Skeleton embedding (384D)
    skeleton_emb = extract_text_embedding(skeleton_text)
    
    # 5. Structure metadata (7D)
    metadata = np.array([
        structure_metadata.get("usage_count", 0),
        structure_metadata.get("outlier_count", 0),
        structure_metadata.get("age_weeks", 0),
        structure_metadata.get("z_score", 0),
        structure_metadata.get("ai_score", 5.0),
        encode_renderer(structure_metadata.get("renderer", "ImageFX")),
        structure_metadata.get("structure_id", 0)
    ], dtype=np.float32)
    
    # Concatenate all features
    features = np.concatenate([
        visual_emb,      # 512D
        text_emb,        # 384D
        attr_vector,     # 100D
        skeleton_emb,    # 384D
        metadata         # 7D
    ])
    
    return features.astype(np.float32)


def extract_structure_proxy_features(
    skeleton_text: str,
    structure_metadata: Dict[str, Any],
    global_pref_vector: Dict[str, float]
) -> np.ndarray:
    """
    Extract proxy features for a structure (no specific image).
    Uses global preference vector as proxy for visual/attribute features.
    
    Args:
        skeleton_text: Skeleton template text
        structure_metadata: Dict with usage_count, outlier_count, etc.
        global_pref_vector: Averaged attributes from liked images
        
    Returns:
        numpy array of shape (1387,)
    """
    from config import encode_renderer
    
    # 1. Visual embedding proxy - use zero vector or could compute average
    visual_proxy = np.zeros(CLIP_EMBEDDING_DIM)
    
    # 2. Text embedding from skeleton (use as proxy for prompt)
    skeleton_emb = extract_text_embedding(skeleton_text)
    
    # 3. Attribute vector from global preference
    attr_vector = attributes_to_vector(global_pref_vector)
    
    # 4. Skeleton embedding (same as text proxy)
    # Already computed above
    
    # 5. Structure metadata
    metadata = np.array([
        structure_metadata.get("usage_count", 0),
        structure_metadata.get("outlier_count", 0),
        structure_metadata.get("age_weeks", 0),
        structure_metadata.get("z_score", 0),
        structure_metadata.get("ai_score", 5.0),
        encode_renderer(structure_metadata.get("renderer", "ImageFX")),
        structure_metadata.get("structure_id", 0)
    ], dtype=np.float32)
    
    # Concatenate
    features = np.concatenate([
        visual_proxy,    # 512D (zeros)
        skeleton_emb,    # 384D (proxy for text)
        attr_vector,     # 100D (from global pref)
        skeleton_emb,    # 384D
        metadata         # 7D
    ])
    
    return features.astype(np.float32)
