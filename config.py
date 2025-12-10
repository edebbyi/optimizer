"""
Optimizer Agent Configuration
All constants, environment variables, and vocabulary definitions.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# AIRTABLE CONFIGURATION
# =============================================================================

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID", "appW8hvRj3lUrqEH2")

# Tables
AIRTABLE_IMAGES_TABLE = os.getenv("AIRTABLE_IMAGES_TABLE", "tblwnQTC0Tp8xgTue")
AIRTABLE_STRUCTURES_TABLE = os.getenv("AIRTABLE_STRUCTURES_TABLE", "tblPPDf9vlTBv2kyl")

# Views
AIRTABLE_IMAGES_VIEW = os.getenv("AIRTABLE_IMAGES_VIEW", "viwZ4vi03GR4oMXGg")
AIRTABLE_STRUCTURES_VIEW_ACTIVE = os.getenv("AIRTABLE_STRUCTURES_VIEW_ACTIVE", "viwoC5FLPRgpvkjj6")
AIRTABLE_STRUCTURES_VIEW_ALL = os.getenv("AIRTABLE_STRUCTURES_VIEW_ALL", "viw0O8ywpWywfjZPt")

# =============================================================================
# OPENAI CONFIGURATION
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

CLIP_MODEL = os.getenv("CLIP_MODEL", "clip-ViT-B-32")
TEXT_MODEL = os.getenv("TEXT_MODEL", "all-MiniLM-L6-v2")

# Training thresholds
MIN_TRAINING_SAMPLES = int(os.getenv("MIN_TRAINING_SAMPLES", "100"))
MIN_POSITIVE_SAMPLES = int(os.getenv("MIN_POSITIVE_SAMPLES", "20"))
# Optional cap on training samples (most recent first)
MAX_TRAINING_SAMPLES = int(os.getenv("MAX_TRAINING_SAMPLES", "100"))

# XGBoost hyperparameters
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "early_stopping_rounds": 10,
    "random_state": 42,
    "n_jobs": -1
}

# =============================================================================
# PATHS
# =============================================================================

MODEL_DIR = os.getenv("MODEL_DIR", "model")
CACHE_DIR = os.getenv("CACHE_DIR", "cache")
CLIP_CACHE_DIR = os.path.join(CACHE_DIR, "clip")
ATTRIBUTES_CACHE_DIR = os.path.join(CACHE_DIR, "attributes")

MODEL_PATH = os.path.join(MODEL_DIR, "preference_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")
GLOBAL_PREF_PATH = os.path.join(MODEL_DIR, "global_preference_vector.json")

# =============================================================================
# FEATURE DIMENSIONS
# =============================================================================

CLIP_EMBEDDING_DIM = 512
TEXT_EMBEDDING_DIM = 384
ATTRIBUTE_DIM = 100  # Length of ATTRIBUTE_VOCABULARY
SKELETON_EMBEDDING_DIM = 384
METADATA_DIM = 7

TOTAL_FEATURE_DIM = (
    CLIP_EMBEDDING_DIM + 
    TEXT_EMBEDDING_DIM + 
    ATTRIBUTE_DIM + 
    SKELETON_EMBEDDING_DIM + 
    METADATA_DIM
)  # = 1387

# =============================================================================
# RENDERER ENCODING
# =============================================================================

RENDERER_ENCODING = {
    "ImageFX": 0,
    "Recraft": 1,
    "Midjourney": 2,
    "DALL-E": 3,
    "Stable Diffusion": 4
}

def encode_renderer(renderer: str) -> int:
    """Encode renderer name to integer."""
    return RENDERER_ENCODING.get(renderer, 0)

# =============================================================================
# FASHION ATTRIBUTE VOCABULARY (100 attributes)
# =============================================================================

ATTRIBUTE_VOCABULARY = [
    # Silhouette & Fit (10)
    "oversized",
    "tailored",
    "relaxed_fit",
    "slim_fit",
    "cropped",
    "elongated",
    "boxy",
    "fitted",
    "flowy",
    "structured",
    
    # Construction (10)
    "patch_pockets",
    "welt_pockets",
    "flap_pockets",
    "hidden_placket",
    "visible_stitching",
    "raw_edge",
    "clean_finish",
    "double_breasted",
    "single_breasted",
    "asymmetric_closure",
    
    # Collar & Neckline (10)
    "high_collar",
    "mandarin_collar",
    "spread_collar",
    "notch_lapel",
    "shawl_collar",
    "collarless",
    "hood",
    "funnel_neck",
    "v_neck",
    "crew_neck",
    
    # Sleeve (9)
    "raglan_sleeve",
    "set_in_sleeve",
    "dolman_sleeve",
    "rolled_sleeve",
    "cuffed",
    "elastic_cuff",
    "ribbed_cuff",
    "sleeveless",
    "three_quarter_sleeve",
    
    # Details (10)
    "drawstring",
    "elasticated_waist",
    "belt_loops",
    "self_belt",
    "epaulettes",
    "shoulder_tabs",
    "back_vent",
    "side_vents",
    "kick_pleat",
    "button_front",
    
    # Fabric Appearance (10)
    "matte_finish",
    "subtle_sheen",
    "textured",
    "smooth",
    "heathered",
    "marled",
    "ribbed_knit",
    "jersey_knit",
    "woven",
    "bonded",
    
    # Color & Tone (10)
    "monochromatic",
    "tonal",
    "earth_tones",
    "neutral",
    "muted",
    "saturated",
    "dark_palette",
    "light_palette",
    "contrast_trim",
    "color_blocked",
    
    # Material Impression (9)
    "performance_fabric",
    "technical",
    "natural_fiber_look",
    "luxe_hand",
    "lightweight",
    "mid_weight",
    "substantial",
    "drapey",
    "crisp",
    
    # Style DNA (10)
    "safari_influence",
    "military_influence",
    "athletic_influence",
    "minimalist",
    "utilitarian",
    "refined_casual",
    "elevated_basic",
    "travel_ready",
    "versatile",
    "layering_piece",
    
    # Hardware (8)
    "tonal_hardware",
    "matte_hardware",
    "minimal_hardware",
    "contrast_hardware",
    "hidden_zipper",
    "exposed_zipper",
    "snap_closure",
    "toggle_closure",
    
    # ANATOMIE Specific (4)
    "wrinkle_resistant_look",
    "packable_appearance",
    "stretch_evident",
    "ponte_like"
]

# Verify vocabulary length
assert len(ATTRIBUTE_VOCABULARY) == ATTRIBUTE_DIM, f"Vocabulary length {len(ATTRIBUTE_VOCABULARY)} != {ATTRIBUTE_DIM}"

# Create index mapping for fast lookup
ATTRIBUTE_TO_INDEX = {attr: idx for idx, attr in enumerate(ATTRIBUTE_VOCABULARY)}

# =============================================================================
# GPT-4V ATTRIBUTE EXTRACTION PROMPT
# =============================================================================

ATTRIBUTE_EXTRACTION_PROMPT = """Analyze this fashion garment image. Score each attribute from 0.0 to 1.0 based on how strongly it applies to the PRIMARY garment shown.

Return ONLY valid JSON with no additional text, in this exact format:
{{
    "oversized": 0.0,
    "tailored": 0.8,
    ...
}}

ATTRIBUTES TO SCORE:
{attributes}

SCORING RULES:
- 0.0 = attribute definitely NOT present
- 0.3 = slightly present or ambiguous
- 0.5 = moderately present
- 0.7 = clearly present
- 1.0 = strongly/dominantly present
- Be precise and discriminating - most scores should be < 0.3 or > 0.7
- Focus on the main outfit, not accessories
- This is luxury performance travel wear - calibrate accordingly

Return ONLY the JSON object, no markdown formatting, no explanation."""

def get_attribute_extraction_prompt() -> str:
    """Get the GPT-4V prompt with all attributes listed."""
    attributes_list = "\n".join(f"- {attr}" for attr in ATTRIBUTE_VOCABULARY)
    return ATTRIBUTE_EXTRACTION_PROMPT.format(attributes=attributes_list)

# =============================================================================
# AIRTABLE FIELD NAMES
# =============================================================================

class ImageFields:
    """Airtable Images table field names."""
    NAME = "Name"
    IMAGE_URL = "ImageUrl"
    STATUS = "Status"
    PROMPT = "Prompt"
    STRUCTURE_ID = "Structure ID"
    SKELETON = "Skeleton"
    CREATED = "imgsAirCreated"
    MODIFIED = "imgsAirModified"

class StructureFields:
    """Airtable Structures table field names."""
    STRUCTURE_ID = "Structure ID"
    SKELETON = "Skeleton"
    RENDERER = "Renderer"
    OUTLIER_COUNT = "outlier_count"
    USAGE_COUNT = "usage_count"
    AGE_WEEKS = "age_weeks"
    Z_SCORE = "z_score"
    AI_SCORE = "AI Score"
    STATUS = "Status"
    TREND = "trend"

# =============================================================================
# STATUS VALUES
# =============================================================================

STATUS_OUTLIER = "Outlier"  # Positive label (liked)
STATUS_PASS = "Pass"        # Negative label (not liked)
