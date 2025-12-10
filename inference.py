"""
Inference Module
Structure scoring using trained model.
"""

import os
import json
import logging
import pickle
from typing import Dict, List, Optional, Any
import numpy as np

from config import (
    MODEL_PATH,
    SCALER_PATH,
    METADATA_PATH,
    GLOBAL_PREF_PATH,
    ATTRIBUTE_VOCABULARY,
    StructureFields
)
from airtable_client import (
    AirtableClient,
    get_structure_metadata
)
from feature_extraction import (
    extract_structure_proxy_features,
    attributes_to_vector
)

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """Raised when trained model is not found."""
    pass


def load_model():
    """
    Load trained model and associated data.
    
    Returns:
        Tuple of (model, scaler, metadata, global_pref)
    """
    if not os.path.exists(MODEL_PATH):
        raise ModelNotFoundError(f"Model not found at {MODEL_PATH}. Run /train first.")
    
    # Load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    
    # Load scaler
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    
    # Load metadata
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    
    # Load global preference vector
    with open(GLOBAL_PREF_PATH, "r") as f:
        global_pref = json.load(f)
    
    return model, scaler, metadata, global_pref


def is_model_loaded() -> bool:
    """Check if trained model exists."""
    return os.path.exists(MODEL_PATH)


def get_model_metadata() -> Optional[Dict[str, Any]]:
    """Get model metadata if available."""
    if not os.path.exists(METADATA_PATH):
        return None
    
    with open(METADATA_PATH, "r") as f:
        return json.load(f)


def get_top_attributes(
    global_pref: Dict[str, float],
    feature_vector: np.ndarray,
    top_n: int = 3
) -> tuple[List[str], List[str]]:
    """
    Identify top positive and negative attributes based on deviation from global preference.
    
    Args:
        global_pref: Global preference vector
        feature_vector: Individual feature vector
        top_n: Number of top attributes to return
        
    Returns:
        Tuple of (positive_attributes, negative_attributes)
    """
    # Extract attribute portion of feature vector (indices 512+384 to 512+384+100)
    attr_start = 512 + 384
    attr_end = attr_start + 100
    
    if len(feature_vector) >= attr_end:
        attr_vector = feature_vector[attr_start:attr_end]
    else:
        # Use global pref as fallback
        attr_vector = attributes_to_vector(global_pref)
    
    # Get global pref as vector
    global_vector = attributes_to_vector(global_pref)
    
    # Calculate deviation (positive = above average, negative = below)
    deviations = {}
    for i, attr in enumerate(ATTRIBUTE_VOCABULARY):
        if i < len(attr_vector) and i < len(global_vector):
            deviations[attr] = attr_vector[i] - global_vector[i]
    
    # Sort by deviation
    sorted_attrs = sorted(deviations.items(), key=lambda x: x[1], reverse=True)
    
    # Top positive (highest deviation)
    positive = [attr for attr, dev in sorted_attrs[:top_n] if dev > 0]
    
    # Top negative (lowest deviation)
    negative = [attr for attr, dev in sorted_attrs[-top_n:] if dev < 0]
    
    # If we don't have positive/negative based on deviation, use absolute values from global
    if not positive:
        positive = list(global_pref.keys())[:top_n]
    
    if not negative:
        negative = list(global_pref.keys())[-top_n:]
    
    return positive, negative


def score_structures() -> Dict[str, Any]:
    """
    Score all active structures.
    
    Returns:
        Dict with structure scores and metadata
    """
    logger.info("Scoring active structures...")
    
    # Check if model exists
    if not is_model_loaded():
        logger.warning("Model not found, returning uniform scores")
        
        # Fetch structures anyway
        client = AirtableClient()
        structures = client.fetch_scoring_data()
        
        return {
            "structures": [
                {
                    "structure_id": s.get(StructureFields.STRUCTURE_ID),
                    "predicted_success_score": 0.5,
                    "top_positive_attributes": [],
                    "top_negative_attributes": [],
                    "current_outlier_count": s.get(StructureFields.OUTLIER_COUNT, 0),
                    "current_z_score": s.get(StructureFields.Z_SCORE, 0),
                    "warning": "No trained model - using default score"
                }
                for s in structures
            ],
            "global_preference_vector": {},
            "model_metadata": None,
            "warning": "Model not trained. Call POST /train first."
        }
    
    # Load model
    model, scaler, metadata, global_pref = load_model()
    
    # Fetch active structures
    client = AirtableClient()
    structures = client.fetch_scoring_data()
    
    logger.info(f"Scoring {len(structures)} active structures...")
    
    results = []
    for struct in structures:
        try:
            struct_id = struct.get(StructureFields.STRUCTURE_ID)
            skeleton = struct.get(StructureFields.SKELETON, "") or ""
            struct_metadata = get_structure_metadata(struct)
            
            # Build proxy features
            features = extract_structure_proxy_features(
                skeleton_text=skeleton,
                structure_metadata=struct_metadata,
                global_pref_vector=global_pref
            )
            
            # Scale features
            features_scaled = scaler.transform([features])
            
            # Predict
            score = model.predict_proba(features_scaled)[0][1]
            
            # Get top attributes
            positive_attrs, negative_attrs = get_top_attributes(
                global_pref, features, top_n=3
            )
            
            results.append({
                "structure_id": struct_id,
                "predicted_success_score": round(float(score), 4),
                "top_positive_attributes": positive_attrs,
                "top_negative_attributes": negative_attrs,
                "current_outlier_count": struct_metadata.get("outlier_count", 0),
                "current_z_score": struct_metadata.get("z_score", 0),
                "renderer": struct_metadata.get("renderer", "Unknown"),
                "skeleton_preview": skeleton[:100] + "..." if len(skeleton) > 100 else skeleton
            })
            
        except Exception as e:
            logger.warning(f"Error scoring structure {struct.get(StructureFields.STRUCTURE_ID)}: {e}")
            continue
    
    # Sort by score descending
    results.sort(key=lambda x: x["predicted_success_score"], reverse=True)
    
    logger.info(f"Scored {len(results)} structures successfully")
    
    # Filter global pref to top N for response
    top_global_pref = dict(list(global_pref.items())[:20])
    
    return {
        "structures": results,
        "global_preference_vector": top_global_pref,
        "model_metadata": {
            "trained_at": metadata.get("trained_at"),
            "training_samples": metadata.get("metrics", {}).get("train_samples", 0) + 
                               metadata.get("metrics", {}).get("val_samples", 0),
            "auc_score": metadata.get("metrics", {}).get("auc_score", 0)
        }
    }


def score_single_structure(
    structure_id: int,
    skeleton: str,
    structure_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Score a single structure.
    
    Args:
        structure_id: Structure ID
        skeleton: Skeleton template text
        structure_metadata: Dict with metadata fields
        
    Returns:
        Dict with score and attributes
    """
    if not is_model_loaded():
        return {
            "structure_id": structure_id,
            "predicted_success_score": 0.5,
            "warning": "No trained model"
        }
    
    model, scaler, metadata, global_pref = load_model()
    
    # Build features
    features = extract_structure_proxy_features(
        skeleton_text=skeleton,
        structure_metadata=structure_metadata,
        global_pref_vector=global_pref
    )
    
    # Scale and predict
    features_scaled = scaler.transform([features])
    score = model.predict_proba(features_scaled)[0][1]
    
    # Get attributes
    positive_attrs, negative_attrs = get_top_attributes(global_pref, features)
    
    return {
        "structure_id": structure_id,
        "predicted_success_score": round(float(score), 4),
        "top_positive_attributes": positive_attrs,
        "top_negative_attributes": negative_attrs
    }
