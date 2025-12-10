"""
Training Pipeline
XGBoost model training on image features.
"""

import os
import json
import logging
import pickle
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import xgboost as xgb

from config import (
    XGB_PARAMS,
    MODEL_PATH,
    SCALER_PATH,
    METADATA_PATH,
    GLOBAL_PREF_PATH,
    MODEL_DIR,
    MIN_TRAINING_SAMPLES,
    MIN_POSITIVE_SAMPLES,
    MAX_TRAINING_SAMPLES,
    ATTRIBUTE_VOCABULARY,
    TOTAL_FEATURE_DIM,
    ImageFields,
    StructureFields
)
from airtable_client import (
    AirtableClient,
    get_image_url,
    get_structure_id,
    get_label,
    get_structure_metadata,
    get_image_attachment_data
)
from feature_extraction import (
    extract_all_features,
    attributes_to_vector,
    extract_attributes_gpt4v
)

logger = logging.getLogger(__name__)


def ensure_model_dir():
    """Create model directory if it doesn't exist."""
    os.makedirs(MODEL_DIR, exist_ok=True)


def extract_prompt_text(image_record: Dict) -> str:
    """
    Extract prompt text from image record, handling lookup field formats.
    
    Airtable lookup fields return arrays, so this handles:
    - String: "A minimalist jacket..." → returns as-is
    - List: ["A minimalist jacket..."] → extracts first element
    - None/empty: returns empty string
    
    Args:
        image_record: Image record from Airtable
        
    Returns:
        Prompt text as string
    """
    prompt_raw = image_record.get(ImageFields.PROMPT, "") or ""
    
    # Handle lookup field returning as list
    if isinstance(prompt_raw, list):
        prompt_text = prompt_raw[0] if prompt_raw else ""
    else:
        prompt_text = prompt_raw
    
    # Ensure it's a string
    if not isinstance(prompt_text, str):
        return ""
    
    return prompt_text.strip()


def parse_created_timestamp(ts: Any) -> datetime:
    """Parse Airtable timestamp string into datetime, fallback to epoch."""
    if not ts:
        return datetime.min.replace(tzinfo=timezone.utc)
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)
    return datetime.min.replace(tzinfo=timezone.utc)


def sort_images_by_created(images: List[Dict]) -> List[Dict]:
    """Sort images newest-first by created timestamp field."""
    return sorted(
        images,
        key=lambda img: parse_created_timestamp(img.get(ImageFields.CREATED)),
        reverse=True,
    )


def compute_global_preference_vector(
    images: List[Dict], 
    structures_by_id: Dict[int, Dict]
) -> Dict[str, float]:
    """
    Compute global preference vector by averaging attributes of liked images.
    
    Args:
        images: List of image records
        structures_by_id: Dict mapping structure ID to structure record
        
    Returns:
        Dict mapping attribute names to average scores
    """
    logger.info("Computing global preference vector...")
    
    attribute_sums = {attr: 0.0 for attr in ATTRIBUTE_VOCABULARY}
    count = 0
    
    for img in images:
        # Only consider outliers (liked images)
        if get_label(img) != 1:
            continue
        
        # Get image URL (for cache key)
        image_url = get_image_url(img)
        if not image_url:
            continue
        
        # Get image attachment data
        image_data = get_image_attachment_data(img)
        
        # Extract attributes (uses cache)
        try:
            attributes = extract_attributes_gpt4v(image_url, use_cache=True, image_data=image_data)
            
            for attr, score in attributes.items():
                if attr in attribute_sums:
                    attribute_sums[attr] += score
            
            count += 1
            
        except Exception as e:
            logger.warning(f"Error extracting attributes for global pref: {e}")
            continue
    
    # Average
    if count > 0:
        global_pref = {attr: score / count for attr, score in attribute_sums.items()}
    else:
        global_pref = {attr: 0.5 for attr in ATTRIBUTE_VOCABULARY}
    
    # Sort by score descending
    global_pref = dict(sorted(global_pref.items(), key=lambda x: x[1], reverse=True))
    
    logger.info(f"Computed global preference from {count} outlier images")
    logger.info(f"Top preferences: {list(global_pref.items())[:5]}")
    
    return global_pref


def compute_prompt_success_rates(sample_metadata: List[Dict]) -> Dict[str, Dict]:
    """
    Aggregate success rates by prompt_hash AND by structure.
    
    Args:
        sample_metadata: List of sample metadata dicts from training
        
    Returns:
        Tuple of:
        - prompt_stats: Dict mapping prompt_hash to success stats
        - structure_prompt_stats: Dict mapping structure_id to top prompts for that structure
    """
    from collections import defaultdict
    
    prompt_stats = defaultdict(lambda: {
        "total": 0, 
        "positive": 0, 
        "prompt_preview": "",
        "structure_ids": set()
    })
    
    # Track by structure → prompt
    structure_prompt_stats = defaultdict(lambda: defaultdict(lambda: {
        "total": 0,
        "positive": 0,
        "prompt_preview": ""
    }))
    
    for sample in sample_metadata:
        ph = sample.get("prompt_hash")
        struct_id = sample.get("structure_id")
        label = sample.get("label", 0)
        preview = sample.get("prompt_preview", "")
        
        if not ph:
            continue
        prompt_stats[ph]["total"] += 1
        prompt_stats[ph]["positive"] += label
        prompt_stats[ph]["prompt_preview"] = preview
        if struct_id:
            prompt_stats[ph]["structure_ids"].add(struct_id)
        
        if struct_id:
            structure_prompt_stats[struct_id][ph]["total"] += 1
            structure_prompt_stats[struct_id][ph]["positive"] += label
            structure_prompt_stats[struct_id][ph]["prompt_preview"] = preview
    
    # Compute success rate and format results
    prompt_results = {}
    for ph, stats in prompt_stats.items():
        if stats["total"] >= 2:  # Only include prompts with 2+ images
            prompt_results[ph] = {
                "success_rate": round(stats["positive"] / stats["total"], 4),
                "sample_count": stats["total"],
                "positive_count": stats["positive"],
                "prompt_preview": stats["prompt_preview"],
                "structure_ids": list(stats["structure_ids"])
            }
    
    # Structure-level aggregation
    structure_results = {}
    for struct_id, prompts in structure_prompt_stats.items():
        struct_prompts = []
        for ph, stats in prompts.items():
            if stats["total"] >= 2:
                struct_prompts.append({
                    "prompt_hash": ph,
                    "prompt_preview": stats["prompt_preview"],
                    "success_rate": round(stats["positive"] / stats["total"], 4),
                    "sample_count": stats["total"]
                })
        
        if struct_prompts:
            struct_prompts.sort(key=lambda x: x["success_rate"], reverse=True)
            all_rates = [p["success_rate"] for p in struct_prompts]
            structure_results[str(struct_id)] = {
                "top_prompts": struct_prompts[:5],
                "avg_success_rate": round(sum(all_rates) / len(all_rates), 4) if all_rates else 0
            }
    
    # Sort by success_rate descending
    prompt_results = dict(sorted(prompt_results.items(), key=lambda x: x[1]["success_rate"], reverse=True))
    
    return prompt_results, structure_results


def build_training_data(
    images: List[Dict],
    structures_by_id: Dict[int, Dict],
    structures_by_record_id: Optional[Dict[str, Dict]] = None,
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Build training data from images and structures.
    
    Args:
        images: List of image records
        structures_by_id: Dict mapping structure ID to structure record
        max_samples: Optional limit on number of samples
        
    Returns:
        Tuple of (X features array, y labels array, sample metadata list)
    """
    logger.info(f"Building training data from {len(images)} images...")
    
    X = []
    y = []
    metadata = []
    
    skipped_no_url = 0
    skipped_no_structure = 0
    processed = 0
    
    for i, img in enumerate(images):
        if max_samples and processed >= max_samples:
            break
        
        # Get image URL
        image_url = get_image_url(img)
        if not image_url:
            skipped_no_url += 1
            continue
        
        # Get structure by numeric ID
        struct_id = get_structure_id(img)
        structure = structures_by_id.get(struct_id) if struct_id is not None else None

        # Fallback: if images table uses linked record IDs, map via record_id
        if structure is None and structures_by_record_id:
            raw_struct_field = img.get(ImageFields.STRUCTURE_ID)
            linked_id = raw_struct_field[0] if isinstance(raw_struct_field, list) and raw_struct_field else None
            if isinstance(linked_id, str):
                structure = structures_by_record_id.get(linked_id)
                if structure:
                    try:
                        struct_id = int(float(structure.get(StructureFields.STRUCTURE_ID)))
                    except Exception:
                        struct_id = None
        
        if structure is None or struct_id is None:
            skipped_no_structure += 1
            continue
        
        # Get label
        label = get_label(img)
        
        # Extract features
        try:
            prompt_text = extract_prompt_text(img)
            skeleton_text = structure.get(StructureFields.SKELETON, "") or ""
            struct_metadata = get_structure_metadata(structure)
            
            features = extract_all_features(
                image_url=image_url,
                prompt_text=prompt_text,
                skeleton_text=skeleton_text,
                structure_metadata=struct_metadata,
                use_cache=True
            )
            
            X.append(features)
            y.append(label)
            # Generate prompt hash for prompt-level tracking
            prompt_hash = hashlib.md5(prompt_text.encode()).hexdigest()[:12] if prompt_text else None
            
            metadata.append({
                "image_name": img.get(ImageFields.NAME, ""),
                "structure_id": struct_id,
                "prompt_hash": prompt_hash,
                "prompt_preview": prompt_text[:150] if prompt_text else "",
                "label": label
            })
            
            processed += 1
            
            if processed % 100 == 0:
                logger.info(f"Processed {processed} images...")
                
        except Exception as e:
            logger.warning(f"Error processing image {img.get(ImageFields.NAME, 'unknown')}: {e} — using zero features")
            X.append(np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32))
            y.append(label)
            # Generate prompt hash even for failed extractions
            prompt_text = extract_prompt_text(img)
            prompt_hash = hashlib.md5(prompt_text.encode()).hexdigest()[:12] if prompt_text else None
            
            metadata.append({
                "image_name": img.get(ImageFields.NAME, ""),
                "structure_id": struct_id,
                "prompt_hash": prompt_hash,
                "prompt_preview": prompt_text[:150] if prompt_text else "",
                "label": label,
                "warning": str(e)
            })
            processed += 1
    
    logger.info(f"Training data built:")
    logger.info(f"  - Processed: {processed}")
    logger.info(f"  - Skipped (no URL): {skipped_no_url}")
    logger.info(f"  - Skipped (no structure): {skipped_no_structure}")
    
    if len(X) == 0:
        raise ValueError("No valid training samples found")
    
    X = np.array(X)
    y = np.array(y)
    
    positive_count = np.sum(y == 1)
    negative_count = np.sum(y == 0)
    logger.info(f"  - Positive samples: {positive_count}")
    logger.info(f"  - Negative samples: {negative_count}")
    
    return X, y, metadata


def train_xgboost_model(
    X: np.ndarray, 
    y: np.ndarray
) -> Tuple[xgb.XGBClassifier, StandardScaler, Dict[str, Any]]:
    """
    Train XGBoost classifier.
    
    Args:
        X: Feature matrix
        y: Label vector
        
    Returns:
        Tuple of (trained model, scaler, metrics dict)
    """
    logger.info("Training XGBoost model...")
    
    # Compute class weights
    positive_count = np.sum(y == 1)
    negative_count = np.sum(y == 0)
    scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0
    
    logger.info(f"Class balance - scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Scale features (only metadata columns, leave embeddings normalized)
    scaler = StandardScaler()
    
    # Split into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Fit scaler on training data
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create model with updated params
    params = XGB_PARAMS.copy()
    params["scale_pos_weight"] = scale_pos_weight
    
    # Remove early_stopping_rounds from params (passed to fit instead)
    early_stopping = params.pop("early_stopping_rounds", 10)
    
    model = xgb.XGBClassifier(**params)
    
    # Train
    model.fit(
        X_train_scaled, 
        y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=True
    )
    
    # Evaluate
    y_pred = model.predict(X_val_scaled)
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    
    metrics = {
        "auc_score": float(roc_auc_score(y_val, y_pred_proba)),
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "train_samples": len(y_train),
        "val_samples": len(y_val),
        "positive_samples": int(positive_count),
        "negative_samples": int(negative_count)
    }
    
    logger.info(f"Training metrics:")
    for key, value in metrics.items():
        logger.info(f"  - {key}: {value}")
    
    return model, scaler, metrics


def save_model(
    model: xgb.XGBClassifier,
    scaler: StandardScaler,
    metrics: Dict[str, Any],
    global_pref: Dict[str, float],
    prompt_stats: Dict[str, Dict],
    structure_prompt_stats: Dict[str, Dict]
):
    """
    Save trained model and associated data.
    
    Args:
        model: Trained XGBoost model
        scaler: Fitted StandardScaler
        metrics: Training metrics
        global_pref: Global preference vector
    """
    ensure_model_dir()
    
    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {MODEL_PATH}")
    
    # Save scaler
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {SCALER_PATH}")
    
    # Save metadata
    metadata = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics,
        "feature_importance": get_feature_importance(model)
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {METADATA_PATH}")
    
    # Save global preference vector
    with open(GLOBAL_PREF_PATH, "w") as f:
        json.dump(global_pref, f, indent=2)
    logger.info(f"Global preference vector saved to {GLOBAL_PREF_PATH}")

    # Save prompt success rates
    from config import PROMPT_STATS_PATH, STRUCTURE_PROMPT_STATS_PATH
    with open(PROMPT_STATS_PATH, "w") as f:
        json.dump(prompt_stats, f, indent=2)
    logger.info(f"Prompt success rates saved to {PROMPT_STATS_PATH}")
    
    # Save structure-specific prompt insights
    with open(STRUCTURE_PROMPT_STATS_PATH, "w") as f:
        json.dump(structure_prompt_stats, f, indent=2)
    logger.info(f"Structure prompt insights saved to {STRUCTURE_PROMPT_STATS_PATH}")


def get_feature_importance(model: xgb.XGBClassifier, top_n: int = 20) -> List[Dict]:
    """
    Get top feature importances from model.
    
    Args:
        model: Trained model
        top_n: Number of top features to return
        
    Returns:
        List of dicts with feature index and importance
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    return [
        {"feature_index": int(idx), "importance": float(importances[idx])}
        for idx in indices
    ]


def train_model() -> Dict[str, Any]:
    """
    Main training function. Fetches data, trains model, saves results.
    
    Returns:
        Dict with training results and status
    """
    logger.info("=" * 60)
    logger.info("Starting model training...")
    logger.info("=" * 60)
    
    try:
        # Fetch data from Airtable
        client = AirtableClient()
        images, structures_by_id = client.fetch_training_data()
        structures_by_record_id = {
            s.get("_record_id"): s for s in structures_by_id.values() if s.get("_record_id")
        }
        
        logger.info(f"Fetched {len(images)} images and {len(structures_by_id)} structures")
        
        # Check minimum samples
        if len(images) < MIN_TRAINING_SAMPLES:
            return {
                "status": "skipped",
                "reason": f"Not enough samples ({len(images)} < {MIN_TRAINING_SAMPLES})",
                "training_samples": len(images)
            }
        
        # Sort newest-first and cap to MAX_TRAINING_SAMPLES
        images_sorted = sort_images_by_created(images)
        if MAX_TRAINING_SAMPLES:
            images_sorted = images_sorted[:MAX_TRAINING_SAMPLES]
            logger.info(f"Using {len(images_sorted)} most recent images for training (cap={MAX_TRAINING_SAMPLES})")
        
        # Build training data
        X, y, sample_metadata = build_training_data(
            images_sorted, structures_by_id, structures_by_record_id, max_samples=MAX_TRAINING_SAMPLES
        )
        
        # Check minimum positive samples
        positive_count = np.sum(y == 1)
        if positive_count < MIN_POSITIVE_SAMPLES:
            return {
                "status": "skipped",
                "reason": f"Not enough positive samples ({positive_count} < {MIN_POSITIVE_SAMPLES})",
                "positive_samples": int(positive_count)
            }
        
        # Compute global preference vector
        global_pref = compute_global_preference_vector(images_sorted, structures_by_id)
        
        # Compute prompt-level success rates (both views)
        prompt_stats, structure_prompt_stats = compute_prompt_success_rates(sample_metadata)
        logger.info(f"Computed success rates for {len(prompt_stats)} unique prompts across {len(structure_prompt_stats)} structures")
        
        # Train model
        model, scaler, metrics = train_xgboost_model(X, y)
        
        # Save everything
        save_model(model, scaler, metrics, global_pref, prompt_stats, structure_prompt_stats)
        
        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info("=" * 60)
        
        return {
            "status": "success",
            "training_samples": int(len(y)),
            "positive_samples": int(positive_count),
            "negative_samples": int(len(y) - positive_count),
            "auc_score": metrics["auc_score"],
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "model_path": MODEL_PATH
        }
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    result = train_model()
    print(json.dumps(result, indent=2))
