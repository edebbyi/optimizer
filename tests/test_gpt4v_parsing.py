"""
Test GPT-4V Attribute Extraction with Improved Parsing

This script tests the improved GPT-4V parsing on a small batch of images
to verify that attributes are being extracted correctly.
"""

import os
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from airtable_client import AirtableClient
from feature_extraction import (
    extract_attributes_gpt4v,
    get_cache_stats
)
from config import ImageFields, ATTRIBUTE_VOCABULARY

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_gpt4v_extraction(num_images: int = 10):
    """
    Test GPT-4V extraction on recent images.
    
    Args:
        num_images: Number of images to test (default 10)
    """
    logger.info("=" * 60)
    logger.info(f"Testing GPT-4V Extraction on {num_images} Recent Images")
    logger.info("=" * 60)
    
    # Get cache stats before
    cache_before = get_cache_stats()
    logger.info(f"Cache before: {cache_before}")
    
    # Fetch images from Airtable
    client = AirtableClient()
    images, structures_by_id = client.fetch_training_data()
    
    # Sort by created date (newest first) and take subset
    from train import sort_images_by_created
    images_sorted = sort_images_by_created(images)[:num_images]
    
    logger.info(f"Testing on {len(images_sorted)} images...")
    logger.info("")
    
    # Test extraction on each image
    results = []
    successes = 0
    failures = 0
    
    for i, img in enumerate(images_sorted):
        image_name = img.get(ImageFields.NAME, "unknown")
        image_url = img.get(ImageFields.IMAGE_URL)
        
        if not image_url:
            logger.warning(f"[{i+1}/{len(images_sorted)}] {image_name}: No URL, skipping")
            continue
        
        logger.info(f"[{i+1}/{len(images_sorted)}] Processing: {image_name}")
        logger.info(f"  URL: {image_url}")
        
        try:
            # Extract attributes (no cache to force new extraction)
            attributes = extract_attributes_gpt4v(image_url, use_cache=False, max_retries=2)
            
            # Analyze results
            non_zero_count = sum(1 for v in attributes.values() if v > 0.1)
            all_zero = all(v == 0.0 for v in attributes.values())
            
            # Get top attributes
            top_attrs = sorted(attributes.items(), key=lambda x: x[1], reverse=True)[:5]
            
            result = {
                "image_name": image_name,
                "image_url": image_url,
                "non_zero_attributes": non_zero_count,
                "all_zero": all_zero,
                "top_attributes": top_attrs,
                "success": not all_zero and non_zero_count > 0
            }
            results.append(result)
            
            if result["success"]:
                successes += 1
                logger.info(f"  ✓ SUCCESS: {non_zero_count}/{len(ATTRIBUTE_VOCABULARY)} attributes extracted")
                logger.info(f"  Top 5: {', '.join(f'{k}={v:.2f}' for k, v in top_attrs)}")
            else:
                failures += 1
                logger.warning(f"  ✗ FAILED: All attributes are zero")
            
            logger.info("")
            
        except Exception as e:
            logger.error(f"  ✗ ERROR: {e}")
            results.append({
                "image_name": image_name,
                "image_url": image_url,
                "error": str(e),
                "success": False
            })
            failures += 1
            logger.info("")
    
    # Get cache stats after
    cache_after = get_cache_stats()
    logger.info("=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Images tested: {len(results)}")
    logger.info(f"Successes: {successes}")
    logger.info(f"Failures: {failures}")
    logger.info(f"Success rate: {successes / len(results) * 100:.1f}%" if results else "N/A")
    logger.info("")
    logger.info(f"Cache before: {cache_before}")
    logger.info(f"Cache after: {cache_after}")
    logger.info(f"New cached attributes: {cache_after['gpt4v_attributes'] - cache_before['gpt4v_attributes']}")
    logger.info("")
    
    # Check debug logs
    debug_dir = "cache/attributes/debug"
    if os.path.exists(debug_dir):
        debug_files = [f for f in os.listdir(debug_dir) if f.endswith('.txt')]
        logger.info(f"Debug logs saved: {len(debug_files)} files in {debug_dir}")
        logger.info("You can inspect these files to see raw GPT-4V responses")
    
    logger.info("=" * 60)
    
    # Save summary
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_file = results_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_images": len(results),
            "successes": successes,
            "failures": failures,
            "cache_before": cache_before,
            "cache_after": cache_after,
            "results": results
        }, f, indent=2)
    
    logger.info(f"Summary saved to: {summary_file}")
    
    # Provide recommendation
    if successes > 0:
        logger.info("")
        logger.info("✓ RECOMMENDATION: GPT-4V parsing is working!")
        logger.info("  You can now re-run training to get real attribute metadata:")
        logger.info("  $ python train.py")
    else:
        logger.info("")
        logger.info("✗ RECOMMENDATION: GPT-4V parsing still failing")
        logger.info(f"  Check debug logs in: {debug_dir}")
        logger.info("  Review raw responses to understand the issue")


if __name__ == "__main__":
    import sys
    
    # Allow specifying number of images as command line arg
    num_images = 10
    if len(sys.argv) > 1:
        try:
            num_images = int(sys.argv[1])
        except ValueError:
            logger.warning(f"Invalid number: {sys.argv[1]}, using default: 10")
    
    test_gpt4v_extraction(num_images)
