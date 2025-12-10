"""
Quick test of base64 image encoding for GPT-4V
"""

import logging
from feature_extraction import extract_attributes_gpt4v
from config import ATTRIBUTE_VOCABULARY

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Use a stable Airtable CDN URL (not Replicate delivery)
# This should be from the Image attachment field in Airtable
test_url = "https://dl.airtable.com/.attachments/aeda6cef62d04c4ec18259f53d12a1dd/4ce24ca9/IMG_0095.jpeg"

print("=" * 60)
print("Testing GPT-4V with Base64 Image Encoding")
print("=" * 60)
print(f"Test URL: {test_url}")
print()

try:
    attributes = extract_attributes_gpt4v(test_url, use_cache=False, max_retries=1)
    
    non_zero = sum(1 for v in attributes.values() if v > 0.1)
    all_zero = all(v == 0.0 for v in attributes.values())
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print("=" * 60)
    print(f"Total attributes: {len(attributes)}")
    print(f"Non-zero attributes (>0.1): {non_zero}")
    print(f"All zero: {all_zero}")
    
    if not all_zero:
        # Show top attributes
        top_attrs = sorted(attributes.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\nTop 10 attributes:")
        for attr, score in top_attrs:
            print(f"  {attr}: {score:.2f}")
        print("\n✓ SUCCESS: GPT-4V extracted attributes!")
    else:
        print("\n✗ FAILED: All attributes are zero")
        
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
