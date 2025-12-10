"""
Test GPT-4V with Airtable Attachment Data
"""

import logging
from airtable_client import AirtableClient, get_image_url, get_image_attachment_data
from feature_extraction import extract_attributes_gpt4v
from train import sort_images_by_created

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

print("=" * 60)
print("Testing GPT-4V with Airtable Attachment Data")
print("=" * 60)

# Fetch recent images
client = AirtableClient()
images, _ = client.fetch_training_data()

# Get most recent image with Image field populated
images_sorted = sort_images_by_created(images)

test_image = None
for img in images_sorted[:50]:  # Check first 50 recent images
    if img.get("Image"):  # Has Image attachment field
        test_image = img
        break

if not test_image:
    print("✗ No images found with Image attachment field!")
    exit(1)

print(f"\nTest Image: {test_image.get('Name', 'Unknown')}")
image_url = get_image_url(test_image)
print(f"Image URL: {image_url}")

# Get attachment data
print("\nFetching attachment data...")
image_data = get_image_attachment_data(test_image)

if not image_data:
    print("✗ Could not fetch attachment data!")
    exit(1)

print(f"✓ Downloaded {len(image_data)} bytes")

# Extract attributes
print("\nExtracting attributes with GPT-4V...")
try:
    attributes = extract_attributes_gpt4v(
        image_url=image_url,
        use_cache=False,
        max_retries=1,
        image_data=image_data
    )
    
    non_zero = sum(1 for v in attributes.values() if v > 0.1)
    all_zero = all(v == 0.0 for v in attributes.values())
    
    print("\n" + "=" * 60)
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
        
        print("\n✓ SUCCESS! GPT-4V is now working with Airtable attachments!")
        print("\nYou can now run training:")
        print("  python train.py")
    else:
        print("\n✗ FAILED: All attributes are zero")
        print("Check debug logs in: cache/attributes/debug/")
        
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
