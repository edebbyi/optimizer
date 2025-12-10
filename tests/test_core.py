import numpy as np
import pytest

import airtable_client
import feature_extraction
import inference
import train
from train import sort_images_by_created
from config import (
    ATTRIBUTE_DIM,
    ATTRIBUTE_TO_INDEX,
    CLIP_EMBEDDING_DIM,
    METADATA_DIM,
    TEXT_EMBEDDING_DIM,
    TOTAL_FEATURE_DIM,
    ImageFields,
    StructureFields,
)


def test_get_structure_id_parses_numeric_strings():
    record = {ImageFields.STRUCTURE_ID: "42.0"}
    assert airtable_client.get_structure_id(record) == 42

    record[ImageFields.STRUCTURE_ID] = None
    assert airtable_client.get_structure_id(record) is None


def test_get_structure_id_handles_list_and_dict():
    record = {ImageFields.STRUCTURE_ID: ["7"]}
    assert airtable_client.get_structure_id(record) == 7

    record = {ImageFields.STRUCTURE_ID: [{"Structure ID": "8"}]}
    assert airtable_client.get_structure_id(record) == 8


def test_get_label_outlier_case_insensitive():
    assert airtable_client.get_label({ImageFields.STATUS: "Outlier"}) == 1
    assert airtable_client.get_label({ImageFields.STATUS: "outlier"}) == 1
    assert airtable_client.get_label({ImageFields.STATUS: "Pass"}) == 0
    assert airtable_client.get_label({ImageFields.STATUS: ""}) == 0


def test_get_image_url_uses_attachment_list():
    record = {
        ImageFields.IMAGE_URL: "https://expired.com/old.jpg",
        "Image": [{"url": "https://example.com/img.jpg"}],
    }
    assert airtable_client.get_image_url(record) == "https://example.com/img.jpg"


@pytest.mark.parametrize(
    "value,expected",
    [("8/10", 8.0), ("7", 7.0), ("", 5.0), ("abc", 5.0)],
)
def test_parse_ai_score(value, expected):
    assert airtable_client.parse_ai_score(value) == expected


def test_attributes_to_vector_sets_known_indices():
    attributes = {"oversized": 0.9, "unknown_attr": 1.0}
    vector = feature_extraction.attributes_to_vector(attributes)

    assert vector.shape == (ATTRIBUTE_DIM,)
    oversized_idx = ATTRIBUTE_TO_INDEX["oversized"]
    assert np.isclose(vector[oversized_idx], 0.9)
    # Unknown attributes should remain zero
    assert np.sum(vector) == pytest.approx(0.9, rel=1e-6)


def test_extract_structure_proxy_features_uses_metadata(monkeypatch):
    # Avoid loading real models
    monkeypatch.setattr(
        feature_extraction,
        "extract_text_embedding",
        lambda text: np.ones(TEXT_EMBEDDING_DIM, dtype=np.float32),
    )

    global_pref = {"tailored": 0.8}
    struct_meta = {
        "usage_count": 10,
        "outlier_count": 3,
        "age_weeks": 4.5,
        "z_score": 1.2,
        "ai_score": 6.0,
        "renderer": "Recraft",
        "structure_id": 77,
    }

    features = feature_extraction.extract_structure_proxy_features(
        skeleton_text="sample skeleton",
        structure_metadata=struct_meta,
        global_pref_vector=global_pref,
    )

    assert features.shape == (TOTAL_FEATURE_DIM,)
    # First CLIP segment should be zeros by design
    assert np.allclose(features[:CLIP_EMBEDDING_DIM], 0.0)
    # Text embeddings are ones from the monkeypatch
    text_segment = features[CLIP_EMBEDDING_DIM : CLIP_EMBEDDING_DIM + TEXT_EMBEDDING_DIM]
    assert np.allclose(text_segment, 1.0)
    # Attribute segment should include tailored value
    attr_start = CLIP_EMBEDDING_DIM + TEXT_EMBEDDING_DIM
    tailored_idx = ATTRIBUTE_TO_INDEX["tailored"]
    assert np.isclose(features[attr_start + tailored_idx], 0.8)
    # Metadata sits at the end
    metadata = features[-METADATA_DIM:]
    expected_metadata = np.array(
        [
            struct_meta["usage_count"],
            struct_meta["outlier_count"],
            struct_meta["age_weeks"],
            struct_meta["z_score"],
            struct_meta["ai_score"],
            1,  # Recraft encoder
            struct_meta["structure_id"],
        ],
        dtype=np.float32,
    )
    assert np.allclose(metadata, expected_metadata)


def test_build_training_data_skips_missing_structure(monkeypatch):
    # Return a deterministic feature vector to avoid heavy extraction
    monkeypatch.setattr(
        train,
        "extract_all_features",
        lambda **kwargs: np.ones(TOTAL_FEATURE_DIM, dtype=np.float32),
    )

    structures_by_id = {
        1: {
            StructureFields.STRUCTURE_ID: 1,
            StructureFields.SKELETON: "skeleton",
            StructureFields.RENDERER: "ImageFX",
            "_record_id": "rec123",
        }
    }
    structures_by_record_id = {"rec123": structures_by_id[1]}

    images = [
        {
            ImageFields.IMAGE_URL: "https://example.com/img1.jpg",
            ImageFields.STRUCTURE_ID: 1,
            ImageFields.PROMPT: "prompt",
            ImageFields.STATUS: "Outlier",
            ImageFields.NAME: "img1",
        },
        {
            ImageFields.IMAGE_URL: "https://example.com/img2.jpg",
            ImageFields.STRUCTURE_ID: None,
            ImageFields.STATUS: "Pass",
        },
        {
            ImageFields.IMAGE_URL: "https://example.com/img3.jpg",
            ImageFields.STRUCTURE_ID: 999,  # not present in structures_by_id
            ImageFields.STATUS: "Pass",
        },
        {
            ImageFields.IMAGE_URL: "https://example.com/img4.jpg",
            ImageFields.STRUCTURE_ID: ["rec123"],  # linked record id
            ImageFields.STATUS: "Outlier",
        },
    ]

    X, y, metadata = train.build_training_data(
        images, structures_by_id, structures_by_record_id
    )

    assert X.shape == (2, TOTAL_FEATURE_DIM)
    assert y.tolist() == [1, 1]  # Two valid outliers processed
    assert len(metadata) == 2
    assert metadata[0]["structure_id"] == 1
    assert metadata[1]["structure_id"] == 1


def test_sort_images_by_created_orders_descending():
    images = [
        {ImageFields.CREATED: "2025-01-01T00:00:00.000Z", "name": "old"},
        {ImageFields.CREATED: "2025-02-01T00:00:00.000Z", "name": "new"},
        {ImageFields.CREATED: None, "name": "no_date"},
    ]

    ordered = sort_images_by_created(images)
    names = [img["name"] for img in ordered]
    assert names[0] == "new"
    assert names[-1] == "no_date"


def test_train_model_skips_when_not_enough_samples(monkeypatch):
    class DummyClient:
        def __init__(self):
            pass

        def fetch_training_data(self):
            return ([{}] * 5, {})

    monkeypatch.setattr(train, "AirtableClient", DummyClient)
    monkeypatch.setattr(train, "MIN_TRAINING_SAMPLES", 10)

    result = train.train_model()

    assert result["status"] == "skipped"
    assert "Not enough samples" in result["reason"]


def test_score_structures_returns_defaults_without_model(monkeypatch):
    class DummyAirtable:
        def __init__(self):
            pass

        def fetch_scoring_data(self):
            return [
                {
                    StructureFields.STRUCTURE_ID: 123,
                    StructureFields.OUTLIER_COUNT: 2,
                    StructureFields.Z_SCORE: 0.5,
                    StructureFields.SKELETON: "template",
                    StructureFields.RENDERER: "ImageFX",
                }
            ]

    monkeypatch.setattr(inference, "AirtableClient", DummyAirtable)
    monkeypatch.setattr(inference, "is_model_loaded", lambda: False)

    result = inference.score_structures()

    assert result["warning"]
    assert result["structures"][0]["predicted_success_score"] == 0.5
    assert result["structures"][0]["structure_id"] == 123
    assert result["global_preference_vector"] == {}
