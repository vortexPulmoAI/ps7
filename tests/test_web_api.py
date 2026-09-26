"""
Tests for FastAPI Web Application endpoints and inference pipeline.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from web.app import app

client = TestClient(app)


def test_index_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "VORTEX PULMO-AI" in response.text
    assert "3D Nodule Surface Morphology" in response.text


def test_api_samples():
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) > 0
    first_sample = data["samples"][0]
    assert "series_uid" in first_sample
    assert "primary_diameter_mm" in first_sample


def test_analyze_with_sample_id():
    response = client.post(
        "/api/analyze",
        data={
            "sample_id": 0,
            "age": 65.0,
            "is_female": False,
            "is_upper_lobe": True,
            "family_history": False,
            "emphysema": True
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    
    # Clinical Triad
    triad = data["triad"]
    assert "risk_percent" in triad
    assert "brock_percent" in triad
    assert "lung_rads_category" in triad
    assert "confidence_flag" in triad

    # Morphology & Density
    assert data["morphology"]["volume_mm3"] > 0
    assert data["morphology"]["d_mean_mm"] > 0
    assert data["density"]["nodule_type"] in ["solid", "part_solid", "pure_ggn", "calcified"]

    # Treatment Pathway
    assert "treatment" in data
    assert "urgency_level" in data["treatment"]
    assert "primary_action" in data["treatment"]
    assert len(data["treatment"]["actionable_pathway"]) > 0

    # Diagnostic Visualizations & 3D Mesh
    assert "axial_slice" in data["images"]
    assert "segmentation_overlay" in data["images"]
    assert "mesh_3d" in data
    assert "vertices" in data["mesh_3d"]


def test_analyze_with_uploaded_raw():
    raw_path = os.path.join(BASE_DIR, "124.raw")
    if os.path.exists(raw_path):
        with open(raw_path, "rb") as f:
            file_bytes = f.read()
        
        response = client.post(
            "/api/analyze",
            files={"file": ("124.raw", file_bytes, "application/octet-stream")},
            data={"age": 60.0}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "triad" in data
        assert "treatment" in data
