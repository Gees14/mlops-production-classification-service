"""Tests for API health and model-info endpoints (model may not be loaded)."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_has_status_field():
    response = client.get("/health")
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_health_has_model_loaded_field():
    response = client.get("/health")
    data = response.json()
    assert "model_loaded" in data


def test_docs_endpoint_accessible():
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema_accessible():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "paths" in data
